"""Temporal model development and validation-selected model bundle."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from riskshield.config.settings import SETTINGS
from riskshield.evaluation.metrics import metrics
from riskshield.evaluation.thresholds import action_cost, optimize
from riskshield.features.engineering import FEATURE_NAMES, build_features
from riskshield.models.calibration import fit_calibrator, predict_probability


def temporal_split(
    frame: pd.DataFrame, settings: dict = SETTINGS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.assign(timestamp=pd.to_datetime(frame.timestamp, utc=True))
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    a = int(len(frame) * settings["train_fraction"])
    b = int(len(frame) * (settings["train_fraction"] + settings["validation_fraction"]))
    parts = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
    for part in parts:
        if part.empty or part.fraud_label.nunique() < 2:
            raise ValueError("Each temporal partition needs both classes; generate more data")
    if not (
        parts[0].timestamp.max() < parts[1].timestamp.min()
        and parts[1].timestamp.max() < parts[2].timestamp.min()
    ):
        raise ValueError("Temporal boundaries overlap; timestamps must separate partitions")
    return parts


def fit_models(x: pd.DataFrame, y: np.ndarray) -> dict:
    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=SETTINGS["logistic_C"], max_iter=1000, random_state=SETTINGS["seed"]
                ),
            ),
        ]
    )
    candidate = XGBClassifier(random_state=SETTINGS["seed"], **SETTINGS["xgboost"])
    return {"logistic": baseline.fit(x, y), "xgboost": candidate.fit(x, y)}


def train(data_path: Path, artifact_dir: Path, report_dir: Path, track: bool = False) -> dict:
    frame = pd.read_csv(data_path)
    train_df, val_df, test_df = temporal_split(frame)
    cut = int(len(train_df) * (1 - SETTINGS["calibration_fraction"]))
    train_df, calibration_df = train_df.iloc[:cut], train_df.iloc[cut:]
    x, xv = build_features(train_df), build_features(val_df)
    models = fit_models(x, train_df.fraud_label.to_numpy())
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results, bundles = {}, {}
    for name, model in models.items():
        calibrator = fit_calibrator(
            model.predict_proba(build_features(calibration_df))[:, 1],
            calibration_df.fraud_label.to_numpy(),
        )
        provisional = {"model": model, "calibrator": calibrator}
        p = predict_probability(provisional, xv)
        policy, table = optimize(val_df.fraud_label.to_numpy(), p)
        table.to_csv(report_dir / f"{name}_thresholds.csv", index=False)
        results[name] = {
            "validation": metrics(val_df.fraud_label.to_numpy(), p, policy.review_threshold),
            "policy": asdict(policy),
            "validation_action_cost": action_cost(val_df.fraud_label.to_numpy(), p, policy),
            "uncalibrated_validation": metrics(
                val_df.fraud_label.to_numpy(),
                model.predict_proba(xv)[:, 1],
                policy.review_threshold,
            ),
        }
        bundles[name] = {
            "model": model,
            "calibrator": calibrator,
            "kind": name,
            "policy": policy,
            "feature_names": FEATURE_NAMES,
            "version": SETTINGS["model_version"],
        }
    # Model selection cannot inspect test scores.
    selected = min(results, key=lambda name: results[name]["validation_action_cost"])
    metadata = {
        "selected_model": selected,
        "selection_rule": "minimum validation three-action cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "settings": SETTINGS,
        "feature_names": FEATURE_NAMES,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]
        },
        "splits": {
            name: {
                "rows": len(part),
                "fraud_rate": float(part.fraud_label.mean()),
                "start": part.timestamp.min().isoformat(),
                "end": part.timestamp.max().isoformat(),
            }
            for name, part in zip(
                ["train", "calibration", "validation", "test"],
                [train_df, calibration_df, val_df, test_df],
                strict=True,
            )
        },
        "models": results,
    }
    for name, bundle in bundles.items():
        bundle["metadata"] = metadata
        joblib.dump(bundle, artifact_dir / f"{name}.joblib")
    joblib.dump(bundles[selected], artifact_dir / "selected.joblib")
    (report_dir / "training.json").write_text(json.dumps(metadata, indent=2))
    if track:
        import mlflow

        mlflow.set_tracking_uri((artifact_dir / "mlruns").resolve().as_uri())
        mlflow.set_experiment("riskshield-synthetic")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": selected,
                    **asdict(bundles[selected]["policy"]),
                    "seed": SETTINGS["seed"],
                    "model_parameters": json.dumps(models[selected].get_params(), default=str),
                }
            )
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in results[selected]["validation"].items()
                    if isinstance(v, (int, float))
                }
            )
            mlflow.log_metric("validation_action_cost", results[selected]["validation_action_cost"])
            mlflow.log_artifact(str(artifact_dir / "selected.joblib"))
            mlflow.log_artifact(str(report_dir / "training.json"))
    return metadata


def evaluate(data_path: Path, artifact_dir: Path, report_dir: Path) -> dict:
    selected = joblib.load(artifact_dir / "selected.joblib")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != selected["metadata"]["data_sha256"]:
        raise ValueError("Dataset differs from training fingerprint")
    saved_settings = selected["metadata"]["settings"]
    _, _, test = temporal_split(pd.read_csv(data_path), saved_settings)
    features = build_features(test)
    results = {}
    for name in ["logistic", "xgboost"]:
        bundle = joblib.load(artifact_dir / f"{name}.joblib")
        p = predict_probability(bundle, features)
        policy = bundle["policy"]
        results[name] = {
            **metrics(test.fraud_label.to_numpy(), p, policy.review_threshold),
            "policy": asdict(policy),
            "test_action_cost": action_cost(test.fraud_label.to_numpy(), p, policy, saved_settings),
            "decisions": {
                action: sum(policy.decide(float(v))[0] == action for v in p)
                for action in ["APPROVE", "REVIEW", "DECLINE"]
            },
        }
    (report_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    return results
