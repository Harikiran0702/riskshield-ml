import json

import joblib
import numpy as np
import pytest
import xgboost as xgb

from riskshield.config.settings import SETTINGS
from riskshield.decisioning.policy import Policy
from riskshield.evaluation.metrics import metrics
from riskshield.evaluation.thresholds import action_cost, optimize
from riskshield.explainability.reasons import contributions, reason_codes
from riskshield.features.engineering import build_features
from riskshield.models.training import evaluate


def test_metrics_known_confusion():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_single_class_metrics():
    m = metrics(np.zeros(3), np.zeros(3), 0.5)
    assert m["roc_auc"] is None
    assert m["average_precision"] is None
    assert m["false_positive_rate"] == 0


@pytest.mark.parametrize(
    "p,decision",
    [
        (0, "APPROVE"),
        (0.199, "APPROVE"),
        (0.2, "REVIEW"),
        (0.799, "REVIEW"),
        (0.8, "DECLINE"),
        (1, "DECLINE"),
    ],
)
def test_policy_boundaries(p, decision):
    assert Policy(0.2, 0.8).decide(p)[0] == decision


@pytest.mark.parametrize("review,decline", [(0.8, 0.2), (-0.1, 0.8), (0.5, 0.5), (0.2, 1.1)])
def test_invalid_policy(review, decline):
    with pytest.raises(ValueError):
        Policy(review, decline)


def test_cost_matches_actions():
    # One missed fraud, one reviewed fraud, one reviewed genuine, one false decline.
    y, p = np.array([1, 1, 0, 0]), np.array([0.1, 0.4, 0.4, 0.9])
    assert action_cost(y, p, Policy(0.2, 0.8)) == 500 + 50 + 10 + 100


def test_optimizer_global_grid_minimum():
    y, p = np.array([0, 1, 0, 1, 1]), np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    settings = {**SETTINGS, "threshold_steps": 11}
    policy, table = optimize(y, p, settings)
    grid = np.linspace(0, 1, 11)
    assert action_cost(y, p, policy, settings) == min(
        action_cost(y, p, Policy(r, d), settings) for r in grid for d in grid if r < d
    )
    assert table.iloc[0].false_negatives == 0
    assert table.iloc[-1].false_negatives == y.sum()
    assert table.recall.is_monotonic_decreasing


def test_model_training_and_selection(trained, bundle, frame):
    assert set(bundle["metadata"]["models"]) == {"logistic", "xgboost"}
    results = bundle["metadata"]["models"]
    assert bundle["kind"] == min(results, key=lambda k: results[k]["validation_action_cost"])
    p = bundle["model"].predict_proba(build_features(frame.iloc[-10:]))[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    report = evaluate(trained / "transactions.csv", trained / "artifacts", trained / "reports")
    assert all(m["average_precision"] is not None for m in report.values())


def test_explanations_add_to_model_margin(trained, frame):
    features = build_features(frame.iloc[-5:])
    for kind in ["xgboost", "logistic"]:
        b = joblib.load(trained / f"artifacts/{kind}.joblib")
        c = contributions(b["model"], features, kind)
        if kind == "xgboost":
            raw = b["model"].get_booster().predict(xgb.DMatrix(features), pred_contribs=True)
            margin = b["model"].get_booster().predict(xgb.DMatrix(features), output_margin=True)
            np.testing.assert_allclose(c.sum(axis=1) + raw[:, -1], margin, rtol=1e-5)
        else:
            np.testing.assert_allclose(
                c.sum(axis=1) + b["model"].named_steps["classifier"].intercept_,
                b["model"].decision_function(features),
            )
        assert all(1 <= len(r) <= 3 for r in reason_codes(b["model"], features, kind))


def test_metadata_fingerprint(trained):
    metadata = json.loads((trained / "reports/training.json").read_text())
    assert len(metadata["data_sha256"]) == 64
    assert metadata["splits"]["train"]["end"] < metadata["splits"]["validation"]["start"]


def test_scaler_fits_training_only(trained, frame):
    from riskshield.models.training import temporal_split

    b = joblib.load(trained / "artifacts/logistic.joblib")
    train_frame, _, _ = temporal_split(frame)
    train_frame = train_frame.iloc[: int(len(train_frame) * (1 - SETTINGS["calibration_fraction"]))]
    np.testing.assert_allclose(
        b["model"].named_steps["scale"].mean_, build_features(train_frame).mean().to_numpy()
    )


def test_changed_evaluation_data_rejected(trained, tmp_path):
    modified = tmp_path / "modified.csv"
    modified.write_bytes((trained / "transactions.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate(modified, trained / "artifacts", trained / "reports")
