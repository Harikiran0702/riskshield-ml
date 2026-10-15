"""Rolling chronological backtests with disjoint test blocks and cluster uncertainty."""

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from riskshield.config.settings import SETTINGS
from riskshield.evaluation.metrics import metrics
from riskshield.evaluation.thresholds import action_cost, optimize
from riskshield.features.engineering import build_features
from riskshield.models.calibration import fit_calibrator, predict_probability
from riskshield.models.training import fit_models


def customer_bootstrap(
    y: np.ndarray, p: np.ndarray, customers: np.ndarray, repeats: int = 200
) -> dict:
    rng = np.random.default_rng(SETTINGS["seed"])
    unique = np.unique(customers)
    groups = {c: np.flatnonzero(customers == c) for c in unique}
    estimates = []
    for _ in range(repeats):
        indices = np.concatenate([groups[c] for c in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[indices])) == 2:
            estimates.append(average_precision_score(y[indices], p[indices]))
    if not estimates:
        raise ValueError("Not enough labeled classes for bootstrap")
    return {
        "ap_95_percentile_interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "replicates": len(estimates),
        "method": "customer-cluster percentile bootstrap",
    }


def rolling_backtest(frame: pd.DataFrame) -> list[dict]:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    block = len(frame) // 10
    results = []
    for fold in range(3):
        cut = (5 + fold) * block
        train, calibration, validation, test = (
            frame.iloc[:cut],
            frame.iloc[cut : cut + block],
            frame.iloc[cut + block : cut + 2 * block],
            frame.iloc[cut + 2 * block : cut + 3 * block],
        )
        models = fit_models(build_features(train), train.fraud_label.to_numpy())
        candidates = {}
        for name, model in models.items():
            calibrator = fit_calibrator(
                model.predict_proba(build_features(calibration))[:, 1],
                calibration.fraud_label.to_numpy(),
            )
            bundle = {"model": model, "calibrator": calibrator}
            pv = predict_probability(bundle, build_features(validation))
            policy, _ = optimize(validation.fraud_label.to_numpy(), pv)
            pt = predict_probability(bundle, build_features(test))
            candidates[name] = {
                "validation_cost": action_cost(validation.fraud_label.to_numpy(), pv, policy),
                "test_metrics": metrics(test.fraud_label.to_numpy(), pt, policy.review_threshold),
                "policy": asdict(policy),
            }
        selected = min(candidates, key=lambda name: candidates[name]["validation_cost"])
        results.append(
            {
                "fold": fold + 1,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_start": str(test.timestamp.min()),
                "test_end": str(test.timestamp.max()),
                "selected_by_validation": selected,
                "models": candidates,
            }
        )
    return results
