"""Monitor CSV feature windows, or run a labeled, deliberate-shift demo."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from riskshield.config.settings import SETTINGS
from riskshield.features.engineering import build_features
from riskshield.models.calibration import predict_probability
from riskshield.models.training import temporal_split
from riskshield.monitoring.drift import monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--reference-hours", type=float, default=1)
    parser.add_argument("--current-hours", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=SETTINGS["default_review_threshold"])
    args = parser.parse_args()
    if bool(args.reference) != bool(args.current):
        parser.error("Supply both --reference and --current")
    latencies = None
    if args.reference:
        reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
        rh, ch, threshold = args.reference_hours, args.current_hours, args.threshold
    else:
        _, validation, test = temporal_split(pd.read_csv("data/generated/transactions.csv"))
        bundle = joblib.load("artifacts/selected.joblib")
        reference, current = build_features(validation), build_features(test)
        reference["score"] = predict_probability(bundle, reference)
        current["score"] = predict_probability(bundle, current)
        current["fraud_label"] = test.fraud_label.to_numpy()
        # Inject telemetry drift after scoring; original scores remain paired with labels.
        current.loc[current.index[:500], "merchant_risk"] = np.nan
        current["amount_ratio"] *= 3
        latencies = []
        for i in range(30):
            start = perf_counter()
            predict_probability(bundle, build_features(test.iloc[i : i + 1]))
            latencies.append((perf_counter() - start) * 1000)
        rh = (validation.timestamp.max() - validation.timestamp.min()).total_seconds() / 3600
        ch = (test.timestamp.max() - test.timestamp.min()).total_seconds() / 3600
        threshold = bundle["policy"].review_threshold
    result = monitor(reference, current, rh, ch, threshold, latencies)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/monitoring.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Monitoring report",
        "",
        "Synthetic demo unless external CSV windows supplied.",
        "Demo latency covers feature creation and prediction, excluding SHAP and HTTP.",
        "Demo injects telemetry drift after scoring; performance is on original test scores.",
        "",
        f"Alerts: {', '.join(result['alerts']) or 'none'}",
        "",
        "| Signal | PSI | KS statistic | Missing delta |",
        "|---|---:|---:|---:|",
    ]
    for name, signal in result["signals"].items():
        lines.append(
            f"| {name} | {signal['psi']:.4f} | {signal['ks_statistic']} | "
            f"{signal['missing_delta']:.4f} |"
        )
    lines += [
        "",
        "```json",
        json.dumps({k: v for k, v in result.items() if k != "signals"}, indent=2),
        "```",
    ]
    Path("reports/monitoring.md").write_text("\n".join(lines))
    print("Monitoring alerts:", result["alerts"])
