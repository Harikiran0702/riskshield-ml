"""Reference-quantile PSI, missingness, rate, latency and delayed-label evaluation."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from riskshield.config.settings import SETTINGS
from riskshield.evaluation.metrics import metrics


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    reference, current = reference[np.isfinite(reference)], current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return 0.0  # Missingness is reported separately.
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(inner) == 1:
        inner = np.array([inner[0] - 1e-6, inner[0] + 1e-6])
    bins = np.r_[-np.inf, inner, np.inf]
    a = np.histogram(reference, bins)[0] + 0.5
    b = np.histogram(current, bins)[0] + 0.5
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def monitor(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_hours: float,
    current_hours: float,
    threshold: float,
    latency_ms: np.ndarray | None = None,
) -> dict:
    if reference_hours <= 0 or current_hours <= 0 or reference.empty or current.empty:
        raise ValueError("Nonempty windows and positive observation durations are required")
    if set(reference.columns) - {"fraud_label"} != set(current.columns) - {"fraud_label"}:
        raise ValueError("Monitoring schemas must match")
    signals, alerts = {}, []
    for col in reference.columns:
        if col == "fraud_label":
            continue
        r, c = reference[col], current[col]
        missing_delta = float(c.isna().mean() - r.isna().mean())
        value = psi(r.to_numpy(), c.to_numpy())
        ks = (
            float(ks_2samp(r.dropna(), c.dropna()).statistic)
            if r.notna().any() and c.notna().any()
            else None
        )
        signals[col] = {"psi": value, "ks_statistic": ks, "missing_delta": missing_delta}
        if value > SETTINGS["psi_alert"] or abs(missing_delta) > SETTINGS["missing_alert"]:
            alerts.append(col)
    rate_change = (len(current) / current_hours) / (len(reference) / reference_hours) - 1
    if abs(rate_change) > SETTINGS["volume_alert"]:
        alerts.append("transaction_volume")
    latency = None
    if latency_ms is not None and len(latency_ms):
        latency = {
            "p50_ms": float(np.percentile(latency_ms, 50)),
            "p95_ms": float(np.percentile(latency_ms, 95)),
        }
        if latency["p95_ms"] > SETTINGS["latency_alert_ms"]:
            alerts.append("latency")
    labeled = (
        current.dropna(subset=["fraud_label", "score"])
        if "fraud_label" in current
        else pd.DataFrame()
    )
    return {
        "signals": signals,
        "alerts": alerts,
        "volume_rate_change": float(rate_change),
        "latency": latency,
        "score_distribution": current["score"].describe().to_dict(),
        "label_coverage": len(labeled) / len(current),
        "eventual_performance": metrics(
            labeled.fraud_label.to_numpy(), labeled.score.to_numpy(), threshold
        )
        if len(labeled)
        else None,
    }
