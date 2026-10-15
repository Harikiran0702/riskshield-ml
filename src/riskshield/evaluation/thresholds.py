"""Validation-only cost optimization for binary screening and three-way actions."""

import numpy as np
import pandas as pd

from riskshield.config.settings import SETTINGS
from riskshield.decisioning.policy import Policy
from riskshield.evaluation.metrics import metrics


def action_cost(y: np.ndarray, p: np.ndarray, policy: Policy, settings: dict = SETTINGS) -> float:
    y, p = np.asarray(y), np.asarray(p)
    approve = p < policy.review_threshold
    review = (p >= policy.review_threshold) & (p < policy.decline_threshold)
    decline = p >= policy.decline_threshold
    return float(
        settings["fraud_cost"] * y[approve].sum()
        + settings["review_cost"] * review.sum()
        + settings["fraud_cost"] * settings["review_miss_rate"] * y[review].sum()
        + settings["false_decline_cost"] * (1 - y[decline]).sum()
    )


def optimize(
    y: np.ndarray, p: np.ndarray, settings: dict = SETTINGS
) -> tuple[Policy, pd.DataFrame]:
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0, 1, settings["threshold_steps"])
    rows = []
    for t in grid:
        m = metrics(y, p, float(t))
        (_, fp), (fn, _) = m["confusion_matrix"]
        rows.append(
            dict(
                threshold=float(t),
                precision=m["precision"],
                recall=m["recall"],
                false_positives=fp,
                false_negatives=fn,
                estimated_binary_cost=fn * settings["fraud_cost"]
                + int((p >= t).sum()) * settings["review_cost"],
            )
        )
    best = (float("inf"), 0.0, 1.0)
    for i, review in enumerate(grid[:-1]):
        for decline in grid[i + 1 :]:
            policy = Policy(float(review), float(decline))
            candidate = (action_cost(y, p, policy, settings), float(review), float(decline))
            if candidate < best:
                best = candidate
    return Policy(best[1], best[2]), pd.DataFrame(rows)
