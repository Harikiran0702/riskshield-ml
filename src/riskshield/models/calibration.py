"""Sigmoid calibration fitted on a disjoint later chronological window."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def logits(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration window needs both classes")
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logits(p), y)
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration would reverse ranking; more calibration evidence is needed")
    return calibrator


def predict_probability(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict_proba(features)[:, 1]
    calibrator = bundle.get("calibrator")
    return raw if calibrator is None else calibrator.predict_proba(logits(raw))[:, 1]
