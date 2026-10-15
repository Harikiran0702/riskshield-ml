import numpy as np
import pytest

from riskshield.evaluation.robustness import customer_bootstrap
from riskshield.features.engineering import build_features
from riskshield.models.calibration import fit_calibrator, predict_probability


def test_calibration_is_monotone_and_persisted(bundle, frame):
    p = predict_probability(bundle, build_features(frame))
    raw = bundle["model"].predict_proba(build_features(frame))[:, 1]
    assert np.all(np.diff(p[np.argsort(raw)]) >= -1e-12)
    assert ((p > 0) & (p < 1)).all()
    assert bundle["calibrator"].coef_[0, 0] > 0
    splits = bundle["metadata"]["splits"]
    assert splits["train"]["end"] < splits["calibration"]["start"]
    assert splits["calibration"]["end"] < splits["validation"]["start"]


def test_calibration_rejects_single_class():
    with pytest.raises(ValueError):
        fit_calibrator(np.array([0.1, 0.2]), np.zeros(2))


def test_customer_bootstrap_reproducible():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.7, 0.2, 0.8, 0.3, 0.9])
    customers = np.array([1, 1, 2, 2, 3, 3])
    first = customer_bootstrap(y, p, customers, repeats=20)
    assert first == customer_bootstrap(y, p, customers, repeats=20)
    assert first["ap_95_percentile_interval"] == [1, 1]
