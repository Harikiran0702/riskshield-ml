import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from riskshield.api.main import create_app
from riskshield.api.security import ServiceSettings
from riskshield.features.engineering import build_features
from riskshield.models.calibration import predict_probability
from riskshield.monitoring.drift import monitor, psi


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_ready": True}
    assert client.get("/model-info").json()["model_version"] == "1.1.0"


def test_api_score_offline_parity(client, payload, bundle):
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    result = response.json()
    expected = predict_probability(bundle, build_features(pd.DataFrame([payload])))[0]
    assert result["fraud_probability"] == pytest.approx(expected)
    assert result["decision"] == bundle["policy"].decide(float(expected))[0]
    assert result["scoring_latency_ms"] > 0
    assert result["reason_codes"]


def test_batch_parity(client, payload):
    single = client.post("/score", json=payload).json()
    batch = client.post("/score/batch", json={"transactions": [payload, payload]}).json()
    assert len(batch) == 2
    assert batch[0]["fraud_probability"] == pytest.approx(single["fraud_probability"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_amount", -1),
        ("merchant_risk", 2),
        ("timestamp", "2025-01-01"),
        ("merchant_category", "invalid"),
        ("fraud_label", 1),
        ("transactions_10m", 99999),
        ("customer_id", ""),
    ],
)
def test_invalid_requests(client, payload, field, value):
    response = client.post("/score", json={**payload, field: value})
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


def test_batch_limits(client, payload):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422
    assert client.post("/score/batch", json={"transactions": [payload] * 101}).status_code == 422


def test_missing_model_returns_503(tmp_path, payload):
    with TestClient(
        create_app(
            tmp_path / "missing.joblib",
            ServiceSettings(demo_mode=True, database=tmp_path / "audit.db"),
        )
    ) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/score", json=payload).status_code == 503


def test_swagger_exposes_api_key_authorization(client):
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
    assert schema["paths"]["/score"]["post"]["security"] == [{"APIKeyHeader": []}]


def test_psi_stable_and_shifted():
    values = np.random.default_rng(1).normal(size=1000)
    assert psi(values, values) == 0
    assert psi(values, values + 4) > 0.2
    assert psi(np.zeros(100), np.ones(100)) > 0.2


def test_monitoring_drift_missing_volume_latency_and_labels():
    reference = pd.DataFrame(
        {"feature": np.arange(100, dtype=float), "score": np.linspace(0, 0.2, 100)}
    )
    current = reference.copy()
    current["score"] += 0.7
    current.loc[:19, "feature"] = np.nan
    current["fraud_label"] = np.r_[np.zeros(50), np.ones(50)]
    report = monitor(reference, current, 2, 1, 0.5, np.array([200, 300]))
    assert set(["feature", "score", "transaction_volume", "latency"]) <= set(report["alerts"])
    assert report["eventual_performance"]["recall"] == 1
    assert report["label_coverage"] == 1


def test_unlabeled_monitoring_is_not_zero_performance():
    frame = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    report = monitor(frame, frame, 1, 1, 0.2)
    assert report["eventual_performance"] is None
    assert report["label_coverage"] == 0
    assert report["alerts"] == []


def test_monitoring_invalid_window():
    frame = pd.DataFrame({"score": [0.1]})
    with pytest.raises(ValueError):
        monitor(frame, frame, 0, 1, 0.2)
