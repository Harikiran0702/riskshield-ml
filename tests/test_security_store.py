from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from riskshield.api.main import create_app
from riskshield.api.security import Admission, ServiceSettings
from riskshield.api.store import AuditStore, ConflictError

KEY = "test-key-long-enough-for-service"


def test_secure_default_requires_key():
    with pytest.raises(ValueError, match="API_KEY"):
        ServiceSettings()


def test_authentication_and_rate_limit(trained, tmp_path, payload):
    config = ServiceSettings(api_key=KEY, database=tmp_path / "db", requests_per_minute=2)
    with TestClient(create_app(trained / "artifacts/selected.joblib", config)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/score", json=payload).status_code == 401
        headers = {"X-API-Key": KEY}
        assert client.post("/score", json=payload, headers=headers).status_code == 200
        assert client.get("/model-info", headers=headers).status_code == 200
        assert client.get("/model-info", headers=headers).status_code == 429


def test_window_expires():
    gate = Admission(ServiceSettings(demo_mode=True, requests_per_minute=1))
    assert gate.admit(0)
    assert not gate.admit(59)
    assert gate.admit(60)


def test_oversized_body_rejected(trained, tmp_path):
    config = ServiceSettings(demo_mode=True, database=tmp_path / "db", max_body_bytes=32)
    with TestClient(create_app(trained / "artifacts/selected.joblib", config)) as client:
        assert client.post("/score", content=b"x" * 100).status_code == 413


def test_duplicate_conflict_and_batch_rollback(client, payload):
    first = client.post("/score", json=payload)
    assert client.post("/score", json=payload).json() == first.json()
    changed = {**payload, "transaction_amount": payload["transaction_amount"] + 1}
    assert client.post("/score", json=changed).status_code == 409
    fresh = {**payload, "transaction_id": "FRESH"}
    assert client.post("/score/batch", json={"transactions": [fresh, changed]}).status_code == 409
    assert (
        client.post("/labels", json={"transaction_id": "FRESH", "fraud_label": 1}).status_code
        == 404
    )


def test_label_join_and_conflict(client, payload):
    client.post("/score", json=payload)
    label = {"transaction_id": payload["transaction_id"], "fraud_label": 1}
    assert client.post("/labels", json=label).status_code == 200
    assert client.post("/labels", json=label).status_code == 200
    assert client.post("/labels", json={**label, "fraud_label": 0}).status_code == 409
    assert client.post("/labels", json={**label, "fraud_label": True}).status_code == 422


def test_durable_concurrent_idempotency(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    payload = {"transaction_id": "one"}
    calls = []

    def calculate(rows):
        calls.append(1)
        return [{"fraud_probability": 0.2} for _ in rows]

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: store.score([payload], calculate), range(8)))
    assert len(calls) == 1
    assert all(output == outputs[0] for output in outputs)
    reopened = AuditStore(tmp_path / "audit.db")
    assert reopened.score([payload], calculate) == outputs[0]
    assert len(calls) == 1
    reopened.label("one", 1)
    assert reopened.outcomes().iloc[0].fraud_label == 1
    with pytest.raises(ConflictError):
        reopened.score([{**payload, "changed": True}], calculate)


def test_failed_calculation_rolls_back(tmp_path):
    store = AuditStore(tmp_path / "audit.db")

    def fail(rows):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError):
        store.score([{"transaction_id": "x"}], fail)
    assert store.outcomes().empty
