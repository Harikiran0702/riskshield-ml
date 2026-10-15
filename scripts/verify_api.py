"""Exercise the saved model through FastAPI and record a real response and latency sample."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi.testclient import TestClient

from riskshield.api.main import create_app
from riskshield.api.security import ServiceSettings

if __name__ == "__main__":
    payload = json.loads(Path("data/sample/transaction.json").read_text())
    with (
        TemporaryDirectory() as directory,
        TestClient(
            create_app(
                settings=ServiceSettings(demo_mode=True, database=Path(directory) / "audit.db")
            )
        ) as client,
    ):
        assert client.get("/health").status_code == 200
        assert client.get("/model-info").status_code == 200
        latencies = []
        for i in range(30):
            request = {**payload, "transaction_id": f"VERIFY-{i}"} if i < 29 else payload
            response = client.post("/score", json=request)
            response.raise_for_status()
            latencies.append(response.json()["scoring_latency_ms"])
        batch = client.post("/score/batch", json={"transactions": [payload, payload]})
        batch.raise_for_status()
        assert len(batch.json()) == 2
    Path("reports/sample_response.json").write_text(json.dumps(response.json(), indent=2))
    summary = {
        "requests": len(latencies),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "scope": "feature transformation, model prediction and explanation; excludes HTTP",
    }
    Path("reports/api_verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
