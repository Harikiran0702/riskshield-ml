"""Stateless scoring API. History comes from a trusted upstream context service."""

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from riskshield.api.schemas import BatchRequest, LabelRequest, ScoreResponse, Transaction
from riskshield.api.security import Admission, ServiceSettings
from riskshield.api.store import AuditStore, ConflictError
from riskshield.explainability.reasons import reason_codes
from riskshield.features.engineering import FEATURE_NAMES, build_features
from riskshield.models.calibration import predict_probability

logger = logging.getLogger("riskshield")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(model_path: Path | None = None, settings: ServiceSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        app.state.configuration_error = False
        try:
            config = settings or ServiceSettings.from_env()
            app.state.admission = Admission(config)
            app.state.store = AuditStore(config.database)
        except (ValueError, OSError, sqlite3.Error):
            app.state.configuration_error = True
            logger.error(json.dumps({"event": "service_configuration_invalid"}))
        path = model_path or Path(os.getenv("RISKSHIELD_MODEL_PATH", "artifacts/selected.joblib"))
        try:
            bundle = joblib.load(path)  # Only load trusted locally produced artifacts.
            if bundle["feature_names"] != FEATURE_NAMES:
                raise ValueError("Incompatible model feature schema")
            app.state.bundle = bundle
            logger.info(json.dumps({"event": "model_loaded", "version": bundle["version"]}))
        except Exception as exc:
            logger.error(
                json.dumps({"event": "model_load_failed", "error_type": type(exc).__name__})
            )
        yield

    app = FastAPI(title="RiskShield ML", version="1.1.0", lifespan=lifespan)
    protected = [Depends(APIKeyHeader(name="X-API-Key", auto_error=False))]

    @app.middleware("http")
    async def admission(request: Request, call_next):
        if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)
        if getattr(app.state, "configuration_error", True):
            return JSONResponse(
                status_code=503, content={"detail": "Service configuration invalid"}
            )
        gate = app.state.admission
        if not gate.authorized(request.headers.get("x-api-key", "")):
            return JSONResponse(status_code=401, content={"detail": "Valid X-API-Key required"})
        if not gate.admit():
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": "60"},
                content={"detail": "Request rate limit exceeded"},
            )
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > gate.settings.max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        logger.warning(json.dumps({"event": "invalid_request", "path": request.url.path}))
        # Default errors echo input; redact it to protect context fields.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in exc.errors()
                ]
            },
        )

    def get_bundle() -> dict:
        if getattr(app.state, "configuration_error", True):
            raise HTTPException(503, "Service configuration invalid")
        bundle = getattr(app.state, "bundle", None)
        if bundle is None:
            raise HTTPException(503, "Model unavailable; run training before scoring")
        return bundle

    @app.get("/health")
    def health():
        get_bundle()
        return {"status": "ok", "model_ready": True}

    @app.get("/model-info", dependencies=protected)
    def model_info():
        b = get_bundle()
        return {
            "model_version": b["version"],
            "model_type": b["kind"],
            "features": b["feature_names"],
            "policy": vars(b["policy"]),
            "training": b["metadata"],
            "calibration": "sigmoid" if b.get("calibrator") is not None else "none",
        }

    def score_many(transactions: list[Transaction]) -> list[ScoreResponse]:
        start = perf_counter()
        b = get_bundle()
        logger.info(json.dumps({"event": "scoring_request", "batch_size": len(transactions)}))
        try:
            features = build_features(pd.DataFrame([t.model_dump() for t in transactions]))
            probabilities = predict_probability(b, features)
            reasons = reason_codes(b["model"], features, b["kind"])
            elapsed = (perf_counter() - start) * 1000
            output = []
            for tx, p, codes in zip(transactions, probabilities, reasons, strict=True):
                decision, band = b["policy"].decide(float(p))
                output.append(
                    ScoreResponse(
                        transaction_id=tx.transaction_id,
                        fraud_probability=float(p),
                        decision=decision,
                        risk_band=band,
                        reason_codes=codes,
                        model_version=b["version"],
                        scoring_latency_ms=round(elapsed, 3),
                    )
                )
            logger.info(
                json.dumps(
                    {
                        "event": "prediction_result",
                        "count": len(output),
                        "latency_ms": elapsed,
                        "mean_probability": float(probabilities.mean()),
                        "decisions": {
                            d: sum(o.decision == d for o in output)
                            for d in ["APPROVE", "REVIEW", "DECLINE"]
                        },
                    }
                )
            )
            return output
        except Exception as exc:
            logger.error(json.dumps({"event": "scoring_failed", "error_type": type(exc).__name__}))
            raise HTTPException(500, "Scoring failed") from exc

    def persisted_score(transactions: list[Transaction]) -> list[dict]:
        get_bundle()
        try:
            return app.state.store.score(
                [t.model_dump(mode="json") for t in transactions],
                lambda payloads: [
                    s.model_dump()
                    for s in score_many([Transaction.model_validate(p) for p in payloads])
                ],
            )
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except sqlite3.Error as exc:
            logger.error(json.dumps({"event": "audit_store_unavailable"}))
            raise HTTPException(503, "Audit store unavailable") from exc

    @app.post("/score", response_model=ScoreResponse, dependencies=protected)
    def score(transaction: Transaction):
        return persisted_score([transaction])[0]

    @app.post("/score/batch", response_model=list[ScoreResponse], dependencies=protected)
    def batch(request: BatchRequest):
        return persisted_score(request.transactions)

    @app.post("/labels", dependencies=protected)
    def label(request: LabelRequest):
        try:
            app.state.store.label(request.transaction_id, request.fraud_label)
        except KeyError as exc:
            raise HTTPException(404, "Transaction has not been scored") from exc
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "recorded", "transaction_id": request.transaction_id}

    return app


app = create_app()
