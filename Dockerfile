FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY scripts ./scripts
RUN python scripts/generate_data.py && python scripts/train.py

FROM base
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser && mkdir /app/state && chown appuser:appuser /app/state
ENV RISKSHIELD_DATABASE=/app/state/service.sqlite3
COPY --from=builder /app/artifacts/selected.joblib ./artifacts/selected.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "riskshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
