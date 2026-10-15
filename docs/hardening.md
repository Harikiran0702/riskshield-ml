# Service configuration

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `RISKSHIELD_API_KEY` | API key, at least 24 characters | Required |
| `RISKSHIELD_DATABASE` | SQLite audit database | `artifacts/service.sqlite3` |
| `RISKSHIELD_MODEL_PATH` | Model artifact | `artifacts/selected.joblib` |
| `RISKSHIELD_REQUESTS_PER_MINUTE` | Per-process request limit | `120` |
| `RISKSHIELD_DEMO_MODE` | Disable authentication for local development | Disabled |

Model parameters, cost assumptions, and monitoring thresholds are in `src/riskshield/config/settings.json`.

Generate a key in PowerShell:

```powershell
$env:RISKSHIELD_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Or in Bash:

```bash
export RISKSHIELD_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Send the key in the `X-API-Key` header. Authentication applies to scoring, model metadata, and label submission. Health and API schema endpoints are public. Demo mode must remain disabled when authentication is required.

## Request handling

- Request bodies are limited to 256 KiB; batches are limited to 100 transactions.
- Repeated transaction IDs with identical input return the original response.
- Reused IDs with changed input return HTTP 409. Conflicting batches roll back atomically.
- SQLite stores request hashes, decisions, timestamps, and labels. Raw customer and device context is not stored.
- Unknown transaction IDs on label submission return HTTP 404. Conflicting labels return HTTP 409.
- Invalid credentials return 401, oversized bodies return 413, validation errors return 422, and exhausted request limits return 429.

The rate limit is process-local. SQLite serializes scoring writes; it is not a distributed feature store. Retried responses retain their original model version and computation time.

## Label evaluation

Submit outcomes through `POST /labels` with `transaction_id` and a binary `fraud_label`. Report performance at an explicit threshold:

```bash
python scripts/label_report.py --threshold 0.02
```

The report includes label coverage. Missing labels are not treated as legitimate transactions. Separate policy versions when comparing operational performance.

## Docker

With the API key configured:

```bash
docker compose up --build -d
docker compose ps
```

Compose binds the service to loopback and mounts a persistent audit volume. `docker compose down` stops the service while retaining the volume.

Build and verify the image independently:

```bash
docker build -t riskshield-ml:1.1 .
python scripts/verify_container.py --image riskshield-ml:1.1
```

The verifier checks authentication, scoring, duplicate handling, labels, restart persistence, non-root execution, and container health. It removes its temporary container and volume after execution.

External deployment requires TLS, managed secrets, access controls, and a data-retention policy.
