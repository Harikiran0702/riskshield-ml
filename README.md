# RiskShield ML

A transaction fraud detection service built with Python, scikit-learn, XGBoost, and FastAPI. Each request returns a fraud probability, a risk decision, and the main contributing factors.

The project uses reproducible synthetic data.

## Features

- Logistic Regression and XGBoost with temporal validation and probability calibration
- Transaction amount, spending history, velocity, device, and location features
- Cost-based APPROVE, REVIEW, and DECLINE decisions
- Authenticated single and batch scoring
- SQLite audit storage, duplicate protection, and delayed fraud labels
- Data drift monitoring and rolling model evaluation

## Installation

Requires Python 3.12. Commands below use PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux or macOS, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

## Training

```bash
python scripts/generate_data.py
python scripts/train.py
python scripts/evaluate.py
```

Models use earlier transactions for training and later transactions for calibration, threshold selection, and testing. Historical features exclude the current transaction and future activity.

## API

Set an API key through the environment and start the service:

```powershell
$env:RISKSHIELD_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
uvicorn riskshield.api.main:app --host 127.0.0.1 --port 8000
```

Requests require the `X-API-Key` header. Keep keys out of source control. See [configuration](docs/hardening.md) for environment settings and Docker persistence.

Example request, using the same key in the client terminal:

```powershell
curl.exe -X POST http://127.0.0.1:8000/score -H "X-API-Key: $env:RISKSHIELD_API_KEY" -H "Content-Type: application/json" --data-binary @data/sample/transaction.json
```

Responses include `fraud_probability`, `risk_band`, `decision`, `reason_codes`, `model_version`, and `scoring_latency_ms`.

| Endpoint | Description |
|---|---|
| `GET /health` | Service readiness |
| `GET /model-info` | Model metadata and decision thresholds |
| `POST /score` | Single transaction scoring |
| `POST /score/batch` | Batch scoring, up to 100 transactions |
| `POST /labels` | Record a fraud label for a scored transaction |

## Results

Measured on 6,000 held-out transactions from a dataset of 30,000 synthetic transactions. Precision and recall use each model's selected review threshold.

<!-- RESULTS START -->
| Model | Average Precision | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.2145 | 0.0730 | 0.6939 |
| XGBoost | 0.2300 | 0.0446 | 0.7959 |

**Logistic Regression** was selected because it had the lowest estimated cost on the validation data.

Its review threshold is **0.02** and its decline threshold is **1.00**. It declined 0 transactions in the test set.
<!-- RESULTS END -->

Detailed metrics are available in [reports](reports/).

## Monitoring and tests

```bash
python scripts/monitor.py
python scripts/backtest.py
pytest -q
ruff check .
ruff format --check .
```

## Docker

With `RISKSHIELD_API_KEY` set:

```bash
docker compose up --build -d
```

The service runs as a non-root user and stores audit records in a persistent volume. Validation results are documented in [verification](docs/verification.md).

## Structure

```text
src/riskshield/   Data, features, models, API, decisioning, and monitoring
scripts/         Training, evaluation, and verification commands
tests/           Automated tests
reports/         Evaluation and verification results
data/sample/     Example transaction request
```

Generated data, model artifacts, databases, and secrets are excluded from Git.

## Limitations

Synthetic results do not establish performance on real payments. The service expects precomputed customer history, and SQLite storage is intended for local use. Distributed feature storage and deployment infrastructure are outside the current scope.

## License

[MIT](LICENSE)
