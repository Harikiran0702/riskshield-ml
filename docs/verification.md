# Verification

## Checks

| Check | Result |
|---|---|
| Pytest | 58 passed |
| Ruff lint and formatting | Passed |
| Dependency compatibility | Passed |
| Model training and evaluation | Passed |
| API authentication and request validation | Passed |
| Concurrent duplicate protection and label storage | Passed |
| Docker build and HTTP scoring | Passed |
| Container restart persistence and healthcheck | Passed |
| Non-root container execution | Passed |

Evidence is stored in `reports/test-results.xml`, `reports/container_verification.json`, and the lint and dependency reports. The test run includes one upstream Starlette/AnyIO deprecation warning.

## Model evaluation

The dataset contains 30,000 synthetic transactions, split chronologically into 14,400 training, 3,600 calibration, 6,000 validation, and 6,000 test events.

| Model | Average Precision | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2145 | 0.8078 | 0.0730 | 0.6939 |
| XGBoost | 0.2300 | 0.7821 | 0.0446 | 0.7959 |

Logistic Regression was selected by validation cost, with review and decline thresholds of 0.02 and 1.00. Three rolling backtests and customer-cluster uncertainty estimates are recorded in `reports/backtest.json`.

Calibration uses a separate chronological window. Feature contributions explain the underlying model before calibration. Synthetic evaluation does not establish performance on real transactions.

## Reproduce

```bash
python scripts/generate_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/backtest.py
python scripts/monitor.py
python scripts/verify_api.py
pytest -q
ruff check .
ruff format --check .
docker build -t riskshield-ml:1.1 .
python scripts/verify_container.py --image riskshield-ml:1.1
```

Local latency samples are available in `reports/api_verification.json`. They measure feature processing, prediction, calibration, and explanations, excluding HTTP and database writes. They are not load-test results.
