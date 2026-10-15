"""Generate robustness evidence without tuning on the held-out test window."""

import json
from pathlib import Path

import joblib
import pandas as pd

from riskshield.evaluation.robustness import customer_bootstrap, rolling_backtest
from riskshield.features.engineering import build_features
from riskshield.models.calibration import predict_probability
from riskshield.models.training import temporal_split

if __name__ == "__main__":
    frame = pd.read_csv("data/generated/transactions.csv")
    bundle = joblib.load("artifacts/selected.joblib")
    _, _, test = temporal_split(frame)
    p = predict_probability(bundle, build_features(test))
    report = {
        "rolling_folds": rolling_backtest(frame),
        "selected_model_test_uncertainty": customer_bootstrap(
            test.fraud_label.to_numpy(), p, test.customer_id.to_numpy()
        ),
    }
    Path("reports/backtest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
