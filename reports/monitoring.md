# Monitoring report

Synthetic demo unless external CSV windows supplied.
Demo latency covers feature creation and prediction, excluding SHAP and HTTP.
Demo injects telemetry drift after scoring; performance is on original test scores.

Alerts: amount_ratio, merchant_risk, customer_frequency

| Signal | PSI | KS statistic | Missing delta |
|---|---:|---:|---:|
| log_amount | 0.0037 | 0.012666666666666666 | 0.0000 |
| amount_ratio | 1.4284 | 0.475 | 0.0000 |
| transactions_10m | 0.0008 | 0.0075 | 0.0000 |
| transactions_1h | 0.0007 | 0.009833333333333333 | 0.0000 |
| spending_velocity | 0.0021 | 0.017666666666666667 | 0.0000 |
| log_seconds_since_previous | 0.0032 | 0.0105 | 0.0000 |
| new_device | 0.0007 | 0.008333333333333333 | 0.0000 |
| unusual_location | 0.0003 | 0.006333333333333333 | 0.0000 |
| unusual_hour | 0.0006 | 0.010833333333333334 | 0.0000 |
| merchant_risk | 0.0027 | 0.01603030303030303 | 0.0833 |
| category_risk | 0.0001 | 0.003 | 0.0000 |
| log_account_age | 0.0100 | 0.019833333333333335 | 0.0000 |
| customer_frequency | 0.2590 | 0.16916666666666666 | 0.0000 |
| amount_deviation | 0.0057 | 0.02 | 0.0000 |
| cross_border | 0.0002 | 0.004166666666666667 | 0.0000 |
| online_channel | 0.0001 | 0.004833333333333334 | 0.0000 |
| score | 0.0044 | 0.015333333333333332 | 0.0000 |

```json
{
  "alerts": [
    "amount_ratio",
    "merchant_risk",
    "customer_frequency"
  ],
  "volume_rate_change": -0.0038818421641299006,
  "latency": {
    "p50_ms": 5.1469000754877925,
    "p95_ms": 7.380030094645915
  },
  "score_distribution": {
    "count": 6000.0,
    "mean": 0.021608848458591626,
    "std": 0.05759888729107073,
    "min": 0.0009797232952692388,
    "25%": 0.005381416836582668,
    "50%": 0.009621476823213605,
    "75%": 0.01872743007126959,
    "max": 0.9972166888331258
  },
  "label_coverage": 1.0,
  "eventual_performance": {
    "brier_score": 0.02160047757220583,
    "log_loss": 0.0951931695018349,
    "roc_auc": 0.8078013368340674,
    "average_precision": 0.21449359611953817,
    "precision": 0.07301360057265568,
    "recall": 0.6938775510204082,
    "f1": 0.13212435233160622,
    "fraud_detection_rate": 0.6938775510204082,
    "false_positive_rate": 0.22125405774816334,
    "confusion_matrix": [
      [
        4558,
        1295
      ],
      [
        45,
        102
      ]
    ]
  }
}
```