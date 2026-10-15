"""Shared offline/online feature transformation with an explicit allowlist."""

import numpy as np
import pandas as pd

from riskshield.config.settings import CATEGORY_RISK

FEATURE_NAMES = [
    "log_amount",
    "amount_ratio",
    "transactions_10m",
    "transactions_1h",
    "spending_velocity",
    "log_seconds_since_previous",
    "new_device",
    "unusual_location",
    "unusual_hour",
    "merchant_risk",
    "category_risk",
    "log_account_age",
    "customer_frequency",
    "amount_deviation",
    "cross_border",
    "online_channel",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(frame["timestamp"], utc=True)
    avg = frame["historical_avg_amount"].clip(lower=1)
    values = {
        "log_amount": np.log1p(frame["transaction_amount"]),
        "amount_ratio": frame["transaction_amount"] / avg,
        "transactions_10m": frame["transactions_10m"],
        "transactions_1h": frame["transactions_1h"],
        "spending_velocity": frame["spend_1h"] / avg,
        "log_seconds_since_previous": np.log1p(frame["seconds_since_previous"]),
        "new_device": frame["new_device"].astype(float),
        "unusual_location": (frame["ip_country"] != frame["home_country"]).astype(float),
        "unusual_hour": (t.dt.hour < 6).astype(float),
        "merchant_risk": frame["merchant_risk"],
        "category_risk": frame["merchant_category"].map(CATEGORY_RISK),
        "log_account_age": np.log1p(frame["account_age_days"]),
        "customer_frequency": frame["prior_transaction_count"] / frame["account_age_days"].clip(1),
        "amount_deviation": (frame["transaction_amount"] - avg)
        / frame["historical_std_amount"].clip(1),
        "cross_border": (frame["country"] != frame["home_country"]).astype(float),
        "online_channel": (frame["payment_channel"] != "pos").astype(float),
    }
    result = pd.DataFrame(values, index=frame.index)[FEATURE_NAMES].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature values must be finite and categories must be known")
    return result
