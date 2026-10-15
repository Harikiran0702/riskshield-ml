"""Positive log-odds contributions, not causal explanations or generic rule flags."""

import numpy as np
import pandas as pd
import xgboost as xgb

CODES = {
    "log_amount": "TRANSACTION_AMOUNT_SIGNAL",
    "amount_ratio": "AMOUNT_ABOVE_CUSTOMER_BASELINE",
    "amount_deviation": "AMOUNT_DEVIATION",
    "transactions_10m": "HIGH_TRANSACTION_VELOCITY",
    "transactions_1h": "HIGH_TRANSACTION_VELOCITY",
    "spending_velocity": "HIGH_SPENDING_VELOCITY",
    "log_seconds_since_previous": "RECENT_TRANSACTION_ACTIVITY",
    "new_device": "NEW_DEVICE",
    "unusual_location": "UNUSUAL_LOCATION",
    "unusual_hour": "UNUSUAL_TIME",
    "merchant_risk": "MERCHANT_RISK_SIGNAL",
    "category_risk": "MERCHANT_CATEGORY_SIGNAL",
    "log_account_age": "ACCOUNT_AGE_SIGNAL",
    "customer_frequency": "CUSTOMER_ACTIVITY_SIGNAL",
    "cross_border": "CROSS_BORDER",
    "online_channel": "ONLINE_CHANNEL",
}


def contributions(model: object, features: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "xgboost":
        return model.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)[:, :-1]
    scaled = model.named_steps["scale"].transform(features)
    return scaled * model.named_steps["classifier"].coef_[0]


def reason_codes(model: object, features: pd.DataFrame, kind: str) -> list[list[str]]:
    output = []
    for position, row in enumerate(contributions(model, features, kind)):
        codes = []
        for i in np.argsort(-row):
            if row[i] <= 0:
                break
            feature = features.columns[i]
            code = CODES[feature]
            value = features.iloc[position, i]
            # A tree may assign risk to an absent flag. Never claim that flag is present.
            if (
                feature
                in {
                    "new_device",
                    "unusual_location",
                    "unusual_hour",
                    "cross_border",
                    "online_channel",
                    "transactions_10m",
                    "transactions_1h",
                }
                and value == 0
            ):
                code = feature.upper() + "_SIGNAL"
            if feature == "amount_ratio" and value <= 1:
                code = "AMOUNT_RELATIVE_TO_BASELINE"
            if code not in codes:
                codes.append(code)
            if len(codes) == 3:
                break
        output.append(codes or ["NO_POSITIVE_RISK_CONTRIBUTION"])
    return output
