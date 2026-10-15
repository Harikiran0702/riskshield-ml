"""Pre-transaction context comes from a trusted upstream feature service."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskshield.config.settings import SETTINGS

Nonnegative = Annotated[float, Field(ge=0, le=1e12)]
Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: Identifier
    customer_id: Identifier
    timestamp: datetime
    transaction_amount: Annotated[float, Field(gt=0, le=1e7)]
    merchant_id: Identifier
    merchant_category: Literal["grocery", "travel", "electronics", "digital"]
    payment_channel: Literal["pos", "web", "mobile"]
    country: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    home_country: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    device_id: Identifier
    ip_country: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    account_age_days: Annotated[int, Field(ge=0, le=36500)]
    historical_avg_amount: Annotated[float, Field(gt=0, le=1e7)]
    historical_std_amount: Annotated[float, Field(gt=0, le=1e7)]
    prior_transaction_count: Annotated[int, Field(ge=0)]
    transactions_10m: Annotated[int, Field(ge=0)]
    transactions_1h: Annotated[int, Field(ge=0)]
    spend_1h: Nonnegative
    seconds_since_previous: Nonnegative
    new_device: bool
    merchant_risk: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def consistent_context(self) -> "Transaction":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        if not self.transactions_10m <= self.transactions_1h <= self.prior_transaction_count:
            raise ValueError("history counts must satisfy 10m <= 1h <= lifetime")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=SETTINGS["max_batch_size"])


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    reason_codes: list[str]
    model_version: str
    scoring_latency_ms: float


class LabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: Identifier
    fraud_label: Annotated[int, Field(ge=0, le=1, strict=True)]
