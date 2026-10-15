"""Business actions are separate from probability estimation."""

from dataclasses import dataclass

from riskshield.config.settings import SETTINGS


@dataclass(frozen=True)
class Policy:
    review_threshold: float = SETTINGS["default_review_threshold"]
    decline_threshold: float = SETTINGS["default_decline_threshold"]

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold < self.decline_threshold <= 1:
            raise ValueError("Require 0 <= review < decline <= 1")

    def decide(self, probability: float) -> tuple[str, str]:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if probability >= self.decline_threshold:
            return "DECLINE", "HIGH"
        if probability >= self.review_threshold:
            return "REVIEW", "MEDIUM"
        return "APPROVE", "LOW"
