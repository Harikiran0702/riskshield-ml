"""Evaluate frozen models on the untouched newest temporal partition."""

import json
from pathlib import Path

from riskshield.models.training import evaluate

if __name__ == "__main__":
    print(
        json.dumps(
            evaluate(Path("data/generated/transactions.csv"), Path("artifacts"), Path("reports")),
            indent=2,
        )
    )
