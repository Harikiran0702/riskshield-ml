"""Train both candidates and persist a validation-selected model."""

import argparse
from pathlib import Path

from riskshield.models.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    result = train(
        Path("data/generated/transactions.csv"),
        Path("artifacts"),
        Path("reports"),
        track=args.mlflow,
    )
    print("Selected:", result["selected_model"])
