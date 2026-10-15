"""Generate ignored training data and a small committed request example."""

import argparse
import json
from pathlib import Path

from riskshield.api.schemas import Transaction
from riskshield.config.settings import SETTINGS
from riskshield.data.generator import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=SETTINGS["rows"])
    args = parser.parse_args()
    frame = generate(args.rows)
    Path("data/generated").mkdir(parents=True, exist_ok=True)
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    frame.to_csv("data/generated/transactions.csv", index=False)
    sample = Transaction.model_validate(frame.drop(columns="fraud_label").iloc[-1].to_dict())
    Path("data/sample/transaction.json").write_text(
        json.dumps(sample.model_dump(mode="json"), indent=2)
    )
    print(f"Generated {len(frame)} transactions; fraud prevalence={frame.fraud_label.mean():.4f}")
