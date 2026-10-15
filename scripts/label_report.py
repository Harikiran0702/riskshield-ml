"""Report outcomes joined durably to their original scores."""

import argparse
import json
import os
from pathlib import Path

from riskshield.api.store import AuditStore
from riskshield.evaluation.metrics import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    frame = AuditStore(
        Path(os.getenv("RISKSHIELD_DATABASE", "artifacts/service.sqlite3"))
    ).outcomes()
    known = frame.dropna(subset=["fraud_label"])
    report = {
        "scored": len(frame),
        "labeled": len(known),
        "coverage": len(known) / len(frame) if len(frame) else 0,
        "threshold": args.threshold,
        "metrics": metrics(known.fraud_label.to_numpy(), known.score.to_numpy(), args.threshold)
        if len(known)
        else None,
    }
    Path("reports/label_outcomes.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
