"""Central configuration; learned policy is persisted with its model."""

import json
from pathlib import Path

SETTINGS = json.loads(Path(__file__).with_name("settings.json").read_text())
CATEGORY_RISK = {"grocery": 0.05, "travel": 0.3, "electronics": 0.5, "digital": 0.7}
