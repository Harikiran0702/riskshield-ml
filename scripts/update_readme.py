"""Refresh the short README results table from saved evaluation reports."""

import json
from pathlib import Path


def replace_section(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name} START -->", f"<!-- {name} END -->"
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body + "\n" + end + after


if __name__ == "__main__":
    training = json.loads(Path("reports/training.json").read_text())
    evaluation = json.loads(Path("reports/evaluation.json").read_text())
    names = {"logistic": "Logistic Regression", "xgboost": "XGBoost"}
    lines = [
        "| Model | Average Precision | Precision | Recall |",
        "|---|---:|---:|---:|",
    ]
    for name, result in evaluation.items():
        lines.append(
            f"| {names.get(name, name)} | {result['average_precision']:.4f} | "
            f"{result['precision']:.4f} | {result['recall']:.4f} |"
        )
    selected = training["selected_model"]
    policy = evaluation[selected]["policy"]
    lines += [
        "",
        f"**{names.get(selected, selected)}** was selected because it had the lowest "
        "estimated cost on the validation data.",
        "",
        f"Its review threshold is **{policy['review_threshold']:.2f}** and its decline threshold "
        f"is **{policy['decline_threshold']:.2f}**. "
        f"It declined {evaluation[selected]['decisions']['DECLINE']} transactions in the test set.",
    ]
    path = Path("README.md")
    text = replace_section(path.read_text(encoding="utf-8"), "RESULTS", "\n".join(lines))
    path.write_text(text, encoding="utf-8")
