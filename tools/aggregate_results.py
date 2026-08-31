"""Aggregate experiments/run_*/metrics.json into markdown tables.

Run after downloading Kaggle outputs:  python -m tools.aggregate_results
Writes experiments/RESULTS.md and prints it.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"

ORDER = ["e1_baseline", "e2_hinglish_aug", "e3_tinymel_scratch", "e4_no_pause_aug",
         "e5_distill", "e6_full_data"]
SLICES = ["overall", "english", "hindi", "hinglish", "filler", "human_audio"]


def load_all() -> dict[str, dict]:
    runs = {}
    for path in sorted(EXP.glob("run_*/metrics.json")):
        m = json.loads(path.read_text())
        runs[m["experiment"]] = m
    return runs


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def main():
    runs = load_all()
    if not runs:
        print("no experiments/run_*/metrics.json found yet")
        return
    names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]

    lines = ["# Experiment results", ""]

    lines += ["## Accuracy (tuned threshold) by test slice", ""]
    lines.append("| slice | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for s in SLICES:
        row = [s]
        for n in names:
            cell = runs[n]["test"].get(s, {})
            row.append(f"{fmt(cell.get('acc_tuned'))} (n={cell.get('n', 0)})")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## AUC by test slice", ""]
    lines.append("| slice | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for s in SLICES:
        row = [s] + [fmt(runs[n]["test"].get(s, {}).get("auc")) for n in names]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Model footprint & training", ""]
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    rows = [
        ("params", lambda m: f"{m['params']:,}"),
        ("int8 ONNX MB", lambda m: fmt(m.get("int8_subset", {}).get("size_mb"))),
        ("int8 acc (subset)", lambda m: fmt(m.get("int8_subset", {}).get("acc_tuned"))),
        ("int8 AUC (subset)", lambda m: fmt(m.get("int8_subset", {}).get("auc"))),
        ("best val AUC", lambda m: fmt(m.get("best_val_auc"))),
        ("threshold", lambda m: fmt(m.get("threshold"))),
        ("train minutes", lambda m: fmt(m.get("train_minutes"))),
        ("train rows", lambda m: f"{m.get('train_rows', 0):,}"),
    ]
    for label, get in rows:
        lines.append("| " + label + " | " + " | ".join(get(runs[n]) for n in names) + " |")

    out = "\n".join(lines) + "\n"
    (EXP / "RESULTS.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
