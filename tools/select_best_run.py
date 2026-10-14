from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    log_path = PROJECT_ROOT / "experiments" / "experiments.csv"
    frame = pd.read_csv(log_path)
    eligible_rows = []
    for row in frame.to_dict(orient="records"):
        checkpoint = PROJECT_ROOT / str(row["checkpoint_path"])
        summary_path = checkpoint.parent / "summary.json"
        if not checkpoint.is_file() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("selection_eligible", True):
            continue
        eligible_rows.append(row)
    if not eligible_rows:
        raise RuntimeError("no selection-eligible completed runs")
    eligible = pd.DataFrame(eligible_rows)
    eligible["val_acc"] = pd.to_numeric(eligible["val_acc"])
    eligible["val_macro_f1"] = pd.to_numeric(eligible["val_macro_f1"])
    eligible = eligible.sort_values(
        ["val_acc", "val_macro_f1", "timestamp"], ascending=[False, False, True]
    ).reset_index(drop=True)
    selected = eligible.iloc[0].to_dict()
    payload = {
        "selection_rule": "validation accuracy descending, macro F1 descending, timestamp ascending",
        "selected": selected,
        "ranking": eligible[["run_id", "val_acc", "val_macro_f1", "best_epoch", "runtime_min"]].to_dict(orient="records"),
    }
    destination = PROJECT_ROOT / "outputs" / "selected_run.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

