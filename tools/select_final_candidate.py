from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates = []
    checkpoints_root = PROJECT_ROOT / "outputs" / "checkpoints"
    for summary_path in checkpoints_root.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("selection_eligible", True):
            continue
        run_id = summary["run_id"]
        metrics = summary["best_metrics"]
        candidates.append(
            {
                "candidate_id": run_id,
                "type": "individual",
                "members": [run_id],
                "tta": False,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_source": str(summary_path.relative_to(PROJECT_ROOT)),
            }
        )
        tta_path = summary_path.parent / "tta_validation_summary.json"
        if tta_path.is_file():
            tta = json.loads(tta_path.read_text(encoding="utf-8"))
            candidates.append(
                {
                    "candidate_id": run_id + "__tta",
                    "type": "individual",
                    "members": [run_id],
                    "tta": True,
                    "validation_accuracy": tta["tta_accuracy"],
                    "validation_macro_f1": tta["tta_macro_f1"],
                    "validation_source": str(tta_path.relative_to(PROJECT_ROOT)),
                }
            )
    for summary_path in (PROJECT_ROOT / "outputs" / "analysis").glob("ensemble_*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        candidates.append(
            {
                "candidate_id": summary["ensemble_id"],
                "type": "ensemble",
                "members": summary["members"],
                "tta": summary["ensemble_id"].endswith("__tta"),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_source": str(summary_path.relative_to(PROJECT_ROOT)),
            }
        )
    if not candidates:
        raise RuntimeError("no eligible candidates found")
    candidates.sort(
        key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"]), reverse=True
    )
    payload = {
        "selection_rule": "validation accuracy descending, then Macro F1 descending",
        "selected": candidates[0],
        "ranking": candidates,
    }
    destination = PROJECT_ROOT / "outputs" / "selected_candidate.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

