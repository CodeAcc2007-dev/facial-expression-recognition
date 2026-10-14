from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CLASS_NAMES, atomic_json_dump, classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--tta", action="store_true")
    arguments = parser.parse_args()
    if len(arguments.run_ids) < 2:
        raise ValueError("provide at least two run IDs")
    probability_columns = [f"prob_{name}" for name in CLASS_NAMES]
    prediction_name = "validation_tta_predictions.csv" if arguments.tta else "validation_predictions.csv"
    frames = [
        pd.read_csv(PROJECT_ROOT / "outputs" / "checkpoints" / run_id / prediction_name)
        for run_id in arguments.run_ids
    ]
    reference_ids = list(frames[0]["id"])
    reference_truth = (
        frames[0]["true_index"].to_numpy(dtype=int)
        if "true_index" in frames[0]
        else frames[0]["true_label"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    )
    matrices = []
    individual = {}
    for run_id, frame in zip(arguments.run_ids, frames, strict=True):
        frame_truth = (
            frame["true_index"].to_numpy(dtype=int)
            if "true_index" in frame
            else frame["true_label"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
        )
        if list(frame["id"]) != reference_ids or not np.array_equal(frame_truth, reference_truth):
            raise ValueError(f"validation alignment mismatch for {run_id}")
        values = frame[probability_columns].to_numpy(dtype=float)
        matrices.append(values)
        individual[run_id] = classification_metrics(reference_truth, values.argmax(axis=1))
    averaged = np.mean(matrices, axis=0)
    metrics = classification_metrics(reference_truth, averaged.argmax(axis=1))
    ensemble_id = "ensemble_" + "__".join(arguments.run_ids) + ("__tta" if arguments.tta else "")
    output_dir = PROJECT_ROOT / "outputs" / "analysis" / ensemble_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": reference_ids,
            "true_index": reference_truth,
            "true_label": [CLASS_NAMES[index] for index in reference_truth],
            "predicted_index": averaged.argmax(axis=1),
            "predicted_label": [CLASS_NAMES[index] for index in averaged.argmax(axis=1)],
            **{column: averaged[:, index] for index, column in enumerate(probability_columns)},
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    best_individual = max(
        individual.items(), key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"])
    )
    summary = {
        "ensemble_id": ensemble_id,
        "members": arguments.run_ids,
        "method": "equal arithmetic mean of aligned softmax probabilities" + (" with per-member horizontal-flip TTA" if arguments.tta else ""),
        "validation_protocol": "same untouched grouped holdout excluded from every member",
        "metrics": metrics,
        "best_individual": {
            "run_id": best_individual[0],
            "accuracy": best_individual[1]["accuracy"],
            "macro_f1": best_individual[1]["macro_f1"],
        },
        "delta_accuracy": metrics["accuracy"] - best_individual[1]["accuracy"],
        "delta_macro_f1": metrics["macro_f1"] - best_individual[1]["macro_f1"],
        "keep_by_competition_rule": (metrics["accuracy"], metrics["macro_f1"])
        > (best_individual[1]["accuracy"], best_individual[1]["macro_f1"]),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
