from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FERDataset, make_transform
from src.model import create_model
from src.utils import CLASS_NAMES, atomic_json_dump, classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()
    run_dir = PROJECT_ROOT / "outputs" / "checkpoints" / arguments.run_id
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    model = create_model(config["backbone"], len(CLASS_NAMES), False, float(config.get("dropout", 0.2)))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frame = pd.read_csv(PROJECT_ROOT / "outputs" / "eda" / "validation_split.csv")
    transform = make_transform(int(config["input_size"]), False, checkpoint["normalization"], False)
    dataset = FERDataset(frame, PROJECT_ROOT / "data" / "raw" / "fer_challenge" / "train", transform)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    identifiers: list[str] = []
    with torch.inference_mode():
        for images, targets, batch_ids in loader:
            original = model(images).softmax(dim=1)
            flipped = model(torch.flip(images, dims=[3])).softmax(dim=1)
            probabilities.append(((original + flipped) / 2).numpy())
            labels.append(targets.numpy())
            identifiers.extend(list(batch_ids))
    probabilities_array = np.concatenate(probabilities)
    labels_array = np.concatenate(labels)
    predictions = probabilities_array.argmax(axis=1)
    metrics = classification_metrics(labels_array, predictions)
    pd.DataFrame(
        {
            "id": identifiers,
            "true_label": [CLASS_NAMES[index] for index in labels_array],
            "predicted_label": [CLASS_NAMES[index] for index in predictions],
            **{f"prob_{name}": probabilities_array[:, index] for index, name in enumerate(CLASS_NAMES)},
        }
    ).to_csv(run_dir / "validation_tta_predictions.csv", index=False)
    ordinary = checkpoint["best_metrics"]
    summary = {
        "run_id": arguments.run_id,
        "tta": "horizontal flip probability average",
        "ordinary_accuracy": ordinary["accuracy"],
        "ordinary_macro_f1": ordinary["macro_f1"],
        "tta_accuracy": metrics["accuracy"],
        "tta_macro_f1": metrics["macro_f1"],
        "delta_accuracy": metrics["accuracy"] - ordinary["accuracy"],
        "delta_macro_f1": metrics["macro_f1"] - ordinary["macro_f1"],
        "keep_tta_by_competition_rule": (metrics["accuracy"], metrics["macro_f1"])
        > (ordinary["accuracy"], ordinary["macro_f1"]),
    }
    atomic_json_dump(summary, run_dir / "tta_validation_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

