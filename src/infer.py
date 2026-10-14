from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import FERDataset, make_transform
from .model import create_model
from .utils import CLASS_NAMES, atomic_json_dump, record_submission_hash, sha256_file


def validate_submission(frame: pd.DataFrame, test_ids: list[str]) -> None:
    if list(frame.columns) != ["id", "label"]:
        raise ValueError("submission columns must be exactly id,label")
    if list(frame["id"]) != list(test_ids):
        raise ValueError("submission IDs/order do not exactly match test.csv")
    if frame["id"].isna().any() or frame["label"].isna().any():
        raise ValueError("submission contains nulls")
    if not frame["id"].is_unique:
        raise ValueError("submission IDs are not unique")
    invalid = sorted(set(frame["label"]) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"submission contains invalid labels: {invalid}")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False, quoting=csv.QUOTE_MINIMAL)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _predict(model, loader, device: torch.device, tta: bool) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_ids in loader:
            images = images.to(device)
            logits = model(images)
            batch_probabilities = logits.softmax(dim=1)
            if tta:
                flipped = torch.flip(images, dims=[3])
                batch_probabilities = (batch_probabilities + model(flipped).softmax(dim=1)) / 2
            probabilities.append(batch_probabilities.cpu().numpy())
            identifiers.extend(list(batch_ids))
    return identifiers, np.concatenate(probabilities)


def infer(
    checkpoint_path: str | Path,
    output_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 256,
    workers: int = 0,
    tta: bool = False,
    verify_repeat: bool = True,
    update_experiment_log: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    data_dir = Path(data_dir).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError("checkpoint class order mismatch")
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        config["backbone"], len(CLASS_NAMES), pretrained=False, dropout=float(config.get("dropout", 0.2))
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    test = pd.read_csv(data_dir / "test.csv")
    transform = make_transform(
        int(config["input_size"]), False, checkpoint["normalization"], augmentation=False
    )
    dataset = FERDataset(test, data_dir / "test", transform, labelled=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    started = time.perf_counter()
    identifiers, probabilities = _predict(model, loader, device, tta)
    repeat_max_difference = None
    if verify_repeat:
        repeat_ids, repeated = _predict(model, loader, device, tta)
        if repeat_ids != identifiers:
            raise AssertionError("repeat inference ID order changed")
        repeat_max_difference = float(np.max(np.abs(probabilities - repeated)))
        if repeat_max_difference != 0.0:
            raise AssertionError(f"repeat inference changed by {repeat_max_difference}")
    predictions = probabilities.argmax(axis=1)
    submission = pd.DataFrame(
        {"id": identifiers, "label": [CLASS_NAMES[index] for index in predictions]}
    )
    validate_submission(submission, list(test["id"]))
    _atomic_csv(submission, output_path)
    reloaded = pd.read_csv(output_path)
    validate_submission(reloaded, list(test["id"]))
    evidence_path = output_path.with_name(output_path.stem + "_probabilities.csv")
    evidence = pd.DataFrame({"id": identifiers, **{f"prob_{name}": probabilities[:, index] for index, name in enumerate(CLASS_NAMES)}})
    _atomic_csv(evidence, evidence_path)
    summary = {
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_rule": checkpoint["selection_rule"],
        "rows": len(submission),
        "columns": list(submission.columns),
        "exact_test_id_order": True,
        "tta_horizontal_flip": tta,
        "repeat_max_absolute_probability_difference": repeat_max_difference,
        "runtime_seconds": time.perf_counter() - started,
        "submission": str(output_path),
        "submission_sha256": sha256_file(output_path),
        "probability_evidence": str(evidence_path),
        "label_counts": {name: int((submission["label"] == name).sum()) for name in CLASS_NAMES},
    }
    atomic_json_dump(summary, output_path.with_name(output_path.stem + "_summary.json"))
    project_root = checkpoint_path.parents[3]
    experiment_log = project_root / "experiments" / "experiments.csv"
    if update_experiment_log and experiment_log.exists():
        record_submission_hash(experiment_log, config["run_id"], summary["submission_sha256"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/raw/fer_challenge")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-repeat-check", action="store_true")
    parser.add_argument(
        "--no-log-update",
        action="store_true",
        help="Preserve the run's ordinary submission hash when writing a derived candidate such as TTA.",
    )
    arguments = parser.parse_args()
    summary = infer(
        arguments.checkpoint,
        arguments.output,
        arguments.data_dir,
        arguments.batch_size,
        arguments.workers,
        arguments.tta,
        not arguments.no_repeat_check,
        not arguments.no_log_update,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
