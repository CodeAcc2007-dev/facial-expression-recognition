from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
EXPERIMENT_FIELDS = (
    "run_id",
    "timestamp",
    "config",
    "backbone",
    "input_size",
    "seed",
    "split_hash",
    "loss",
    "aug",
    "optimizer",
    "lr",
    "epoch_budget",
    "best_epoch",
    "runtime_min",
    "val_acc",
    "val_macro_f1",
    "per_class_f1_json",
    "checkpoint_path",
    "checkpoint_hash",
    "submission_hash",
    "hypothesis",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for truth, prediction in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if not (0 <= truth < classes and 0 <= prediction < classes):
            raise ValueError("class index outside expected range")
        matrix[truth, prediction] += 1
    return matrix


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...] = CLASS_NAMES
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal size")
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    accuracy = float(true_positive.sum() / matrix.sum()) if matrix.sum() else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must be [samples, classes] and align with labels")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def plot_confusion_matrices(
    matrix: np.ndarray, output_path: str | Path, class_names: tuple[str, ...] = CLASS_NAMES
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    row = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0)
    column = np.divide(matrix, matrix.sum(axis=0, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=0, keepdims=True) > 0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for axis, values, title in zip(
        axes, (row, column), ("Row-normalized (recall)", "Column-normalized (precision)"), strict=True
    ):
        image = axis.imshow(values, cmap="magma", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                axis.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if values[y, x] < 0.55 else "black")
    figure.colorbar(image, ax=axes, shrink=0.82)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def append_experiment(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    unexpected = set(row) - set(EXPERIMENT_FIELDS)
    if unexpected:
        raise ValueError(f"unexpected experiment fields: {sorted(unexpected)}")
    exists = destination.exists() and destination.stat().st_size > 0
    normalized = {field: row.get(field, "") for field in EXPERIMENT_FIELDS}
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def record_submission_hash(path: str | Path, run_id: str, submission_hash: str) -> None:
    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(destination)
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [index for index, row in enumerate(rows) if row.get("run_id") == run_id]
    if not matches:
        raise ValueError(f"run_id {run_id!r} is absent from experiment log")
    rows[matches[-1]]["submission_hash"] = submission_hash
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
