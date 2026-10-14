from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import CLASS_NAMES, atomic_json_dump, sha256_lines


LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}


def decoded_pixel_hash(path: str | Path) -> tuple[str, tuple[int, int], str]:
    with Image.open(path) as image:
        gray = image.convert("L")
        payload = gray.tobytes()
        digest = hashlib.sha256()
        digest.update(f"{gray.width}x{gray.height}:L:".encode("ascii"))
        digest.update(payload)
        return digest.hexdigest(), gray.size, image.format or "UNKNOWN"


def _validate_manifest(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(frame.columns) != expected:
        raise ValueError(f"{name} columns must be {expected}, found {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError(f"{name} contains nulls")
    if not frame["id"].is_unique:
        raise ValueError(f"{name} IDs are not unique")


def _select_grouped_validation(
    frame: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(frame["pixel_hash"]):
        by_hash[str(digest)].append(index)
    rng = random.Random(seed)
    validation_indices: set[int] = set()
    for label in CLASS_NAMES:
        class_indices = set(frame.index[frame["label"] == label].tolist())
        candidate_groups: list[list[int]] = []
        for indices in by_hash.values():
            labels = {str(frame.iloc[index]["label"]) for index in indices}
            if labels == {label}:
                candidate_groups.append(indices)
        rng.shuffle(candidate_groups)
        target = round(len(class_indices) * validation_fraction)
        chosen = 0
        deferred: list[list[int]] = []
        for indices in candidate_groups:
            if chosen + len(indices) <= target:
                validation_indices.update(indices)
                chosen += len(indices)
            else:
                deferred.append(indices)
        if chosen < target:
            for indices in sorted(deferred, key=len):
                if chosen >= target:
                    break
                validation_indices.update(indices)
                chosen += len(indices)
    validation = frame.iloc[sorted(validation_indices)].copy()
    training = frame.drop(frame.index[list(validation_indices)]).copy()
    overlap = set(training["pixel_hash"]) & set(validation["pixel_hash"])
    if overlap:
        raise AssertionError("decoded-pixel hash leakage between train and validation")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def _pixel_statistics(paths: list[Path]) -> tuple[float, float]:
    total = 0.0
    squared = 0.0
    count = 0
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        total += float(pixels.sum())
        squared += float(np.square(pixels).sum())
        count += pixels.size
    mean = total / count
    variance = max(0.0, squared / count - mean * mean)
    return mean, variance**0.5


def audit_and_split(
    data_dir: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    _validate_manifest(train, ["id", "label"], "train.csv")
    _validate_manifest(test, ["id"], "test.csv")
    _validate_manifest(sample, ["id", "label"], "sample_submission.csv")
    if list(test["id"]) != list(sample["id"]):
        raise ValueError("sample_submission IDs do not exactly match test.csv order")
    unknown = sorted(set(train["label"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    integrity = Counter()
    train_hashes: list[str] = []
    for identifier in train["id"]:
        path = data_dir / "train" / str(identifier)
        if not path.is_file():
            integrity["missing_train"] += 1
            train_hashes.append("")
            continue
        digest, size, image_format = decoded_pixel_hash(path)
        train_hashes.append(digest)
        integrity["unexpected_dimensions"] += int(size != (48, 48))
        integrity["unexpected_format"] += int(image_format != "JPEG")
    test_hashes: list[str] = []
    for identifier in test["id"]:
        path = data_dir / "test" / str(identifier)
        if not path.is_file():
            integrity["missing_test"] += 1
            test_hashes.append("")
            continue
        digest, size, image_format = decoded_pixel_hash(path)
        test_hashes.append(digest)
        integrity["unexpected_dimensions"] += int(size != (48, 48))
        integrity["unexpected_format"] += int(image_format != "JPEG")
    if any(integrity.values()):
        raise ValueError(f"dataset integrity failed: {dict(integrity)}")

    train = train.assign(pixel_hash=train_hashes)
    test = test.assign(pixel_hash=test_hashes)
    training, validation = _select_grouped_validation(train, validation_fraction, seed)
    split_hash = sha256_lines(
        [f"train,{identifier}" for identifier in sorted(training["id"])]
        + [f"validation,{identifier}" for identifier in sorted(validation["id"])]
    )
    training = training.assign(split="train")
    validation = validation.assign(split="validation")
    training.to_csv(output_dir / "train_split.csv", index=False)
    validation.to_csv(output_dir / "validation_split.csv", index=False)
    test.to_csv(output_dir / "test_manifest.csv", index=False)

    mean, std = _pixel_statistics([data_dir / "train" / str(value) for value in training["id"]])
    all_groups: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    for row in train.itertuples(index=False):
        all_groups[row.pixel_hash].append(("train", row.id, row.label))
    for row in test.itertuples(index=False):
        all_groups[row.pixel_hash].append(("test", row.id, None))
    duplicate_groups = [members for members in all_groups.values() if len(members) > 1]
    conflicting = sum(
        1 for members in duplicate_groups if len({label for split, _, label in members if split == "train"}) > 1
    )
    crossing = sum(
        1 for members in duplicate_groups if {split for split, _, _ in members} == {"train", "test"}
    )
    summary = {
        "status": "passed",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_hash": split_hash,
        "dataset": {
            "train_rows": len(train),
            "test_rows": len(test),
            "class_counts": {name: int((train["label"] == name).sum()) for name in CLASS_NAMES},
            "image_size": [48, 48],
            "mode": "L",
        },
        "split": {
            "strategy": "class-stratified grouped holdout by decoded-pixel SHA-256",
            "train_rows": len(training),
            "validation_rows": len(validation),
            "exact_hash_overlap": 0,
        },
        "normalization": {"source": "training partition only", "mean": mean, "std": std},
        "duplicates": {
            "all_exact_duplicate_groups": len(duplicate_groups),
            "conflicting_training_label_groups": conflicting,
            "groups_crossing_supplied_train_test": crossing,
        },
        "test_policy": "test hashes audited for overlap disclosure only; no labels or predictions used for training",
    }
    atomic_json_dump(summary, output_dir / "audit_summary.json")
    return summary


class FERDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, transform: Any, labelled: bool = True):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labelled = labelled

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.image_dir / str(row["id"])) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        identifier = str(row["id"])
        if self.labelled:
            return tensor, LABEL_TO_INDEX[str(row["label"])], identifier
        return tensor, identifier


def make_transform(
    input_size: int,
    training: bool,
    normalization: dict[str, Any],
    augmentation: bool,
):
    operations: list[Any] = [transforms.Resize((input_size, input_size))]
    if training and augmentation:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.96, 1.04)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    operations.extend([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    if normalization["mode"] == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif normalization["mode"] == "training_grayscale":
        mean = (float(normalization["mean"]),) * 3
        std = (float(normalization["std"]),) * 3
    else:
        raise ValueError(f"unsupported normalization mode: {normalization['mode']}")
    operations.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(operations)


def build_loaders(
    data_dir: str | Path,
    split_dir: str | Path,
    batch_size: int,
    input_size: int,
    normalization_mode: str,
    augmentation: bool,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    train_frame = pd.read_csv(split_dir / "train_split.csv")
    validation_frame = pd.read_csv(split_dir / "validation_split.csv")
    audit = json.loads((split_dir / "audit_summary.json").read_text(encoding="utf-8"))
    normalization = {"mode": normalization_mode, **audit["normalization"]}
    generator = torch.Generator().manual_seed(seed)
    train_dataset = FERDataset(
        train_frame,
        data_dir / "train",
        make_transform(input_size, True, normalization, augmentation),
    )
    validation_dataset = FERDataset(
        validation_frame,
        data_dir / "train",
        make_transform(input_size, False, normalization, False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return train_loader, validation_loader, {"audit": audit, "normalization": normalization}

