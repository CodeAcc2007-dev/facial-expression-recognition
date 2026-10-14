from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import _select_grouped_validation, make_transform
from src.infer import validate_submission
from src.losses import sqrt_inverse_frequency_weights
from src.utils import CLASS_NAMES, append_experiment, classification_metrics, expected_calibration_error, record_submission_hash


def test_accuracy_and_macro_f1_are_computed_independently() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2, 3])
    prediction = np.array([0, 0, 0, 1, 2, 1, 3])
    result = classification_metrics(truth, prediction)
    assert result["accuracy"] == pytest.approx(5 / 7)
    assert 0 < result["macro_f1"] < result["accuracy"]


def test_grouped_split_never_separates_exact_duplicates() -> None:
    rows = []
    for label in CLASS_NAMES:
        for index in range(12):
            digest = f"{label}-{index // 2}" if index < 4 else f"{label}-single-{index}"
            rows.append({"id": f"{label}-{index}.jpg", "label": label, "pixel_hash": digest})
    frame = pd.DataFrame(rows)
    training, validation = _select_grouped_validation(frame, 0.2, 42)
    assert set(training["pixel_hash"]).isdisjoint(set(validation["pixel_hash"]))
    assert set(validation["label"]) == set(CLASS_NAMES)


def test_conflicting_duplicate_group_stays_out_of_validation() -> None:
    frame = pd.DataFrame(
        [
            {"id": "a.jpg", "label": "angry", "pixel_hash": "conflict"},
            {"id": "b.jpg", "label": "happy", "pixel_hash": "conflict"},
        ]
        + [
            {"id": f"{label}-{index}.jpg", "label": label, "pixel_hash": f"{label}-{index}"}
            for label in CLASS_NAMES
            for index in range(10)
        ]
    )
    training, validation = _select_grouped_validation(frame, 0.2, 7)
    assert "conflict" in set(training["pixel_hash"])
    assert "conflict" not in set(validation["pixel_hash"])


def test_submission_contract_requires_exact_order_and_vocabulary() -> None:
    frame = pd.DataFrame({"id": ["test_1.jpg", "test_2.jpg"], "label": ["happy", "sad"]})
    validate_submission(frame, ["test_1.jpg", "test_2.jpg"])
    with pytest.raises(ValueError):
        validate_submission(frame.iloc[::-1].reset_index(drop=True), ["test_1.jpg", "test_2.jpg"])


def test_sqrt_weights_upweight_the_minority_class() -> None:
    weights = sqrt_inverse_frequency_weights([100, 4])
    assert float(weights[1]) > float(weights[0])
    assert float(weights.mean()) == pytest.approx(1.0)


def test_transform_shape_and_calibration_contract() -> None:
    from PIL import Image

    image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
    transform = make_transform(96, False, {"mode": "imagenet", "mean": 0.5, "std": 0.25}, False)
    assert transform(image).shape == torch.Size([3, 96, 96])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6]])
    assert 0 <= expected_calibration_error(probabilities, np.array([0, 1])) <= 1


def test_submission_hash_is_recorded_against_exact_run(tmp_path) -> None:
    log = tmp_path / "experiments.csv"
    append_experiment(log, {"run_id": "r1", "val_acc": "0.5"})
    append_experiment(log, {"run_id": "r2", "val_acc": "0.6"})
    record_submission_hash(log, "r1", "abc123")
    frame = pd.read_csv(log)
    assert frame.loc[frame.run_id == "r1", "submission_hash"].item() == "abc123"
    assert pd.isna(frame.loc[frame.run_id == "r2", "submission_hash"]).item()
