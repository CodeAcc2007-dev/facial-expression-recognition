from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .infer import _atomic_csv, validate_submission
from .utils import CLASS_NAMES, atomic_json_dump, sha256_file


PROBABILITY_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]


def average_probability_files(
    probability_paths: list[str | Path], test_csv: str | Path, output_path: str | Path
) -> dict:
    if len(probability_paths) < 2:
        raise ValueError("an ensemble requires at least two probability files")
    test_ids = list(pd.read_csv(test_csv)["id"])
    matrices: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    source_inference: dict[str, dict] = {}
    for raw_path in probability_paths:
        path = Path(raw_path).resolve()
        frame = pd.read_csv(path)
        if list(frame.columns) != ["id", *PROBABILITY_COLUMNS]:
            raise ValueError(f"unexpected probability schema in {path}")
        if list(frame["id"]) != test_ids:
            raise ValueError(f"test ID order mismatch in {path}")
        values = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if not np.allclose(values.sum(axis=1), 1.0, atol=2e-6):
            raise ValueError(f"probability rows do not sum to one in {path}")
        matrices.append(values)
        source_hashes[str(path)] = sha256_file(path)
        source_summary_path = path.with_name(path.name.replace("_probabilities.csv", "_summary.json"))
        if not source_summary_path.exists():
            raise FileNotFoundError(f"missing inference summary for ensemble source: {source_summary_path}")
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary.get("exact_test_id_order") is not True:
            raise ValueError(f"source did not verify exact test ID order: {source_summary_path}")
        if source_summary.get("repeat_max_absolute_probability_difference") != 0.0:
            raise ValueError(f"source did not pass deterministic repeat inference: {source_summary_path}")
        source_inference[str(path)] = {
            "summary": str(source_summary_path),
            "summary_sha256": sha256_file(source_summary_path),
            "checkpoint": source_summary.get("checkpoint"),
            "checkpoint_sha256": source_summary.get("checkpoint_sha256"),
            "tta_horizontal_flip": source_summary.get("tta_horizontal_flip"),
        }
    averaged = np.mean(matrices, axis=0)
    labels = [CLASS_NAMES[index] for index in averaged.argmax(axis=1)]
    submission = pd.DataFrame({"id": test_ids, "label": labels})
    validate_submission(submission, test_ids)
    output_path = Path(output_path).resolve()
    _atomic_csv(submission, output_path)
    evidence_path = output_path.with_name(output_path.stem + "_probabilities.csv")
    _atomic_csv(
        pd.DataFrame({"id": test_ids, **{column: averaged[:, index] for index, column in enumerate(PROBABILITY_COLUMNS)}}),
        evidence_path,
    )
    summary = {
        "status": "passed",
        "sources": source_hashes,
        "source_inference": source_inference,
        "rows": len(submission),
        "columns": list(submission.columns),
        "exact_test_id_order": True,
        "repeat_max_absolute_probability_difference": 0.0,
        "submission": str(output_path),
        "submission_sha256": sha256_file(output_path),
        "probability_evidence": str(evidence_path),
        "label_counts": {name: int((submission.label == name).sum()) for name in CLASS_NAMES},
    }
    atomic_json_dump(summary, output_path.with_name(output_path.stem + "_summary.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", nargs="+", required=True)
    parser.add_argument("--test-csv", default="data/raw/fer_challenge/test.csv")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(json.dumps(average_probability_files(arguments.probabilities, arguments.test_csv, arguments.output), indent=2))


if __name__ == "__main__":
    main()
