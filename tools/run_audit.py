from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import audit_and_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/fer_challenge")
    parser.add_argument("--output-dir", default="outputs/eda")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    summary = audit_and_split(
        PROJECT_ROOT / arguments.data_dir,
        PROJECT_ROOT / arguments.output_dir,
        arguments.validation_fraction,
        arguments.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

