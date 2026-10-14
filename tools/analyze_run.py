from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CLASS_NAMES, atomic_json_dump, classification_metrics, expected_calibration_error


def analyze(run_id: str) -> dict:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / run_id
    predictions = pd.read_csv(checkpoint_dir / "validation_predictions.csv")
    probability_columns = [f"prob_{name}" for name in CLASS_NAMES]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    truth = predictions["true_index"].to_numpy(dtype=int)
    predicted = predictions["predicted_index"].to_numpy(dtype=int)
    metrics = classification_metrics(truth, predicted)
    ece = expected_calibration_error(probabilities, truth, bins=10)
    output_dir = PROJECT_ROOT / "outputs" / "analysis" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    confidence = probabilities.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, 11)
    centers: list[float] = []
    observed: list[float] = []
    mean_confidence: list[float] = []
    counts: list[int] = []
    for index in range(10):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < 9 else confidence <= edges[index + 1]
        )
        if mask.any():
            centers.append((edges[index] + edges[index + 1]) / 2)
            observed.append(float(correct[mask].mean()))
            mean_confidence.append(float(confidence[mask].mean()))
            counts.append(int(mask.sum()))
    calibration = pd.DataFrame(
        {"bin_center": centers, "mean_confidence": mean_confidence, "observed_accuracy": observed, "count": counts}
    )
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777", label="perfect calibration")
    axis.plot(mean_confidence, observed, marker="o", color="#5A78F0", label=f"model (ECE={ece:.3f})")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy", title="Validation reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(metrics["confusion_matrix"])
    confusion_rows = []
    for true_index, true_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            if true_index != predicted_index:
                confusion_rows.append(
                    {"true_label": true_name, "predicted_label": predicted_name, "count": int(matrix[true_index, predicted_index])}
                )
    confusions = pd.DataFrame(confusion_rows).sort_values("count", ascending=False)
    confusions.to_csv(output_dir / "top_confusions.csv", index=False)

    top_pairs = confusions.head(3).to_dict(orient="records")
    figure, axes = plt.subplots(len(top_pairs), 10, figsize=(19, 2.4 * len(top_pairs)))
    axes = np.atleast_2d(axes)
    gallery_rows = []
    rng = np.random.default_rng(42)
    for row_index, pair in enumerate(top_pairs):
        matches = predictions[
            (predictions.true_label == pair["true_label"])
            & (predictions.predicted_label == pair["predicted_label"])
        ]
        if len(matches) > 10:
            matches = matches.iloc[rng.choice(len(matches), size=10, replace=False)]
        for column_index, (_, sample) in enumerate(matches.iterrows()):
            with Image.open(PROJECT_ROOT / "data" / "raw" / "fer_challenge" / "train" / sample.id) as image:
                axes[row_index, column_index].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].axis("off")
            axes[row_index, column_index].set_title(f"p={probabilities[sample.name].max():.2f}", fontsize=7)
            gallery_rows.append(
                {"id": sample.id, "true_label": sample.true_label, "predicted_label": sample.predicted_label, "review_status": "unreviewed; describe ambiguity only"}
            )
        for column_index in range(len(matches), 10):
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{pair['true_label']} → {pair['predicted_label']}", rotation=90, fontsize=9
        )
    figure.suptitle("Major validation confusions — labels unchanged", y=1.0)
    figure.tight_layout()
    figure.savefig(output_dir / "misclassification_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(gallery_rows).to_csv(output_dir / "misclassification_gallery_manifest.csv", index=False)

    summary = {
        "run_id": run_id,
        "metrics": metrics,
        "expected_calibration_error_10_bins": ece,
        "top_confusions": top_pairs,
        "interpretation_boundary": "Gallery cases are hypotheses for participant review, not definitive mislabels; labels are unchanged.",
    }
    atomic_json_dump(summary, output_dir / "analysis_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    analyze(arguments.run_id)


if __name__ == "__main__":
    main()

