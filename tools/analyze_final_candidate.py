from __future__ import annotations

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

from src.utils import CLASS_NAMES, atomic_json_dump, classification_metrics, expected_calibration_error, plot_confusion_matrices


def main() -> None:
    candidate = json.loads((PROJECT_ROOT / "outputs" / "selected_candidate.json").read_text(encoding="utf-8"))["selected"]
    if candidate["type"] == "ensemble":
        prediction_path = PROJECT_ROOT / "outputs" / "analysis" / candidate["candidate_id"] / "validation_predictions.csv"
    else:
        suffix = "validation_tta_predictions.csv" if candidate["tta"] else "validation_predictions.csv"
        prediction_path = PROJECT_ROOT / "outputs" / "checkpoints" / candidate["members"][0] / suffix
    frame = pd.read_csv(prediction_path)
    truth = (
        frame["true_index"].to_numpy(dtype=int)
        if "true_index" in frame
        else frame["true_label"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)
    )
    probabilities = frame[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
    predicted = probabilities.argmax(axis=1)
    metrics = classification_metrics(truth, predicted)
    ece = expected_calibration_error(probabilities, truth)
    output_dir = PROJECT_ROOT / "outputs" / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(np.asarray(metrics["confusion_matrix"]), output_dir / "confusion_matrices.png")

    confidence = probabilities.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, 11)
    bin_rows = []
    for index in range(10):
        mask = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < 9 else confidence <= edges[index + 1])
        if mask.any():
            bin_rows.append(
                {
                    "lower": edges[index],
                    "upper": edges[index + 1],
                    "mean_confidence": float(confidence[mask].mean()),
                    "observed_accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 5.5))
    axis.plot([0, 1], [0, 1], "--", color="#777", label="perfect calibration")
    axis.plot(bins.mean_confidence, bins.observed_accuracy, marker="o", color="#5A78F0", label=f"candidate (ECE={ece:.3f})")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy", title="Final-candidate reliability")
    axis.grid(alpha=.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(metrics["confusion_matrix"])
    confusion_rows = [
        {"true_label": CLASS_NAMES[y], "predicted_label": CLASS_NAMES[x], "count": int(matrix[y, x])}
        for y in range(len(CLASS_NAMES))
        for x in range(len(CLASS_NAMES))
        if y != x
    ]
    confusions = pd.DataFrame(confusion_rows).sort_values("count", ascending=False)
    confusions.to_csv(output_dir / "top_confusions.csv", index=False)
    top_pairs = confusions.head(3).to_dict(orient="records")
    figure, axes = plt.subplots(3, 10, figsize=(19, 7.2), constrained_layout=True)
    gallery_rows = []
    rng = np.random.default_rng(42)
    for row_index, pair in enumerate(top_pairs):
        matches = frame[(frame.true_label == pair["true_label"]) & (frame.predicted_label == pair["predicted_label"])]
        if len(matches) > 10:
            matches = matches.iloc[rng.choice(len(matches), size=10, replace=False)]
        for column_index, (_, sample) in enumerate(matches.iterrows()):
            with Image.open(PROJECT_ROOT / "data" / "raw" / "fer_challenge" / "train" / sample.id) as image:
                axes[row_index, column_index].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].axis("off")
            gallery_rows.append({"id": sample.id, "true_label": sample.true_label, "predicted_label": sample.predicted_label, "review_status": "unreviewed; descriptive only"})
        for column_index in range(len(matches), 10):
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"true {pair['true_label']}\npredicted {pair['predicted_label']}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            color="#263247",
        )
    figure.suptitle("Final candidate: major validation confusions")
    figure.savefig(output_dir / "misclassification_gallery.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(gallery_rows).to_csv(output_dir / "misclassification_gallery_manifest.csv", index=False)
    summary = {
        "candidate": candidate,
        "prediction_source": str(prediction_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "expected_calibration_error_10_bins": ece,
        "top_confusions": top_pairs,
        "interpretation_boundary": "Possible ambiguity is descriptive only; organizer labels remain unchanged.",
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
