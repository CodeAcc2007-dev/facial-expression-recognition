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
import torch
from PIL import Image
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import make_transform
from src.model import create_model
from src.utils import CLASS_NAMES, atomic_json_dump


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __call__(self, tensor: torch.Tensor, target_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[0, target_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    run_dir = PROJECT_ROOT / "outputs" / "checkpoints" / arguments.run_id
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    model = create_model(config["backbone"], len(CLASS_NAMES), False, float(config.get("dropout", 0.2)))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = make_transform(int(config["input_size"]), False, checkpoint["normalization"], False)
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")
    rows = []
    for label in CLASS_NAMES:
        correct = predictions[(predictions.true_label == label) & (predictions.predicted_label == label)]
        incorrect = predictions[(predictions.true_label == label) & (predictions.predicted_label != label)]
        if not correct.empty:
            rows.append((label, "correct", correct.iloc[0]))
        if not incorrect.empty:
            rows.append((label, "incorrect", incorrect.iloc[0]))

    cam_engine = GradCAM(model, model.layer4[-1].conv2)
    figure, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(11, 17), constrained_layout=True)
    manifest = []
    by_key = {(label, status): sample for label, status, sample in rows}
    for row_index, label in enumerate(CLASS_NAMES):
        for pair_index, status in enumerate(("correct", "incorrect")):
            original_axis = axes[row_index, pair_index * 2]
            cam_axis = axes[row_index, pair_index * 2 + 1]
            sample = by_key.get((label, status))
            if sample is None:
                original_axis.axis("off")
                cam_axis.axis("off")
                continue
            path = PROJECT_ROOT / "data" / "raw" / "fer_challenge" / "train" / sample.id
            with Image.open(path) as image:
                gray_image = image.convert("L")
                gray = np.asarray(gray_image.resize((int(config["input_size"]), int(config["input_size"]))))
                tensor = transform(gray_image).unsqueeze(0)
            predicted_index = CLASS_NAMES.index(sample.predicted_label)
            cam = cam_engine(tensor, predicted_index)
            original_axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
            original_axis.set_title(f"{status}: {sample.id}", fontsize=7)
            cam_axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
            cam_axis.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
            cam_axis.set_title(f"Grad-CAM → {sample.predicted_label}", fontsize=8)
            original_axis.axis("off")
            cam_axis.axis("off")
            if pair_index == 0:
                original_axis.set_ylabel(f"true: {label}", fontsize=9)
            manifest.append(
                {"id": sample.id, "true_label": label, "predicted_label": sample.predicted_label, "status": status, "cam_target": sample.predicted_label}
            )
    cam_engine.close()
    figure.suptitle("Validation Grad-CAM — predicted-class attention", fontsize=14)
    output_dir = PROJECT_ROOT / "outputs" / "analysis" / arguments.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "gradcam_grid.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(manifest).to_csv(output_dir / "gradcam_manifest.csv", index=False)
    summary = {
        "run_id": arguments.run_id,
        "target_layer": "resnet18.layer4[-1].conv2",
        "target": "predicted class logit",
        "samples": len(manifest),
        "interpretation": "Qualitative attention evidence only; Grad-CAM is not a causal explanation.",
    }
    atomic_json_dump(summary, output_dir / "gradcam_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

