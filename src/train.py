from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import LABEL_TO_INDEX, build_loaders
from .losses import build_loss
from .model import create_model, parameter_groups
from .utils import (
    CLASS_NAMES,
    append_experiment,
    atomic_json_dump,
    classification_metrics,
    plot_confusion_matrices,
    set_seed,
    sha256_file,
    utc_now,
)


def _epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    identifiers_all: list[str] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, labels, identifiers) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            rows = labels.shape[0]
            total_loss += float(loss.detach()) * rows
            total_rows += rows
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(logits.detach().softmax(dim=1).cpu().numpy())
            identifiers_all.extend(list(identifiers))
    y_true = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    y_pred = probabilities.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred)
    return total_loss / total_rows, metrics, y_true, y_pred, probabilities, identifiers_all


def _plot_history(history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy (primary)")
    axes[2].plot(frame["epoch"], frame["train_macro_f1"], label="train")
    axes[2].plot(frame["epoch"], frame["val_macro_f1"], label="validation")
    axes[2].set_title("Macro F1 (diagnostic)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def train(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    run_id = config["run_id"]
    output_dir = project_root / "outputs" / "checkpoints" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, metadata = build_loaders(
        project_root / config["data_dir"],
        project_root / config["split_dir"],
        batch_size=int(config["batch_size"]),
        input_size=int(config["input_size"]),
        normalization_mode=config["normalization"],
        augmentation=bool(config["augmentation"]),
        workers=int(config.get("workers", 0)),
        seed=int(config["seed"]),
    )
    counts = [int((train_loader.dataset.frame["label"] == label).sum()) for label in CLASS_NAMES]
    model = create_model(
        config["backbone"],
        len(CLASS_NAMES),
        pretrained=bool(config["pretrained"]),
        dropout=float(config.get("dropout", 0.2)),
        torch_hub_dir=project_root / ".cache" / "torch" / "hub",
    ).to(device)
    criterion = build_loss(
        config["loss"], counts, label_smoothing=float(config.get("label_smoothing", 0.0))
    ).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(config["backbone_lr"]), float(config["head_lr"])),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config.get("minimum_lr", 1e-6))
    )

    history: list[dict[str, float]] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    patience = int(config.get("patience", config["epochs"]))
    stale = 0
    started = time.perf_counter()
    best_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss, train_metrics, *_ = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            config.get("max_train_batches"),
        )
        val_loss, val_metrics, y_true, y_pred, probabilities, identifiers = _epoch(
            model,
            validation_loader,
            criterion,
            device,
            None,
            config.get("max_validation_batches"),
        )
        key = (val_metrics["accuracy"], val_metrics["macro_f1"])
        record = {
            "epoch": epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = val_metrics
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "class_names": CLASS_NAMES,
                    "normalization": metadata["normalization"],
                    "split_hash": metadata["audit"]["split_hash"],
                    "selection_rule": "validation accuracy first, macro F1 second",
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                },
                best_path,
            )
            pd.DataFrame(
                {
                    "id": identifiers,
                    "true_index": y_true,
                    "true_label": [CLASS_NAMES[value] for value in y_true],
                    "predicted_index": y_pred,
                    "predicted_label": [CLASS_NAMES[value] for value in y_pred],
                    **{f"prob_{name}": probabilities[:, index] for index, name in enumerate(CLASS_NAMES)},
                }
            ).to_csv(output_dir / "validation_predictions.csv", index=False)
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break

    elapsed = time.perf_counter() - started
    if best_metrics is None:
        raise RuntimeError("no checkpoint was selected")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    _plot_history(history, output_dir / "training_curves.png")
    plot_confusion_matrices(
        np.asarray(best_metrics["confusion_matrix"]), output_dir / "confusion_matrices.png"
    )
    checkpoint_hash = sha256_file(best_path)
    summary = {
        "status": "completed",
        "run_id": run_id,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "selection_rule": "validation accuracy first, macro F1 second",
        "split_hash": metadata["audit"]["split_hash"],
        "normalization": metadata["normalization"],
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "runtime_seconds": elapsed,
        "checkpoint": str(best_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_hash,
        "test_set_used_for_training_or_selection": False,
        "selection_eligible": bool(config.get("selection_eligible", True)),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    append_experiment(
        project_root / "experiments" / "experiments.csv",
        {
            "run_id": run_id,
            "timestamp": utc_now(),
            "config": str(config_path.relative_to(project_root)),
            "backbone": config["backbone"],
            "input_size": config["input_size"],
            "seed": config["seed"],
            "split_hash": metadata["audit"]["split_hash"],
            "loss": f"{config['loss']};smoothing={config.get('label_smoothing', 0.0)}",
            "aug": config["augmentation"],
            "optimizer": "AdamW",
            "lr": f"backbone={config['backbone_lr']};head={config['head_lr']}",
            "epoch_budget": config["epochs"],
            "best_epoch": best_epoch,
            "runtime_min": f"{elapsed / 60:.3f}",
            "val_acc": f"{best_metrics['accuracy']:.8f}",
            "val_macro_f1": f"{best_metrics['macro_f1']:.8f}",
            "per_class_f1_json": json.dumps(
                {name: values["f1"] for name, values in best_metrics["per_class"].items()},
                sort_keys=True,
            ),
            "checkpoint_path": str(best_path.relative_to(project_root)),
            "checkpoint_hash": checkpoint_hash,
            "hypothesis": config.get("hypothesis", ""),
            "notes": config.get("notes", ""),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    summary = train(arguments.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
