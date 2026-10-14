from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


SUPPORTED_BACKBONES = ("resnet18",)


def create_model(
    backbone: str,
    classes: int,
    pretrained: bool,
    dropout: float,
    torch_hub_dir: str | Path | None = None,
) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"unsupported backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}")
    if torch_hub_dir is not None:
        torch.hub.set_dir(str(Path(torch_hub_dir)))
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, classes))
    return model


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float):
    head_ids = {id(parameter) for parameter in model.fc.parameters()}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return [
        {"params": backbone_parameters, "lr": backbone_lr},
        {"params": list(model.fc.parameters()), "lr": head_lr},
    ]

