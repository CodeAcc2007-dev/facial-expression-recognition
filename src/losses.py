from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        selected_log = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        selected_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        loss = -((1.0 - selected_probability) ** self.gamma) * selected_log
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def sqrt_inverse_frequency_weights(counts: list[int]) -> torch.Tensor:
    values = torch.tensor(counts, dtype=torch.float32)
    weights = torch.rsqrt(values)
    return weights / weights.mean()


def build_loss(name: str, counts: list[int], label_smoothing: float = 0.0) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    weights = sqrt_inverse_frequency_weights(counts)
    if name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=2.0, weight=weights)
    raise ValueError(f"unsupported loss: {name}")

