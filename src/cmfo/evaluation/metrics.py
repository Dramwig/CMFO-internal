from __future__ import annotations

import torch
from torch import Tensor


def classification_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected logits [batch, classes] and labels [batch].")
    return (logits.argmax(dim=-1) == labels).float().mean()


def masked_binary_iou(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Dense logits, targets, and mask must share a shape.")
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    predictions = predictions & mask
    truth = truth & mask
    intersection = (predictions & truth).sum(dim=1).float()
    union = (predictions | truth).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union)).mean()


def expected_calibration_error(logits: Tensor, labels: Tensor, bins: int = 10) -> Tensor:
    if bins <= 1:
        raise ValueError("bins must be greater than one.")
    probabilities = torch.softmax(logits, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).float()
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    result = torch.zeros((), device=logits.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            fraction = in_bin.float().mean()
            result = result + fraction * (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
    return result


def mean_squared_consistency(
    prediction: Tensor, reference: Tensor, mask: Tensor | None = None
) -> Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must share a shape.")
    squared = (prediction - reference).square()
    if mask is None:
        return squared.mean()
    if mask.shape != prediction.shape[: mask.ndim]:
        raise ValueError("mask must match the leading prediction dimensions.")
    expanded_mask = mask
    while expanded_mask.ndim < squared.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    numerator = torch.where(expanded_mask, squared, torch.zeros_like(squared)).sum()
    denominator = expanded_mask.expand_as(squared).sum().clamp_min(1)
    return numerator / denominator
