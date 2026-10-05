from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from cmfo.data.fields import MultimodalBatch
from cmfo.models.common import MultimodalOutput


@dataclass
class LossBreakdown:
    total: Tensor
    classification: Tensor
    dense: Tensor
    consistency: Tensor


def dense_binary_cross_entropy(output: MultimodalOutput, batch: MultimodalBatch) -> Tensor:
    element_loss = functional.binary_cross_entropy_with_logits(
        output.dense_logits,
        batch.queries.targets,
        reduction="none",
    )
    mask = batch.queries.mask.to(element_loss.dtype)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)


def discretization_consistency_loss(
    first: MultimodalOutput,
    second: MultimodalOutput,
    dense_mask: Tensor,
) -> Tensor:
    class_loss = functional.mse_loss(
        torch.softmax(first.class_logits, dim=-1),
        torch.softmax(second.class_logits, dim=-1),
    )
    dense_difference = (
        torch.sigmoid(first.dense_logits) - torch.sigmoid(second.dense_logits)
    ).square()
    mask = dense_mask.to(dense_difference.dtype)
    dense_loss = (dense_difference * mask).sum() / mask.sum().clamp_min(1.0)
    return class_loss + dense_loss


def multitask_loss(
    output: MultimodalOutput,
    batch: MultimodalBatch,
    dense_weight: float,
    consistency_weight: float = 0.0,
    second_output: MultimodalOutput | None = None,
) -> LossBreakdown:
    classification = functional.cross_entropy(output.class_logits, batch.class_labels)
    dense = dense_binary_cross_entropy(output, batch)
    consistency = torch.zeros((), device=output.class_logits.device)
    if second_output is not None and consistency_weight > 0:
        consistency = discretization_consistency_loss(
            output,
            second_output,
            batch.queries.mask,
        )
    total = classification + dense_weight * dense + consistency_weight * consistency
    return LossBreakdown(total, classification, dense, consistency)
