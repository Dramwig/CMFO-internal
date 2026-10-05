import torch

from cmfo.evaluation.metrics import (
    classification_accuracy,
    expected_calibration_error,
    masked_binary_iou,
    mean_squared_consistency,
)


def test_classification_and_iou_metrics() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1])
    assert classification_accuracy(logits, labels).item() == 1.0
    dense_logits = torch.tensor([[4.0, -4.0, 4.0], [-4.0, 4.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    assert masked_binary_iou(dense_logits, targets, mask).item() == 1.0


def test_calibration_and_consistency_are_finite() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    labels = torch.tensor([0, 1])
    assert torch.isfinite(expected_calibration_error(logits, labels, bins=4))
    assert mean_squared_consistency(logits, logits).item() == 0.0
