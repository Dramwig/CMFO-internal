from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from cmfo.data.fields import GeometryBatch, pack_geometry

SUPPORTED_LAYOUTS = {"regular", "jittered", "random"}


def make_1d_coordinates(
    count: int,
    layout: str = "regular",
    generator: torch.Generator | None = None,
) -> Tensor:
    if count <= 0:
        raise ValueError("count must be positive.")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout!r}.")
    if layout == "random":
        return torch.rand(count, 1, generator=generator)
    base = (torch.arange(count, dtype=torch.float32) + 0.5) / float(count)
    if layout == "jittered":
        jitter = (torch.rand(count, generator=generator) - 0.5) / float(count)
        base = (base + jitter).clamp(0.0, 1.0)
    return base.unsqueeze(-1)


def make_2d_coordinates(
    count: int,
    layout: str = "regular",
    generator: torch.Generator | None = None,
) -> Tensor:
    if count <= 0:
        raise ValueError("count must be positive.")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout!r}.")
    if layout == "random":
        return torch.rand(count, 2, generator=generator)
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError("Regular and jittered 2D quadrature counts must be perfect squares.")
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) / float(side)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    if layout == "jittered":
        jitter = (torch.rand(count, 2, generator=generator) - 0.5) / float(side)
        coords = (coords + jitter).clamp(0.0, 1.0)
    return coords


def make_geometry_batch(
    batch_size: int,
    counts: int | Sequence[int],
    coordinate_dim: int,
    layout: str,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> GeometryBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(counts, int):
        counts = [counts] * batch_size
    else:
        counts = list(counts)
    if len(counts) != batch_size:
        raise ValueError("counts must contain one value per batch element.")
    builder = make_1d_coordinates if coordinate_dim == 1 else make_2d_coordinates
    if coordinate_dim not in {1, 2}:
        raise ValueError("Only one- and two-dimensional native domains are supported.")
    coordinates = [builder(int(count), layout, generator) for count in counts]
    geometry = pack_geometry(coordinates)
    return geometry if device is None else geometry.to(device)


def integrate(values: Tensor, geometry: GeometryBatch) -> Tensor:
    if values.shape[:2] != geometry.mask.shape:
        raise ValueError("values and geometry must share [batch, nodes].")
    weights = geometry.weights
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1)


def normalized_field_l2(
    prediction: Tensor,
    reference: Tensor,
    mask: Tensor,
    epsilon: float = 1.0e-8,
) -> Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must share a shape.")
    expanded_mask = mask
    while expanded_mask.ndim < prediction.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    difference = torch.where(expanded_mask, prediction - reference, torch.zeros_like(prediction))
    reference_masked = torch.where(expanded_mask, reference, torch.zeros_like(reference))
    numerator = difference.square().sum(dim=tuple(range(1, difference.ndim))).sqrt()
    denominator = reference_masked.square().sum(dim=tuple(range(1, reference_masked.ndim))).sqrt()
    return numerator / denominator.clamp_min(epsilon)


def interpolate_field(
    values: Tensor,
    source_geometry: GeometryBatch,
    target_geometry: GeometryBatch,
    bandwidth: float | None = None,
) -> Tensor:
    """Kernel interpolation used only to compare fields on a common probe grid."""
    if values.shape[:2] != source_geometry.mask.shape:
        raise ValueError("values and source_geometry must share [batch, nodes].")
    if source_geometry.coordinate_dim != target_geometry.coordinate_dim:
        raise ValueError("Source and target fields must share a native coordinate dimension.")
    if source_geometry.batch_size != target_geometry.batch_size:
        raise ValueError("Source and target fields must share a batch size.")
    dimension = source_geometry.coordinate_dim
    if bandwidth is None:
        effective_count = source_geometry.lengths.float().mean().clamp_min(1)
        bandwidth = float(effective_count.pow(-1.0 / float(dimension)).item()) * 1.5
    distance_squared = (
        (target_geometry.coords.unsqueeze(2) - source_geometry.coords.unsqueeze(1))
        .square()
        .sum(dim=-1)
    )
    scores = -distance_squared / (2.0 * bandwidth**2)
    scores = scores + source_geometry.weights.clamp_min(1.0e-12).log().unsqueeze(1)
    pair_mask = source_geometry.mask.unsqueeze(1)
    scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
    interpolation = torch.softmax(scores, dim=-1)
    interpolation = torch.where(pair_mask, interpolation, torch.zeros_like(interpolation))
    output = torch.einsum("bts,bsh->bth", interpolation, values)
    return output * target_geometry.mask.unsqueeze(-1).to(output.dtype)
