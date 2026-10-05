from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GeometryBatch:
    coords: Tensor
    weights: Tensor
    mask: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.coords.ndim != 3:
            raise ValueError("coords must have shape [batch, nodes, coordinate_dim].")
        if self.weights.shape != self.coords.shape[:2]:
            raise ValueError("weights must have shape [batch, nodes].")
        if self.mask.shape != self.coords.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch, nodes].")
        if self.lengths.shape != (self.coords.shape[0],):
            raise ValueError("lengths must have shape [batch].")
        if torch.any(self.lengths <= 0):
            raise ValueError("Every sample must contain at least one node.")

    @property
    def batch_size(self) -> int:
        return int(self.coords.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.coords.shape[1])

    @property
    def coordinate_dim(self) -> int:
        return int(self.coords.shape[2])

    def to(self, device: torch.device | str) -> GeometryBatch:
        return GeometryBatch(
            coords=self.coords.to(device),
            weights=self.weights.to(device),
            mask=self.mask.to(device),
            lengths=self.lengths.to(device),
        )

    def permuted(self, permutation: Tensor) -> GeometryBatch:
        if permutation.shape != self.mask.shape:
            raise ValueError("permutation must have shape [batch, nodes].")
        coordinate_index = permutation.unsqueeze(-1).expand_as(self.coords)
        return GeometryBatch(
            coords=self.coords.gather(1, coordinate_index),
            weights=self.weights.gather(1, permutation),
            mask=self.mask.gather(1, permutation),
            lengths=self.lengths,
        )


@dataclass
class FieldBatch:
    values: Tensor
    geometry: GeometryBatch

    def __post_init__(self) -> None:
        if self.values.shape[:2] != self.geometry.mask.shape:
            raise ValueError("values and geometry must share [batch, nodes].")

    @property
    def coords(self) -> Tensor:
        return self.geometry.coords

    @property
    def weights(self) -> Tensor:
        return self.geometry.weights

    @property
    def mask(self) -> Tensor:
        return self.geometry.mask

    @property
    def lengths(self) -> Tensor:
        return self.geometry.lengths

    def to(self, device: torch.device | str) -> FieldBatch:
        return FieldBatch(self.values.to(device), self.geometry.to(device))

    def permuted(self, permutation: Tensor) -> FieldBatch:
        value_index = permutation
        for _ in self.values.shape[2:]:
            value_index = value_index.unsqueeze(-1)
        value_index = value_index.expand_as(self.values)
        return FieldBatch(
            values=self.values.gather(1, value_index),
            geometry=self.geometry.permuted(permutation),
        )


@dataclass
class QueryBatch:
    coords: Tensor
    targets: Tensor
    mask: Tensor
    weights: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.coords.ndim != 3 or self.coords.shape[-1] != 2:
            raise ValueError("Query coordinates must have shape [batch, queries, 2].")
        if self.targets.shape != self.coords.shape[:2]:
            raise ValueError("Query targets must have shape [batch, queries].")
        if self.mask.shape != self.coords.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("Query mask must be boolean with shape [batch, queries].")
        if self.weights.shape != self.coords.shape[:2]:
            raise ValueError("Query weights must have shape [batch, queries].")

    @property
    def geometry(self) -> GeometryBatch:
        return GeometryBatch(self.coords, self.weights, self.mask, self.lengths)

    def to(self, device: torch.device | str) -> QueryBatch:
        return QueryBatch(
            coords=self.coords.to(device),
            targets=self.targets.to(device),
            mask=self.mask.to(device),
            weights=self.weights.to(device),
            lengths=self.lengths.to(device),
        )


@dataclass
class MultimodalBatch:
    text: FieldBatch
    image: FieldBatch
    queries: QueryBatch
    class_labels: Tensor
    sample_ids: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return int(self.class_labels.shape[0])

    def to(self, device: torch.device | str) -> MultimodalBatch:
        return MultimodalBatch(
            text=self.text.to(device),
            image=self.image.to(device),
            queries=self.queries.to(device),
            class_labels=self.class_labels.to(device),
            sample_ids=self.sample_ids,
        )


def _normalise_weights(weights: Tensor, mask: Tensor) -> Tensor:
    weights = torch.where(mask, weights.clamp_min(0), torch.zeros_like(weights))
    normalizer = weights.sum(dim=1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("Valid quadrature weights must have a positive sum.")
    return weights / normalizer


def pack_geometry(
    coordinates: Sequence[Tensor],
    weights: Sequence[Tensor] | None = None,
) -> GeometryBatch:
    if not coordinates:
        raise ValueError("Cannot pack an empty batch.")
    coordinate_dim = int(coordinates[0].shape[-1])
    if any(item.ndim != 2 or item.shape[-1] != coordinate_dim for item in coordinates):
        raise ValueError("All coordinate tensors must have shape [nodes, common_coordinate_dim].")
    lengths = torch.tensor([item.shape[0] for item in coordinates], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("Every sample must contain at least one coordinate.")
    max_nodes = int(lengths.max().item())
    batch_size = len(coordinates)
    dtype = coordinates[0].dtype
    coords = torch.zeros(batch_size, max_nodes, coordinate_dim, dtype=dtype)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    packed_weights = torch.zeros(batch_size, max_nodes, dtype=dtype)
    for index, sample_coords in enumerate(coordinates):
        length = sample_coords.shape[0]
        coords[index, :length] = sample_coords
        mask[index, :length] = True
        if weights is None:
            packed_weights[index, :length] = 1.0 / float(length)
        else:
            sample_weights = weights[index]
            if sample_weights.shape != (length,):
                raise ValueError("Each weight tensor must match its coordinate count.")
            packed_weights[index, :length] = sample_weights.to(dtype=dtype)
    packed_weights = _normalise_weights(packed_weights, mask)
    return GeometryBatch(coords, packed_weights, mask, lengths)


def pack_fields(
    values: Sequence[Tensor],
    coordinates: Sequence[Tensor],
    weights: Sequence[Tensor] | None = None,
    pad_value: float = 0,
) -> FieldBatch:
    if len(values) != len(coordinates):
        raise ValueError("values and coordinates must contain the same number of samples.")
    geometry = pack_geometry(coordinates, weights)
    trailing_shape = values[0].shape[1:]
    if any(item.ndim < 1 or item.shape[1:] != trailing_shape for item in values):
        raise ValueError("All values must share their trailing shape.")
    output_shape = (len(values), geometry.max_nodes, *trailing_shape)
    packed = torch.full(output_shape, pad_value, dtype=values[0].dtype)
    for index, sample_values in enumerate(values):
        if sample_values.shape[0] != coordinates[index].shape[0]:
            raise ValueError("Each value tensor must match its coordinate count.")
        packed[index, : sample_values.shape[0]] = sample_values
    return FieldBatch(packed, geometry)


def pack_queries(
    coordinates: Sequence[Tensor],
    targets: Sequence[Tensor],
    weights: Sequence[Tensor] | None = None,
) -> QueryBatch:
    if len(coordinates) != len(targets):
        raise ValueError("Query coordinates and targets must contain the same number of samples.")
    geometry = pack_geometry(coordinates, weights)
    packed_targets = torch.zeros(len(targets), geometry.max_nodes, dtype=torch.float32)
    for index, sample_targets in enumerate(targets):
        if sample_targets.shape != (coordinates[index].shape[0],):
            raise ValueError("Each query target tensor must match its coordinate count.")
        packed_targets[index, : sample_targets.shape[0]] = sample_targets.float()
    return QueryBatch(
        coords=geometry.coords,
        targets=packed_targets,
        mask=geometry.mask,
        weights=geometry.weights,
        lengths=geometry.lengths,
    )


def random_valid_permutation(mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, nodes].")
    permutations = []
    for sample_mask in mask.cpu():
        valid = int(sample_mask.sum().item())
        valid_order = torch.randperm(valid, generator=generator)
        padding_order = torch.arange(valid, sample_mask.shape[0])
        permutations.append(torch.cat((valid_order, padding_order), dim=0))
    return torch.stack(permutations, dim=0).to(mask.device)
