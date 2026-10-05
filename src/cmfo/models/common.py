from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cmfo.data.fields import GeometryBatch


@dataclass
class MultimodalOutput:
    class_logits: Tensor
    dense_logits: Tensor
    text_field: Tensor | None = None
    image_field: Tensor | None = None
    text_geometry: GeometryBatch | None = None
    image_geometry: GeometryBatch | None = None


def masked_weighted_mean(values: Tensor, geometry: GeometryBatch) -> Tensor:
    if values.shape[:2] != geometry.mask.shape:
        raise ValueError("values and geometry must share [batch, nodes].")
    weights = geometry.weights
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("values and mask must share [batch, nodes].")
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    numerator = torch.where(expanded_mask, values, torch.zeros_like(values)).sum(dim=1)
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    return numerator / denominator


class FourierCoordinateEncoder(nn.Module):
    def __init__(self, coordinate_dim: int, hidden_dim: int, num_bands: int) -> None:
        super().__init__()
        if coordinate_dim <= 0 or hidden_dim <= 0 or num_bands <= 0:
            raise ValueError("Coordinate encoder dimensions and bands must be positive.")
        self.coordinate_dim = coordinate_dim
        self.num_bands = num_bands
        frequencies = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=True)
        feature_dim = coordinate_dim * (1 + 2 * num_bands)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: Tensor) -> Tensor:
        if coords.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"Expected coordinate dimension {self.coordinate_dim}, got {coords.shape[-1]}."
            )
        angles = coords.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            (
                coords,
                torch.sin(angles).flatten(start_dim=-2),
                torch.cos(angles).flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return self.projection(encoded)


class IntegralAttention(nn.Module):
    """Measure-aware normalized integral attention with optional native-domain locality."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        relative_coordinate_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if relative_coordinate_dim is None:
            self.relative_bias = None
        else:
            self.relative_bias = nn.Sequential(
                nn.Linear(relative_coordinate_dim + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_heads),
            )

    def forward(
        self,
        query: Tensor,
        query_geometry: GeometryBatch,
        source: Tensor,
        source_geometry: GeometryBatch,
        radius: float | None = None,
    ) -> Tensor:
        if query.shape[:2] != query_geometry.mask.shape:
            raise ValueError("query and query_geometry must share [batch, nodes].")
        if source.shape[:2] != source_geometry.mask.shape:
            raise ValueError("source and source_geometry must share [batch, nodes].")
        batch_size, query_count, _ = query.shape
        source_count = source.shape[1]
        query_heads = self.q_projection(query).view(
            batch_size, query_count, self.num_heads, self.head_dim
        )
        key_heads = self.k_projection(source).view(
            batch_size, source_count, self.num_heads, self.head_dim
        )
        value_heads = self.v_projection(source).view(
            batch_size, source_count, self.num_heads, self.head_dim
        )
        scores = torch.einsum("bqhd,bshd->bhqs", query_heads, key_heads)
        scores = scores / math.sqrt(float(self.head_dim))

        pair_mask = source_geometry.mask[:, None, :].expand(batch_size, query_count, source_count)
        if self.relative_bias is not None:
            if query_geometry.coordinate_dim != source_geometry.coordinate_dim:
                raise ValueError("Relative bias requires matching native coordinate dimensions.")
            relative = query_geometry.coords.unsqueeze(2) - source_geometry.coords.unsqueeze(1)
            distance_squared = relative.square().sum(dim=-1, keepdim=True)
            bias_features = torch.cat((relative, distance_squared), dim=-1)
            bias = self.relative_bias(bias_features).permute(0, 3, 1, 2)
            scores = scores + bias
            if radius is not None:
                local_mask = distance_squared.squeeze(-1) <= radius**2
                pair_mask = pair_mask & local_mask

        log_measure = source_geometry.weights.clamp_min(1.0e-12).log()
        scores = scores + log_measure[:, None, None, :]
        scores = scores.masked_fill(~pair_mask[:, None, :, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(
            pair_mask[:, None, :, :],
            attention,
            torch.zeros_like(attention),
        )
        attention = self.dropout(attention)
        output = torch.einsum("bhqs,bshd->bqhd", attention, value_heads)
        output = output.reshape(batch_size, query_count, self.hidden_dim)
        output = self.output_projection(output)
        return output * query_geometry.mask.unsqueeze(-1).to(output.dtype)


class LowRankIntegralOperator(nn.Module):
    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.source_basis = nn.Linear(hidden_dim, rank)
        self.query_basis = nn.Linear(hidden_dim, rank)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        query: Tensor,
        query_geometry: GeometryBatch,
        source: Tensor,
        source_geometry: GeometryBatch,
    ) -> Tensor:
        source_basis = torch.tanh(self.source_basis(source))
        query_basis = torch.tanh(self.query_basis(query))
        values = self.value_projection(source)
        weighted_basis = source_basis * source_geometry.weights.unsqueeze(-1)
        summary = torch.einsum("bsr,bsh->brh", weighted_basis, values)
        output = torch.einsum("bqr,brh->bqh", query_basis, summary)
        output = self.output_projection(output / math.sqrt(float(self.rank)))
        return output * query_geometry.mask.unsqueeze(-1).to(output.dtype)


class PointwiseMixer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, values: Tensor, geometry: GeometryBatch) -> Tensor:
        output = self.network(values)
        return output * geometry.mask.unsqueeze(-1).to(output.dtype)
