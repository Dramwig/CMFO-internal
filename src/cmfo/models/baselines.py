from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cmfo.config import ModelConfig
from cmfo.data.fields import GeometryBatch, MultimodalBatch
from cmfo.data.quadrature import make_2d_coordinates
from cmfo.models.common import (
    FourierCoordinateEncoder,
    MultimodalOutput,
    masked_mean,
)


class ObservationTokenEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.byte_embedding = nn.Embedding(256, hidden_dim)
        self.text_coordinate_encoder = FourierCoordinateEncoder(
            1, hidden_dim, config.coordinate_bands
        )
        self.image_value_encoder = nn.Linear(3, hidden_dim)
        self.image_coordinate_encoder = FourierCoordinateEncoder(
            2, hidden_dim, config.coordinate_bands
        )
        self.text_modality = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.image_modality = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def encode_text(self, batch: MultimodalBatch) -> Tensor:
        return (
            self.byte_embedding(batch.text.values.long())
            + self.text_coordinate_encoder(batch.text.coords)
            + self.text_modality
        )

    def encode_image_values(self, values: Tensor, geometry: GeometryBatch) -> Tensor:
        return (
            self.image_value_encoder(values.float())
            + self.image_coordinate_encoder(geometry.coords)
            + self.image_modality
        )


class MultimodalTransformerBaseline(nn.Module):
    supports_quadrature = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = ObservationTokenEncoder(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
        self.query_encoder = FourierCoordinateEncoder(2, config.hidden_dim, config.coordinate_bands)
        self.query_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dense_norm = nn.LayerNorm(config.hidden_dim)
        self.dense_head = nn.Linear(config.hidden_dim, 1)

    def image_tokens(self, batch: MultimodalBatch) -> tuple[Tensor, GeometryBatch]:
        values = batch.image.values.float()
        return self.token_encoder.encode_image_values(
            values, batch.image.geometry
        ), batch.image.geometry

    def forward(self, batch: MultimodalBatch, **_: object) -> MultimodalOutput:
        text_tokens = self.token_encoder.encode_text(batch)
        image_tokens, image_geometry = self.image_tokens(batch)
        tokens = torch.cat((text_tokens, image_tokens), dim=1)
        mask = torch.cat((batch.text.mask, image_geometry.mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        pooled = masked_mean(encoded, mask)
        class_logits = self.classifier(pooled)
        query_features = self.query_encoder(batch.queries.coords)
        dense_context, _ = self.query_attention(
            query_features,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        dense_features = self.dense_norm(query_features + dense_context)
        dense_logits = self.dense_head(dense_features).squeeze(-1)
        dense_logits = torch.where(
            batch.queries.mask,
            dense_logits,
            torch.zeros_like(dense_logits),
        )
        text_count = text_tokens.shape[1]
        return MultimodalOutput(
            class_logits=class_logits,
            dense_logits=dense_logits,
            text_field=encoded[:, :text_count],
            image_field=encoded[:, text_count:],
            text_geometry=batch.text.geometry,
            image_geometry=image_geometry,
        )


class PatchTransformerBaseline(MultimodalTransformerBaseline):
    """B0: fixed image patch grid plus byte tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.patch_grid <= 1:
            raise ValueError("patch_grid must be greater than one.")

    def image_tokens(self, batch: MultimodalBatch) -> tuple[Tensor, GeometryBatch]:
        batch_size = batch.batch_size
        patch_count = self.config.patch_grid**2
        centers = make_2d_coordinates(patch_count, "regular").to(batch.image.coords.device)
        coords = centers.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        mask = torch.ones(batch_size, patch_count, dtype=torch.bool, device=coords.device)
        weights = torch.full(
            (batch_size, patch_count),
            1.0 / float(patch_count),
            dtype=coords.dtype,
            device=coords.device,
        )
        lengths = torch.full((batch_size,), patch_count, dtype=torch.long, device=coords.device)
        patch_geometry = GeometryBatch(coords, weights, mask, lengths)
        distance_squared = (
            (coords.unsqueeze(2) - batch.image.coords.unsqueeze(1)).square().sum(dim=-1)
        )
        sigma = 0.6 / float(self.config.patch_grid)
        scores = -distance_squared / (2.0 * sigma**2)
        scores = scores + batch.image.weights.clamp_min(1.0e-12).log().unsqueeze(1)
        scores = scores.masked_fill(
            ~batch.image.mask.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        assignment = torch.softmax(scores, dim=-1)
        patch_values = torch.einsum("bpn,bnc->bpc", assignment, batch.image.values.float())
        return (
            self.token_encoder.encode_image_values(patch_values, patch_geometry),
            patch_geometry,
        )


class PointTransformerBaseline(MultimodalTransformerBaseline):
    """B1: raw image observation points with ordinary self-attention."""


class AugmentedPointTransformerBaseline(PointTransformerBaseline):
    """B4: B1 architecture; the matched resolution augmentation is controlled by data config."""


class PerceiverBaseline(nn.Module):
    """B2: fixed learned latent array with cross-attention."""

    supports_quadrature = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = ObservationTokenEncoder(config)
        self.latents = nn.Parameter(
            torch.randn(config.perceiver_latents, config.hidden_dim)
            / math.sqrt(float(config.hidden_dim))
        )
        self.input_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.latent_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
        self.query_encoder = FourierCoordinateEncoder(2, config.hidden_dim, config.coordinate_bands)
        self.output_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dense_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: MultimodalBatch, **_: object) -> MultimodalOutput:
        text_tokens = self.token_encoder.encode_text(batch)
        image_tokens = self.token_encoder.encode_image_values(
            batch.image.values.float(), batch.image.geometry
        )
        tokens = torch.cat((text_tokens, image_tokens), dim=1)
        mask = torch.cat((batch.text.mask, batch.image.mask), dim=1)
        latents = self.latents.unsqueeze(0).expand(batch.batch_size, -1, -1)
        latents, _ = self.input_attention(
            latents,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        latents = self.latent_encoder(latents)
        class_logits = self.classifier(latents.mean(dim=1))
        queries = self.query_encoder(batch.queries.coords)
        dense_features, _ = self.output_attention(
            queries,
            latents,
            latents,
            need_weights=False,
        )
        dense_logits = self.dense_head(queries + dense_features).squeeze(-1)
        dense_logits = torch.where(
            batch.queries.mask,
            dense_logits,
            torch.zeros_like(dense_logits),
        )
        return MultimodalOutput(class_logits=class_logits, dense_logits=dense_logits)
