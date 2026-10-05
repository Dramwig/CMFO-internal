from __future__ import annotations

import torch
from torch import nn

from cmfo.config import ModelConfig
from cmfo.data.fields import GeometryBatch, MultimodalBatch
from cmfo.data.quadrature import make_geometry_batch
from cmfo.models.common import (
    FourierCoordinateEncoder,
    IntegralAttention,
    MultimodalOutput,
    masked_weighted_mean,
)
from cmfo.models.lifting import ByteFieldLifter, ImageFieldLifter
from cmfo.models.operators import CoupledFieldOperatorBlock


class ContinuousMultimodalFieldOperator(nn.Module):
    supports_quadrature = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.text_lifter = ByteFieldLifter(
            hidden_dim,
            config.num_heads,
            config.coordinate_bands,
            config.dropout,
        )
        self.image_lifter = ImageFieldLifter(
            hidden_dim,
            config.num_heads,
            config.coordinate_bands,
            config.dropout,
        )
        step_size = 1.0 / float(config.num_layers)
        self.blocks = nn.ModuleList(
            [
                CoupledFieldOperatorBlock(
                    hidden_dim=hidden_dim,
                    num_heads=config.num_heads,
                    global_rank=config.global_rank,
                    local_radius=config.local_radius,
                    dropout=config.dropout,
                    step_size=step_size,
                    enable_local=config.enable_local,
                    enable_global=config.enable_global,
                    enable_cross=config.enable_cross,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.num_classes),
        )
        self.query_encoder = FourierCoordinateEncoder(2, hidden_dim, config.coordinate_bands)
        self.image_reader = IntegralAttention(
            hidden_dim,
            config.num_heads,
            relative_coordinate_dim=2,
            dropout=config.dropout,
        )
        self.text_reader = IntegralAttention(
            hidden_dim,
            config.num_heads,
            dropout=config.dropout,
        )
        self.dense_norm = nn.LayerNorm(hidden_dim)
        self.dense_head = nn.Linear(hidden_dim, 1)

    def make_hidden_geometries(
        self,
        batch_size: int,
        device: torch.device,
        layout: str | None = None,
        text_nodes: int | None = None,
        image_nodes: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[GeometryBatch, GeometryBatch]:
        selected_layout = layout or self.config.node_layout
        text_geometry = make_geometry_batch(
            batch_size=batch_size,
            counts=text_nodes or self.config.text_nodes,
            coordinate_dim=1,
            layout=selected_layout,
            generator=generator,
            device=device,
        )
        image_geometry = make_geometry_batch(
            batch_size=batch_size,
            counts=image_nodes or self.config.image_nodes,
            coordinate_dim=2,
            layout=selected_layout,
            generator=generator,
            device=device,
        )
        return text_geometry, image_geometry

    def forward(
        self,
        batch: MultimodalBatch,
        *,
        text_geometry: GeometryBatch | None = None,
        image_geometry: GeometryBatch | None = None,
        layout: str | None = None,
        text_nodes: int | None = None,
        image_nodes: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MultimodalOutput:
        device = batch.class_labels.device
        if (text_geometry is None) != (image_geometry is None):
            raise ValueError("text_geometry and image_geometry must be supplied together.")
        if text_geometry is None or image_geometry is None:
            text_geometry, image_geometry = self.make_hidden_geometries(
                batch.batch_size,
                device,
                layout,
                text_nodes,
                image_nodes,
                generator,
            )
        else:
            text_geometry = text_geometry.to(device)
            image_geometry = image_geometry.to(device)
        text = self.text_lifter(batch.text, text_geometry)
        image = self.image_lifter(batch.image, image_geometry)
        for block in self.blocks:
            text, image = block(text, text_geometry, image, image_geometry)

        pooled_text = masked_weighted_mean(text, text_geometry)
        pooled_image = masked_weighted_mean(image, image_geometry)
        class_logits = self.classifier(torch.cat((pooled_text, pooled_image), dim=-1))

        query_geometry = batch.queries.geometry
        query_seed = self.query_encoder(query_geometry.coords)
        image_context = self.image_reader(
            query_seed,
            query_geometry,
            image,
            image_geometry,
        )
        text_context = self.text_reader(
            query_seed,
            query_geometry,
            text,
            text_geometry,
        )
        dense_features = self.dense_norm(query_seed + image_context + text_context)
        dense_logits = self.dense_head(dense_features).squeeze(-1)
        dense_logits = torch.where(
            query_geometry.mask,
            dense_logits,
            torch.zeros_like(dense_logits),
        )
        return MultimodalOutput(
            class_logits=class_logits,
            dense_logits=dense_logits,
            text_field=text,
            image_field=image,
            text_geometry=text_geometry,
            image_geometry=image_geometry,
        )
