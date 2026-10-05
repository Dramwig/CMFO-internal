from __future__ import annotations

import math

from torch import Tensor, nn

from cmfo.data.fields import GeometryBatch
from cmfo.models.common import IntegralAttention, LowRankIntegralOperator, PointwiseMixer


class CoupledFieldOperatorBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        global_rank: int,
        local_radius: float,
        dropout: float,
        step_size: float,
        enable_local: bool = True,
        enable_global: bool = True,
        enable_cross: bool = True,
    ) -> None:
        super().__init__()
        self.local_radius = float(local_radius)
        self.step_size = float(step_size)
        self.enable_local = bool(enable_local)
        self.enable_global = bool(enable_global)
        self.enable_cross = bool(enable_cross)

        self.text_pointwise = PointwiseMixer(hidden_dim, dropout)
        self.image_pointwise = PointwiseMixer(hidden_dim, dropout)
        self.text_local = IntegralAttention(
            hidden_dim, num_heads, relative_coordinate_dim=1, dropout=dropout
        )
        self.image_local = IntegralAttention(
            hidden_dim, num_heads, relative_coordinate_dim=2, dropout=dropout
        )
        self.text_global = LowRankIntegralOperator(hidden_dim, global_rank)
        self.image_global = LowRankIntegralOperator(hidden_dim, global_rank)
        self.image_to_text = IntegralAttention(hidden_dim, num_heads, dropout=dropout)
        self.text_to_image = IntegralAttention(hidden_dim, num_heads, dropout=dropout)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text: Tensor,
        text_geometry: GeometryBatch,
        image: Tensor,
        image_geometry: GeometryBatch,
    ) -> tuple[Tensor, Tensor]:
        text_branches = [self.text_pointwise(text, text_geometry)]
        image_branches = [self.image_pointwise(image, image_geometry)]
        if self.enable_local:
            text_branches.append(
                self.text_local(
                    text,
                    text_geometry,
                    text,
                    text_geometry,
                    radius=self.local_radius,
                )
            )
            image_branches.append(
                self.image_local(
                    image,
                    image_geometry,
                    image,
                    image_geometry,
                    radius=self.local_radius,
                )
            )
        if self.enable_global:
            text_branches.append(self.text_global(text, text_geometry, text, text_geometry))
            image_branches.append(self.image_global(image, image_geometry, image, image_geometry))
        if self.enable_cross:
            text_branches.append(self.image_to_text(text, text_geometry, image, image_geometry))
            image_branches.append(self.text_to_image(image, image_geometry, text, text_geometry))
        text_update = sum(text_branches) / math.sqrt(float(len(text_branches)))
        image_update = sum(image_branches) / math.sqrt(float(len(image_branches)))
        text_output = self.text_norm(text + self.step_size * self.dropout(text_update))
        image_output = self.image_norm(image + self.step_size * self.dropout(image_update))
        text_output = text_output * text_geometry.mask.unsqueeze(-1).to(text_output.dtype)
        image_output = image_output * image_geometry.mask.unsqueeze(-1).to(image_output.dtype)
        return text_output, image_output
