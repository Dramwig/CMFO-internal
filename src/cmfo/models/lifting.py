from __future__ import annotations

from torch import Tensor, nn

from cmfo.data.fields import FieldBatch, GeometryBatch
from cmfo.models.common import FourierCoordinateEncoder, IntegralAttention


class ByteFieldLifter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        coordinate_bands: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_encoder = nn.Embedding(256, hidden_dim)
        self.coordinate_encoder = FourierCoordinateEncoder(1, hidden_dim, coordinate_bands)
        self.integral = IntegralAttention(
            hidden_dim,
            num_heads,
            relative_coordinate_dim=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, observations: FieldBatch, queries: GeometryBatch) -> Tensor:
        if observations.values.ndim != 2:
            raise ValueError("Byte observations must have shape [batch, bytes].")
        source = self.value_encoder(observations.values.long())
        source = source + self.coordinate_encoder(observations.coords)
        query_seed = self.coordinate_encoder(queries.coords)
        lifted = self.integral(
            query_seed,
            queries,
            source,
            observations.geometry,
        )
        output = self.norm(query_seed + lifted)
        return output * queries.mask.unsqueeze(-1).to(output.dtype)


class ImageFieldLifter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        coordinate_bands: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_encoder = nn.Linear(3, hidden_dim)
        self.coordinate_encoder = FourierCoordinateEncoder(2, hidden_dim, coordinate_bands)
        self.integral = IntegralAttention(
            hidden_dim,
            num_heads,
            relative_coordinate_dim=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, observations: FieldBatch, queries: GeometryBatch) -> Tensor:
        if observations.values.ndim != 3 or observations.values.shape[-1] != 3:
            raise ValueError("Image observations must have shape [batch, pixels, 3].")
        source = self.value_encoder(observations.values.float())
        source = source + self.coordinate_encoder(observations.coords)
        query_seed = self.coordinate_encoder(queries.coords)
        lifted = self.integral(
            query_seed,
            queries,
            source,
            observations.geometry,
        )
        output = self.norm(query_seed + lifted)
        return output * queries.mask.unsqueeze(-1).to(output.dtype)
