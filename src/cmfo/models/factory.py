from __future__ import annotations

from dataclasses import replace

from torch import nn

from cmfo.config import ModelConfig
from cmfo.models.baselines import (
    AugmentedPointTransformerBaseline,
    PatchTransformerBaseline,
    PerceiverBaseline,
    PointTransformerBaseline,
)
from cmfo.models.cmfo import ContinuousMultimodalFieldOperator


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "cmfo":
        return ContinuousMultimodalFieldOperator(config)
    if config.name == "b0_patch_transformer":
        return PatchTransformerBaseline(config)
    if config.name == "b1_point_transformer":
        return PointTransformerBaseline(config)
    if config.name == "b2_perceiver":
        return PerceiverBaseline(config)
    if config.name == "b3_fixed_cmfo":
        return ContinuousMultimodalFieldOperator(replace(config, node_layout="regular"))
    if config.name == "b4_augmented_transformer":
        return AugmentedPointTransformerBaseline(config)
    raise ValueError(f"Unsupported model name: {config.name!r}.")
