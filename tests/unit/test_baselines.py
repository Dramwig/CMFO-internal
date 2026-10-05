import pytest
import torch

from cmfo.config import ModelConfig
from cmfo.data.synthetic import ControlledSceneDataset, collate_controlled_scenes
from cmfo.models.factory import build_model
from cmfo.training.losses import multitask_loss


@pytest.mark.parametrize(
    "name",
    [
        "b0_patch_transformer",
        "b1_point_transformer",
        "b2_perceiver",
        "b3_fixed_cmfo",
        "b4_augmented_transformer",
    ],
)
def test_all_declared_baselines_forward_and_backward(name: str) -> None:
    config = ModelConfig(
        name=name,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        global_rank=4,
        coordinate_bands=3,
        text_nodes=8,
        image_nodes=16,
        patch_grid=2,
        perceiver_latents=8,
        dropout=0.0,
    )
    dataset = ControlledSceneDataset(
        2, image_resolutions=(8,), image_layouts=("regular",), query_resolution=8
    )
    batch = collate_controlled_scenes([dataset[0], dataset[1]])
    model = build_model(config)
    output = model(batch)
    assert output.class_logits.shape == (2, 4)
    assert output.dense_logits.shape == (2, 64)
    assert torch.isfinite(output.class_logits).all()
    assert torch.isfinite(output.dense_logits).all()
    loss = multitask_loss(output, batch, dense_weight=1.0).total
    loss.backward()
    assert torch.isfinite(loss)
