import torch

from cmfo.config import ModelConfig
from cmfo.data.fields import MultimodalBatch, random_valid_permutation
from cmfo.data.quadrature import make_geometry_batch
from cmfo.data.synthetic import ControlledSceneDataset, collate_controlled_scenes
from cmfo.models.cmfo import ContinuousMultimodalFieldOperator
from cmfo.training.losses import multitask_loss


def tiny_batch() -> MultimodalBatch:
    dataset = ControlledSceneDataset(
        2,
        image_resolutions=(8,),
        image_layouts=("regular",),
        query_resolution=8,
    )
    return collate_controlled_scenes([dataset[0], dataset[1]])


def tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        global_rank=4,
        coordinate_bands=3,
        local_radius=0.5,
        text_nodes=8,
        image_nodes=16,
        dropout=0.0,
    )


def test_cmfo_forward_backward_and_shapes() -> None:
    torch.manual_seed(3)
    model = ContinuousMultimodalFieldOperator(tiny_config())
    batch = tiny_batch()
    output = model(batch, layout="regular")
    assert output.class_logits.shape == (2, 4)
    assert output.dense_logits.shape == (2, 64)
    assert output.text_field is not None and output.text_field.shape == (2, 8, 16)
    assert output.image_field is not None and output.image_field.shape == (2, 16, 16)
    losses = multitask_loss(output, batch, dense_weight=1.0)
    losses.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_observation_and_hidden_node_permutation_invariance() -> None:
    torch.manual_seed(5)
    model = ContinuousMultimodalFieldOperator(tiny_config()).eval()
    batch = tiny_batch()
    text_geometry = make_geometry_batch(2, 8, 1, "regular")
    image_geometry = make_geometry_batch(2, 16, 2, "regular")
    with torch.no_grad():
        reference = model(
            batch,
            text_geometry=text_geometry,
            image_geometry=image_geometry,
        )
        text_permutation = random_valid_permutation(
            batch.text.mask, torch.Generator().manual_seed(11)
        )
        image_permutation = random_valid_permutation(
            batch.image.mask, torch.Generator().manual_seed(12)
        )
        permuted_batch = MultimodalBatch(
            text=batch.text.permuted(text_permutation),
            image=batch.image.permuted(image_permutation),
            queries=batch.queries,
            class_labels=batch.class_labels,
            sample_ids=batch.sample_ids,
        )
        observation_permuted = model(
            permuted_batch,
            text_geometry=text_geometry,
            image_geometry=image_geometry,
        )
        hidden_text_permutation = random_valid_permutation(
            text_geometry.mask, torch.Generator().manual_seed(13)
        )
        hidden_image_permutation = random_valid_permutation(
            image_geometry.mask, torch.Generator().manual_seed(14)
        )
        hidden_permuted = model(
            batch,
            text_geometry=text_geometry.permuted(hidden_text_permutation),
            image_geometry=image_geometry.permuted(hidden_image_permutation),
        )
    torch.testing.assert_close(
        reference.class_logits, observation_permuted.class_logits, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        reference.dense_logits, observation_permuted.dense_logits, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        reference.class_logits, hidden_permuted.class_logits, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        reference.dense_logits, hidden_permuted.dense_logits, atol=1e-5, rtol=1e-5
    )


def test_unseen_node_counts_and_layouts_do_not_change_parameter_shapes() -> None:
    model = ContinuousMultimodalFieldOperator(tiny_config()).eval()
    batch = tiny_batch()
    state_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with torch.no_grad():
        for layout, text_nodes, image_nodes in (
            ("regular", 7, 9),
            ("jittered", 11, 25),
            ("random", 13, 20),
        ):
            output = model(
                batch,
                layout=layout,
                text_nodes=text_nodes,
                image_nodes=image_nodes,
                generator=torch.Generator().manual_seed(19),
            )
            assert torch.isfinite(output.class_logits).all()
            assert torch.isfinite(output.dense_logits).all()
    assert state_shapes == {name: tuple(value.shape) for name, value in model.state_dict().items()}
    forbidden = [
        name
        for name, _ in model.named_parameters()
        if "node_embedding" in name or "position_embedding" in name
    ]
    assert forbidden == []


def test_variable_hidden_node_counts_within_batch() -> None:
    model = ContinuousMultimodalFieldOperator(tiny_config()).eval()
    batch = tiny_batch()
    text_geometry = make_geometry_batch(2, (7, 11), 1, "random")
    image_geometry = make_geometry_batch(2, (9, 16), 2, "random")
    with torch.no_grad():
        output = model(
            batch,
            text_geometry=text_geometry,
            image_geometry=image_geometry,
        )
    assert output.text_geometry is not None
    assert output.image_geometry is not None
    assert output.text_geometry.lengths.tolist() == [7, 11]
    assert output.image_geometry.lengths.tolist() == [9, 16]
    assert torch.isfinite(output.class_logits).all()


def test_cpu_bfloat16_autocast_is_finite() -> None:
    model = ContinuousMultimodalFieldOperator(tiny_config()).eval()
    batch = tiny_batch()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch, layout="regular")
    assert torch.isfinite(output.class_logits.float()).all()
    assert torch.isfinite(output.dense_logits.float()).all()
