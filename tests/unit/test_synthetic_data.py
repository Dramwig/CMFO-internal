import torch

from cmfo.data.synthetic import (
    ControlledSceneDataset,
    collate_controlled_scenes,
    generate_scene,
)


def test_scene_generation_is_deterministic() -> None:
    assert generate_scene(11, 1307) == generate_scene(11, 1307)
    assert generate_scene(11, 1307) != generate_scene(12, 1307)


def test_same_latent_scene_survives_observation_discretization_change() -> None:
    regular = ControlledSceneDataset(
        1,
        image_resolutions=(8,),
        image_layouts=("regular",),
        query_resolution=10,
        base_seed=1307,
    )[0]
    random_points = ControlledSceneDataset(
        1,
        image_resolutions=(11,),
        image_layouts=("random",),
        query_resolution=10,
        base_seed=1307,
    )[0]
    assert regular["prompt"] == random_points["prompt"]
    assert regular["class_label"].item() == random_points["class_label"].item()
    torch.testing.assert_close(regular["query_targets"], random_points["query_targets"])
    assert regular["image_values"].shape[0] == 64
    assert random_points["image_values"].shape[0] == 121


def test_collate_supports_variable_image_and_text_counts() -> None:
    first = ControlledSceneDataset(
        1, image_resolutions=(8,), image_layouts=("regular",), query_resolution=8
    )[0]
    second = ControlledSceneDataset(
        1,
        image_resolutions=(10,),
        image_layouts=("jittered",),
        query_resolution=8,
        index_offset=7,
    )[0]
    batch = collate_controlled_scenes([first, second])
    assert batch.batch_size == 2
    assert batch.image.lengths.tolist() == [64, 100]
    assert batch.text.lengths[0] != 0 and batch.text.lengths[1] != 0
    assert batch.queries.coords.shape == (2, 64, 2)
