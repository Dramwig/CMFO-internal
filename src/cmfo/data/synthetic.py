from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from cmfo.data.fields import MultimodalBatch, pack_fields, pack_queries
from cmfo.data.quadrature import make_1d_coordinates, make_2d_coordinates

COLOR_VALUES: dict[str, tuple[float, float, float]] = {
    "red": (0.90, 0.15, 0.12),
    "green": (0.15, 0.80, 0.25),
    "blue": (0.15, 0.30, 0.92),
    "yellow": (0.92, 0.78, 0.12),
}
SHAPES = ("circle", "square")
RELATIONS = ("left", "right", "above", "below")


@dataclass(frozen=True)
class SceneObject:
    color: str
    shape: str
    center_x: float
    center_y: float
    size: float

    @property
    def description(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(frozen=True)
class SceneSpec:
    sample_id: int
    target: SceneObject
    reference: SceneObject
    relation_label: int

    @property
    def prompt(self) -> str:
        return (
            f"where is the {self.target.description} relative to the "
            f"{self.reference.description}? segment the {self.target.description}."
        )


def _relation_label(target: SceneObject, reference: SceneObject) -> int:
    delta_x = target.center_x - reference.center_x
    delta_y = target.center_y - reference.center_y
    if abs(delta_x) >= abs(delta_y):
        return 0 if delta_x < 0 else 1
    return 2 if delta_y < 0 else 3


def generate_scene(sample_id: int, base_seed: int = 1307) -> SceneSpec:
    rng = random.Random(base_seed + sample_id * 1009)
    combinations = list(product(COLOR_VALUES, SHAPES))
    target_identity, reference_identity = rng.sample(combinations, 2)
    target_center = (rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82))
    reference_center = target_center
    for _ in range(64):
        candidate = (rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82))
        distance_squared = sum(
            (left - right) ** 2 for left, right in zip(target_center, candidate, strict=True)
        )
        if distance_squared >= 0.16:
            reference_center = candidate
            break
    if reference_center == target_center:
        reference_center = (
            0.82 if target_center[0] < 0.5 else 0.18,
            0.82 if target_center[1] < 0.5 else 0.18,
        )
    target = SceneObject(
        color=target_identity[0],
        shape=target_identity[1],
        center_x=target_center[0],
        center_y=target_center[1],
        size=rng.uniform(0.10, 0.15),
    )
    reference = SceneObject(
        color=reference_identity[0],
        shape=reference_identity[1],
        center_x=reference_center[0],
        center_y=reference_center[1],
        size=rng.uniform(0.10, 0.15),
    )
    return SceneSpec(
        sample_id=sample_id,
        target=target,
        reference=reference,
        relation_label=_relation_label(target, reference),
    )


def object_mask(coords: Tensor, scene_object: SceneObject) -> Tensor:
    delta_x = coords[:, 0] - scene_object.center_x
    delta_y = coords[:, 1] - scene_object.center_y
    if scene_object.shape == "circle":
        return delta_x.square() + delta_y.square() <= scene_object.size**2
    if scene_object.shape == "square":
        return torch.maximum(delta_x.abs(), delta_y.abs()) <= scene_object.size
    raise ValueError(f"Unsupported shape: {scene_object.shape!r}.")


def render_scene(spec: SceneSpec, coords: Tensor) -> Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [points, 2].")
    x = coords[:, 0]
    y = coords[:, 1]
    image = torch.stack(
        (
            0.035 + 0.035 * x,
            0.035 + 0.025 * y,
            0.045 + 0.020 * (1.0 - x),
        ),
        dim=-1,
    )
    for scene_object in (spec.reference, spec.target):
        mask = object_mask(coords, scene_object)
        color = torch.tensor(COLOR_VALUES[scene_object.color], dtype=image.dtype)
        image = torch.where(mask.unsqueeze(-1), color.unsqueeze(0), image)
    return image


def encode_text_observations(text: str) -> tuple[Tensor, Tensor, Tensor]:
    encoded = text.encode("utf-8")
    if not encoded:
        raise ValueError("Text observations cannot be empty.")
    values = torch.tensor(list(encoded), dtype=torch.long)
    coords = make_1d_coordinates(len(encoded), "regular")
    weights = torch.full((len(encoded),), 1.0 / float(len(encoded)), dtype=torch.float32)
    return values, coords, weights


class ControlledSceneDataset(Dataset[dict[str, Any]]):
    """Deterministic E1 scenes with arbitrary observation and query discretizations."""

    def __init__(
        self,
        size: int,
        image_resolutions: tuple[int, ...] = (16,),
        image_layouts: tuple[str, ...] = ("regular",),
        query_resolution: int = 16,
        base_seed: int = 1307,
        index_offset: int = 0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive.")
        if not image_resolutions or not image_layouts:
            raise ValueError("At least one resolution and layout are required.")
        self.size = int(size)
        self.image_resolutions = tuple(int(value) for value in image_resolutions)
        self.image_layouts = tuple(str(value) for value in image_layouts)
        self.query_resolution = int(query_resolution)
        self.base_seed = int(base_seed)
        self.index_offset = int(index_offset)

    def __len__(self) -> int:
        return self.size

    def _variant(self, sample_id: int) -> tuple[int, str]:
        rng = random.Random(self.base_seed + sample_id * 7919 + 17)
        return rng.choice(self.image_resolutions), rng.choice(self.image_layouts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        sample_id = self.index_offset + index
        spec = generate_scene(sample_id, self.base_seed)
        resolution, layout = self._variant(sample_id)
        generator = torch.Generator().manual_seed(
            self.base_seed + sample_id * 104729 + resolution * 101 + len(layout)
        )
        image_coords = make_2d_coordinates(resolution * resolution, layout, generator)
        image_values = render_scene(spec, image_coords)
        image_weights = torch.full(
            (image_coords.shape[0],),
            1.0 / float(image_coords.shape[0]),
            dtype=torch.float32,
        )
        query_coords = make_2d_coordinates(self.query_resolution**2, "regular")
        query_targets = object_mask(query_coords, spec.target).float()
        query_weights = torch.full(
            (query_coords.shape[0],),
            1.0 / float(query_coords.shape[0]),
            dtype=torch.float32,
        )
        text_values, text_coords, text_weights = encode_text_observations(spec.prompt)
        return {
            "sample_id": sample_id,
            "prompt": spec.prompt,
            "relation": RELATIONS[spec.relation_label],
            "text_values": text_values,
            "text_coords": text_coords,
            "text_weights": text_weights,
            "image_values": image_values,
            "image_coords": image_coords,
            "image_weights": image_weights,
            "query_coords": query_coords,
            "query_targets": query_targets,
            "query_weights": query_weights,
            "class_label": torch.tensor(spec.relation_label, dtype=torch.long),
            "image_resolution": resolution,
            "image_layout": layout,
        }


def collate_controlled_scenes(samples: list[dict[str, Any]]) -> MultimodalBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    text = pack_fields(
        [sample["text_values"] for sample in samples],
        [sample["text_coords"] for sample in samples],
        [sample["text_weights"] for sample in samples],
        pad_value=0,
    )
    image = pack_fields(
        [sample["image_values"] for sample in samples],
        [sample["image_coords"] for sample in samples],
        [sample["image_weights"] for sample in samples],
        pad_value=0.0,
    )
    queries = pack_queries(
        [sample["query_coords"] for sample in samples],
        [sample["query_targets"] for sample in samples],
        [sample["query_weights"] for sample in samples],
    )
    class_labels = torch.stack([sample["class_label"] for sample in samples], dim=0)
    sample_ids = tuple(int(sample["sample_id"]) for sample in samples)
    return MultimodalBatch(text, image, queries, class_labels, sample_ids)
