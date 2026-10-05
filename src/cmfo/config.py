from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_size: int = 256
    val_size: int = 64
    image_resolutions: tuple[int, ...] = (16, 24)
    image_layouts: tuple[str, ...] = ("regular", "jittered")
    query_resolution: int = 16
    base_seed: int = 1307
    num_objects: int = 2

    def __post_init__(self) -> None:
        self.image_resolutions = tuple(int(value) for value in self.image_resolutions)
        self.image_layouts = tuple(str(value) for value in self.image_layouts)
        if self.train_size <= 0 or self.val_size <= 0:
            raise ValueError("Dataset sizes must be positive.")
        if not self.image_resolutions or any(value <= 1 for value in self.image_resolutions):
            raise ValueError("image_resolutions must contain values greater than one.")
        if not self.image_layouts:
            raise ValueError("At least one image layout is required.")
        if self.query_resolution <= 1:
            raise ValueError("query_resolution must be greater than one.")
        if self.num_objects != 2:
            raise ValueError("The current controlled task requires exactly two objects.")


@dataclass
class ModelConfig:
    name: str = "cmfo"
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    global_rank: int = 8
    coordinate_bands: int = 6
    local_radius: float = 0.35
    dropout: float = 0.0
    text_nodes: int = 32
    image_nodes: int = 64
    node_layout: str = "random"
    num_classes: int = 4
    patch_grid: int = 4
    perceiver_latents: int = 16
    enable_local: bool = True
    enable_global: bool = True
    enable_cross: bool = True

    def __post_init__(self) -> None:
        supported = {
            "cmfo",
            "b0_patch_transformer",
            "b1_point_transformer",
            "b2_perceiver",
            "b3_fixed_cmfo",
            "b4_augmented_transformer",
        }
        if self.name not in supported:
            raise ValueError(f"Unsupported model name: {self.name!r}.")
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads.")
        if self.num_layers <= 0 or self.global_rank <= 0:
            raise ValueError("num_layers and global_rank must be positive.")
        if self.text_nodes <= 0 or self.image_nodes <= 0:
            raise ValueError("Quadrature node counts must be positive.")
        if self.node_layout not in {"regular", "jittered", "random"}:
            raise ValueError(f"Unsupported node layout: {self.node_layout!r}.")


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    dense_loss_weight: float = 1.0
    consistency_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.consistency_weight < 0 or self.dense_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative.")


@dataclass
class EvaluationConfig:
    quadrature_layouts: tuple[str, ...] = ("regular", "jittered", "random")
    text_node_counts: tuple[int, ...] = (16, 32, 48)
    image_node_counts: tuple[int, ...] = (16, 36, 64)
    observation_resolutions: tuple[int, ...] = (16, 24, 32, 40)
    observation_layouts: tuple[str, ...] = ("regular", "jittered", "random")
    query_resolutions: tuple[int, ...] = (16, 24, 32)
    max_observation_samples: int = 64
    calibration_bins: int = 10
    latency_warmup: int = 1
    latency_repeats: int = 3

    def __post_init__(self) -> None:
        self.quadrature_layouts = tuple(str(value) for value in self.quadrature_layouts)
        self.text_node_counts = tuple(int(value) for value in self.text_node_counts)
        self.image_node_counts = tuple(int(value) for value in self.image_node_counts)
        self.observation_resolutions = tuple(int(value) for value in self.observation_resolutions)
        self.observation_layouts = tuple(str(value) for value in self.observation_layouts)
        self.query_resolutions = tuple(int(value) for value in self.query_resolutions)
        if any(value <= 0 for value in self.text_node_counts + self.image_node_counts):
            raise ValueError("Evaluation node counts must be positive.")
        if any(value <= 1 for value in self.observation_resolutions + self.query_resolutions):
            raise ValueError("Observation and query resolutions must be greater than one.")
        if self.max_observation_samples <= 0:
            raise ValueError("max_observation_samples must be positive.")
        if self.calibration_bins <= 1:
            raise ValueError("calibration_bins must be greater than one.")


@dataclass
class ExperimentConfig:
    experiment_name: str = "cmfo_e1"
    seed: int = 17
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        allowed = {"experiment_name", "seed", "data", "model", "train", "evaluation"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            experiment_name=str(raw.get("experiment_name", "cmfo_e1")),
            seed=int(raw.get("seed", 17)),
            data=DataConfig(**dict(raw.get("data", {}))),
            model=ModelConfig(**dict(raw.get("model", {}))),
            train=TrainConfig(**dict(raw.get("train", {}))),
            evaluation=EvaluationConfig(**dict(raw.get("evaluation", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()

    def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load_mapping(current: Path, seen: set[Path]) -> dict[str, Any]:
        current = current.resolve()
        if current in seen:
            raise ValueError(f"Cyclic config inheritance detected at {current}.")
        with current.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected a mapping in {current}, got {type(raw).__name__}.")
        raw = dict(raw)
        parent = raw.pop("extends", None)
        if parent is None:
            return raw
        parent_path = (current.parent / str(parent)).resolve()
        return deep_merge(load_mapping(parent_path, seen | {current}), raw)

    return ExperimentConfig.from_mapping(load_mapping(config_path, set()))
