from pathlib import Path

import pytest

from cmfo.config import ExperimentConfig, ModelConfig, load_experiment_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_config_loads() -> None:
    config = load_experiment_config(REPOSITORY_ROOT / "configs/smoke/e1_cpu.yaml")
    assert config.model.name == "cmfo"
    assert config.data.train_size == 8


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("e1_b0_patch_transformer.yaml", "b0_patch_transformer"),
        ("e1_b1_point_transformer.yaml", "b1_point_transformer"),
        ("e1_b2_perceiver.yaml", "b2_perceiver"),
        ("e1_b3_fixed_cmfo.yaml", "b3_fixed_cmfo"),
        ("e1_b4_augmented_transformer.yaml", "b4_augmented_transformer"),
    ],
)
def test_inherited_baseline_configs(filename: str, expected_name: str) -> None:
    config = load_experiment_config(REPOSITORY_ROOT / "configs/ablation" / filename)
    assert config.model.name == expected_name
    assert config.model.hidden_dim == 64


def test_invalid_model_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_dim=10, num_heads=4)


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_mapping({"unexpected": True})
