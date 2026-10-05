import torch

from cmfo.config import DataConfig, EvaluationConfig, ModelConfig
from cmfo.evaluation.evaluator import observation_grid_sweep, query_resolution_sweep
from cmfo.models.factory import build_model


def test_observation_and_query_sweeps_cover_declared_axes() -> None:
    data_config = DataConfig(
        train_size=2,
        val_size=2,
        image_resolutions=(6,),
        image_layouts=("regular",),
        query_resolution=6,
    )
    model_config = ModelConfig(
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        global_rank=2,
        coordinate_bands=2,
        text_nodes=6,
        image_nodes=9,
        dropout=0.0,
    )
    evaluation = EvaluationConfig(
        quadrature_layouts=("regular",),
        text_node_counts=(6,),
        image_node_counts=(9,),
        observation_resolutions=(6, 8),
        observation_layouts=("regular", "random"),
        query_resolutions=(6, 8),
        max_observation_samples=2,
        calibration_bins=4,
        latency_warmup=0,
        latency_repeats=1,
    )
    model = build_model(model_config)
    observation_records = observation_grid_sweep(
        model,
        data_config,
        model_config,
        evaluation,
        torch.device("cpu"),
        batch_size=2,
    )
    query_records = query_resolution_sweep(
        model,
        data_config,
        model_config,
        evaluation,
        torch.device("cpu"),
        batch_size=2,
    )
    assert len(observation_records) == 4
    assert len(query_records) == 2
    assert {
        (record["image_resolution"], record["observation_layout"]) for record in observation_records
    } == {(6, "regular"), (8, "regular"), (6, "random"), (8, "random")}
    assert {record["query_resolution"] for record in query_records} == {6, 8}
    assert all(record["class_probability_mse"] >= 0 for record in observation_records)
