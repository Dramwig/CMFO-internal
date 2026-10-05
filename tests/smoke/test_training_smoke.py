from pathlib import Path

import torch

from cmfo.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from cmfo.data.synthetic import ControlledSceneDataset, collate_controlled_scenes
from cmfo.evaluation.metrics import classification_accuracy, masked_binary_iou
from cmfo.models.cmfo import ContinuousMultimodalFieldOperator
from cmfo.training.experiment import run_experiment
from cmfo.training.losses import multitask_loss


def test_end_to_end_training_writes_checkpoint_and_report(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_name="pytest_smoke",
        seed=23,
        data=DataConfig(
            train_size=4,
            val_size=2,
            image_resolutions=(6,),
            image_layouts=("regular",),
            query_resolution=6,
        ),
        model=ModelConfig(
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            global_rank=2,
            coordinate_bands=2,
            text_nodes=6,
            image_nodes=9,
            local_radius=0.6,
            dropout=0.0,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            consistency_weight=0.01,
            device="cpu",
            log_every=0,
        ),
        evaluation=EvaluationConfig(
            quadrature_layouts=("regular", "random"),
            text_node_counts=(6,),
            image_node_counts=(9,),
            observation_resolutions=(6, 8),
            observation_layouts=("regular", "random"),
            query_resolutions=(6, 8),
            max_observation_samples=2,
            calibration_bins=4,
            latency_warmup=0,
            latency_repeats=1,
        ),
    )
    report = run_experiment(
        config,
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        run_id="run",
        device_override="cpu",
    )
    assert Path(report["training"]["best_checkpoint"]).is_file()
    assert (tmp_path / "outputs/run/report.json").is_file()
    assert report["validation"]["loss"] > 0
    assert report["quadrature_sweep"]
    assert len(report["observation_grid_sweep"]) == 4
    assert len(report["query_resolution_sweep"]) == 2


def test_single_batch_can_be_fully_overfit() -> None:
    torch.manual_seed(41)
    dataset = ControlledSceneDataset(
        1,
        image_resolutions=(6,),
        image_layouts=("regular",),
        query_resolution=6,
    )
    batch = collate_controlled_scenes([dataset[0]])
    model = ContinuousMultimodalFieldOperator(
        ModelConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            global_rank=4,
            coordinate_bands=3,
            text_nodes=8,
            image_nodes=16,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    initial_loss = None
    for step in range(101):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, layout="regular")
        loss = multitask_loss(output, batch, dense_weight=1.0).total
        if step == 0:
            initial_loss = float(loss.item())
        if step < 100:
            loss.backward()
            optimizer.step()
    assert initial_loss is not None
    assert float(loss.item()) < initial_loss * 0.15
    assert classification_accuracy(output.class_logits, batch.class_labels).item() == 1.0
    assert (
        masked_binary_iou(
            output.dense_logits,
            batch.queries.targets,
            batch.queries.mask,
        ).item()
        > 0.8
    )
