from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cmfo.config import ExperimentConfig
from cmfo.data.synthetic import ControlledSceneDataset, collate_controlled_scenes
from cmfo.evaluation.evaluator import (
    benchmark_latency,
    evaluate_loader,
    observation_grid_sweep,
    quadrature_sweep,
    query_resolution_sweep,
)
from cmfo.models.factory import build_model
from cmfo.training.trainer import Trainer
from cmfo.utils import (
    atomic_write_json,
    count_parameters,
    resolve_device,
    seed_everything,
    worker_seed,
)


def make_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ControlledSceneDataset(
        size=config.data.train_size,
        image_resolutions=config.data.image_resolutions,
        image_layouts=config.data.image_layouts,
        query_resolution=config.data.query_resolution,
        base_seed=config.data.base_seed,
        index_offset=0,
    )
    validation_dataset = ControlledSceneDataset(
        size=config.data.val_size,
        image_resolutions=config.data.image_resolutions,
        image_layouts=config.data.image_layouts,
        query_resolution=config.data.query_resolution,
        base_seed=config.data.base_seed,
        index_offset=1_000_000,
    )
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "collate_fn": collate_controlled_scenes,
        "worker_init_fn": worker_seed,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def default_run_id(experiment_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}_{timestamp}"


def _prepare_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_experiment(
    config: ExperimentConfig,
    output_root: str | Path,
    checkpoint_root: str | Path,
    run_id: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(device_override or config.train.device)
    selected_run_id = run_id or default_run_id(config.experiment_name)
    run_output_dir = Path(output_root) / selected_run_id
    run_checkpoint_dir = Path(checkpoint_root) / selected_run_id
    _prepare_new_directory(run_output_dir)
    _prepare_new_directory(run_checkpoint_dir)
    atomic_write_json(run_output_dir / "resolved_config.json", config.to_dict())

    train_loader, validation_loader = make_dataloaders(config)
    model = build_model(config.model)
    trainer = Trainer(
        model=model,
        experiment_config=config,
        device=device,
        run_output_dir=run_output_dir,
        run_checkpoint_dir=run_checkpoint_dir,
    )
    training = trainer.fit(train_loader, validation_loader)
    validation = evaluate_loader(
        trainer.model,
        validation_loader,
        device,
        config.model,
        calibration_bins=config.evaluation.calibration_bins,
    )
    first_batch = next(iter(validation_loader))
    sweep = quadrature_sweep(
        trainer.model,
        first_batch,
        device,
        config.evaluation,
    )
    observation_sweep = observation_grid_sweep(
        trainer.model,
        config.data,
        config.model,
        config.evaluation,
        device,
        config.train.batch_size,
    )
    query_sweep = query_resolution_sweep(
        trainer.model,
        config.data,
        config.model,
        config.evaluation,
        device,
        config.train.batch_size,
    )
    efficiency = benchmark_latency(
        trainer.model,
        first_batch,
        device,
        config.evaluation.latency_warmup,
        config.evaluation.latency_repeats,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": selected_run_id,
        "experiment_name": config.experiment_name,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "parameters": count_parameters(trainer.model),
        "training": training,
        "validation": validation,
        "quadrature_sweep": sweep,
        "observation_grid_sweep": observation_sweep,
        "query_resolution_sweep": query_sweep,
        "efficiency": efficiency,
        "output_dir": str(run_output_dir),
        "checkpoint_dir": str(run_checkpoint_dir),
    }
    atomic_write_json(run_output_dir / "report.json", report)
    return report
