from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cmfo.config import ExperimentConfig, load_experiment_config
from cmfo.evaluation.evaluator import (
    benchmark_latency,
    evaluate_loader,
    observation_grid_sweep,
    quadrature_sweep,
    query_resolution_sweep,
)
from cmfo.models.factory import build_model
from cmfo.training.experiment import make_dataloaders, run_experiment
from cmfo.utils import atomic_write_json, count_parameters, resolve_device, seed_everything


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device", default=None)


def _load_checkpoint_model(
    config: ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(config.model).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    return model


def command_train(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    report = run_experiment(
        config,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        run_id=args.run_id,
        device_override=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if config.data.train_size > 64 or config.data.val_size > 32 or config.train.epochs > 3:
        raise ValueError("Smoke configs must remain small: train<=64, val<=32, epochs<=3.")
    report = run_experiment(
        config,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        run_id=args.run_id,
        device_override=args.device or "cpu",
    )
    if not torch.isfinite(torch.tensor(report["validation"]["loss"])):
        raise RuntimeError("Smoke validation produced a non-finite loss.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    model = _load_checkpoint_model(config, args.checkpoint, device)
    _, validation_loader = make_dataloaders(config)
    metrics = evaluate_loader(
        model,
        validation_loader,
        device,
        config.model,
        calibration_bins=config.evaluation.calibration_bins,
    )
    first_batch = next(iter(validation_loader))
    sweep = quadrature_sweep(model, first_batch, device, config.evaluation)
    observation_sweep = observation_grid_sweep(
        model,
        config.data,
        config.model,
        config.evaluation,
        device,
        config.train.batch_size,
    )
    query_sweep = query_resolution_sweep(
        model,
        config.data,
        config.model,
        config.evaluation,
        device,
        config.train.batch_size,
    )
    efficiency = benchmark_latency(
        model,
        first_batch,
        device,
        config.evaluation.latency_warmup,
        config.evaluation.latency_repeats,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "metrics": metrics,
        "quadrature_sweep": sweep,
        "observation_grid_sweep": observation_sweep,
        "query_resolution_sweep": query_sweep,
        "efficiency": efficiency,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    model = build_model(config.model)
    suspicious = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if "position_embedding" in name or "node_embedding" in name or "patch_embedding" in name
    ]
    payload = {
        "model": config.model.name,
        "parameters": count_parameters(model),
        "supports_quadrature": bool(getattr(model, "supports_quadrature", False)),
        "suspicious_fixed_index_parameters": suspicious,
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cmfo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a configured experiment.")
    _add_common_run_arguments(train_parser)
    train_parser.set_defaults(handler=command_train)

    smoke_parser = subparsers.add_parser("smoke", help="Run a bounded end-to-end smoke experiment.")
    _add_common_run_arguments(smoke_parser)
    smoke_parser.set_defaults(handler=command_smoke)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.set_defaults(handler=command_evaluate)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect parameterization constraints.")
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.set_defaults(handler=command_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
