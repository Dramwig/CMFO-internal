from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from cmfo.config import DataConfig, EvaluationConfig, ModelConfig
from cmfo.data.fields import MultimodalBatch
from cmfo.data.quadrature import (
    interpolate_field,
    normalized_field_l2,
)
from cmfo.data.synthetic import ControlledSceneDataset, collate_controlled_scenes
from cmfo.evaluation.metrics import (
    classification_accuracy,
    expected_calibration_error,
    masked_binary_iou,
    mean_squared_consistency,
)
from cmfo.models.common import MultimodalOutput
from cmfo.training.losses import multitask_loss
from cmfo.utils import count_parameters


def _model_forward(
    model: nn.Module,
    batch: MultimodalBatch,
    *,
    layout: str | None = None,
    text_nodes: int | None = None,
    image_nodes: int | None = None,
    generator: torch.Generator | None = None,
) -> MultimodalOutput:
    if getattr(model, "supports_quadrature", False):
        return model(
            batch,
            layout=layout,
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            generator=generator,
        )
    return model(batch)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable[MultimodalBatch],
    device: torch.device,
    model_config: ModelConfig,
    calibration_bins: int = 10,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "dense_loss": 0.0,
        "accuracy": 0.0,
        "miou": 0.0,
    }
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    generator = torch.Generator().manual_seed(424242)
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        output = _model_forward(
            model,
            batch,
            layout="regular" if getattr(model, "supports_quadrature", False) else None,
            text_nodes=model_config.text_nodes,
            image_nodes=model_config.image_nodes,
            generator=generator,
        )
        loss = multitask_loss(output, batch, dense_weight=1.0)
        batch_size = batch.batch_size
        total_samples += batch_size
        totals["loss"] += float(loss.total.item()) * batch_size
        totals["classification_loss"] += float(loss.classification.item()) * batch_size
        totals["dense_loss"] += float(loss.dense.item()) * batch_size
        totals["accuracy"] += (
            float(classification_accuracy(output.class_logits, batch.class_labels).item())
            * batch_size
        )
        totals["miou"] += (
            float(
                masked_binary_iou(
                    output.dense_logits, batch.queries.targets, batch.queries.mask
                ).item()
            )
            * batch_size
        )
        all_logits.append(output.class_logits.detach().cpu())
        all_labels.append(batch.class_labels.detach().cpu())
    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples.")
    metrics = {key: value / total_samples for key, value in totals.items()}
    metrics["ece"] = float(
        expected_calibration_error(
            torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0),
            bins=calibration_bins,
        ).item()
    )
    return metrics


@torch.no_grad()
def quadrature_sweep(
    model: nn.Module,
    batch: MultimodalBatch,
    device: torch.device,
    evaluation: EvaluationConfig,
) -> list[dict[str, Any]]:
    if not getattr(model, "supports_quadrature", False):
        return []
    model.eval()
    batch = batch.to(device)
    reference_text_nodes = max(evaluation.text_node_counts)
    reference_image_nodes = max(evaluation.image_node_counts)
    reference_generator = torch.Generator().manual_seed(9001)
    reference = _model_forward(
        model,
        batch,
        layout="regular",
        text_nodes=reference_text_nodes,
        image_nodes=reference_image_nodes,
        generator=reference_generator,
    )
    if reference.text_geometry is None or reference.image_geometry is None:
        raise RuntimeError("CMFO output did not expose field geometries.")
    records: list[dict[str, Any]] = []
    for layout in evaluation.quadrature_layouts:
        for text_nodes in evaluation.text_node_counts:
            for image_nodes in evaluation.image_node_counts:
                if layout != "random":
                    side = int(image_nodes**0.5)
                    if side * side != image_nodes:
                        continue
                generator = torch.Generator().manual_seed(
                    100_000 + text_nodes * 101 + image_nodes * 17 + len(layout)
                )
                output = _model_forward(
                    model,
                    batch,
                    layout=layout,
                    text_nodes=text_nodes,
                    image_nodes=image_nodes,
                    generator=generator,
                )
                record: dict[str, Any] = {
                    "layout": layout,
                    "text_nodes": text_nodes,
                    "image_nodes": image_nodes,
                    "accuracy": float(
                        classification_accuracy(output.class_logits, batch.class_labels).item()
                    ),
                    "miou": float(
                        masked_binary_iou(
                            output.dense_logits,
                            batch.queries.targets,
                            batch.queries.mask,
                        ).item()
                    ),
                    "class_probability_mse": float(
                        mean_squared_consistency(
                            torch.softmax(output.class_logits, dim=-1),
                            torch.softmax(reference.class_logits, dim=-1),
                        ).item()
                    ),
                    "dense_probability_mse": float(
                        mean_squared_consistency(
                            torch.sigmoid(output.dense_logits),
                            torch.sigmoid(reference.dense_logits),
                            batch.queries.mask,
                        ).item()
                    ),
                }
                if (
                    output.text_field is not None
                    and output.image_field is not None
                    and output.text_geometry is not None
                    and output.image_geometry is not None
                    and reference.text_field is not None
                    and reference.image_field is not None
                ):
                    text_on_reference = interpolate_field(
                        output.text_field,
                        output.text_geometry,
                        reference.text_geometry,
                    )
                    image_on_reference = interpolate_field(
                        output.image_field,
                        output.image_geometry,
                        reference.image_geometry,
                    )
                    record["text_field_normalized_l2"] = float(
                        normalized_field_l2(
                            text_on_reference,
                            reference.text_field,
                            reference.text_geometry.mask,
                        )
                        .mean()
                        .item()
                    )
                    record["image_field_normalized_l2"] = float(
                        normalized_field_l2(
                            image_on_reference,
                            reference.image_field,
                            reference.image_geometry.mask,
                        )
                        .mean()
                        .item()
                    )
                records.append(record)
    return records


def _controlled_validation_loader(
    data_config: DataConfig,
    *,
    size: int,
    resolution: int,
    layout: str,
    query_resolution: int,
    batch_size: int,
) -> DataLoader:
    dataset = ControlledSceneDataset(
        size=size,
        image_resolutions=(resolution,),
        image_layouts=(layout,),
        query_resolution=query_resolution,
        base_seed=data_config.base_seed,
        index_offset=1_000_000,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_controlled_scenes,
    )


@torch.no_grad()
def paired_output_consistency(
    model: nn.Module,
    reference_loader: Iterable[MultimodalBatch],
    variant_loader: Iterable[MultimodalBatch],
    device: torch.device,
    model_config: ModelConfig,
) -> dict[str, float]:
    model.eval()
    class_total = 0.0
    dense_total = 0.0
    sample_count = 0
    reference_iterator = iter(reference_loader)
    variant_iterator = iter(variant_loader)
    while True:
        try:
            reference_cpu = next(reference_iterator)
        except StopIteration:
            try:
                next(variant_iterator)
            except StopIteration:
                break
            raise ValueError("Variant loader contains more batches than the reference loader.")
        try:
            variant_cpu = next(variant_iterator)
        except StopIteration as error:
            raise ValueError("Variant loader ended before the reference loader.") from error
        if reference_cpu.sample_ids != variant_cpu.sample_ids:
            raise ValueError("Paired consistency requires aligned latent scene ids.")
        if reference_cpu.queries.coords.shape != variant_cpu.queries.coords.shape:
            raise ValueError("Paired consistency requires a shared output query grid.")
        reference = reference_cpu.to(device)
        variant = variant_cpu.to(device)
        forward_kwargs = {
            "layout": "regular" if getattr(model, "supports_quadrature", False) else None,
            "text_nodes": model_config.text_nodes,
            "image_nodes": model_config.image_nodes,
        }
        reference_output = _model_forward(model, reference, **forward_kwargs)
        variant_output = _model_forward(model, variant, **forward_kwargs)
        batch_size = reference.batch_size
        sample_count += batch_size
        class_total += (
            float(
                mean_squared_consistency(
                    torch.softmax(variant_output.class_logits, dim=-1),
                    torch.softmax(reference_output.class_logits, dim=-1),
                ).item()
            )
            * batch_size
        )
        dense_total += (
            float(
                mean_squared_consistency(
                    torch.sigmoid(variant_output.dense_logits),
                    torch.sigmoid(reference_output.dense_logits),
                    reference.queries.mask,
                ).item()
            )
            * batch_size
        )
    if sample_count == 0:
        raise ValueError("Paired consistency loaders produced no samples.")
    return {
        "class_probability_mse": class_total / sample_count,
        "dense_probability_mse": dense_total / sample_count,
    }


@torch.no_grad()
def observation_grid_sweep(
    model: nn.Module,
    data_config: DataConfig,
    model_config: ModelConfig,
    evaluation: EvaluationConfig,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    size = min(data_config.val_size, evaluation.max_observation_samples)
    reference_resolution = max(evaluation.observation_resolutions)
    query_resolution = data_config.query_resolution
    reference_loader = _controlled_validation_loader(
        data_config,
        size=size,
        resolution=reference_resolution,
        layout="regular",
        query_resolution=query_resolution,
        batch_size=batch_size,
    )
    records: list[dict[str, Any]] = []
    for layout in evaluation.observation_layouts:
        for resolution in evaluation.observation_resolutions:
            variant_loader = _controlled_validation_loader(
                data_config,
                size=size,
                resolution=resolution,
                layout=layout,
                query_resolution=query_resolution,
                batch_size=batch_size,
            )
            task_metrics = evaluate_loader(
                model,
                variant_loader,
                device,
                model_config,
                calibration_bins=evaluation.calibration_bins,
            )
            consistency = paired_output_consistency(
                model,
                reference_loader,
                variant_loader,
                device,
                model_config,
            )
            records.append(
                {
                    "image_resolution": resolution,
                    "observation_layout": layout,
                    "reference_resolution": reference_resolution,
                    **task_metrics,
                    **consistency,
                }
            )
    return records


@torch.no_grad()
def query_resolution_sweep(
    model: nn.Module,
    data_config: DataConfig,
    model_config: ModelConfig,
    evaluation: EvaluationConfig,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    size = min(data_config.val_size, evaluation.max_observation_samples)
    observation_resolution = max(data_config.image_resolutions)
    records: list[dict[str, Any]] = []
    for query_resolution in evaluation.query_resolutions:
        loader = _controlled_validation_loader(
            data_config,
            size=size,
            resolution=observation_resolution,
            layout="regular",
            query_resolution=query_resolution,
            batch_size=batch_size,
        )
        task_metrics = evaluate_loader(
            model,
            loader,
            device,
            model_config,
            calibration_bins=evaluation.calibration_bins,
        )
        records.append(
            {
                "query_resolution": query_resolution,
                "observation_resolution": observation_resolution,
                **task_metrics,
            }
        )
    return records


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    batch: MultimodalBatch,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    model.eval()
    batch = batch.to(device)
    for _ in range(warmup):
        _model_forward(model, batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _model_forward(model, batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "parameters": count_parameters(model),
        "latency_seconds_mean": elapsed / max(repeats, 1),
        "samples_per_second": batch.batch_size * max(repeats, 1) / max(elapsed, 1.0e-12),
    }
