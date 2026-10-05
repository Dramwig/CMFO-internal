from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cmfo.config import ExperimentConfig, TrainConfig
from cmfo.data.fields import MultimodalBatch
from cmfo.evaluation.evaluator import evaluate_loader
from cmfo.evaluation.metrics import classification_accuracy, masked_binary_iou
from cmfo.training.losses import multitask_loss
from cmfo.utils import atomic_write_json


def save_checkpoint_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        experiment_config: ExperimentConfig,
        device: torch.device,
        run_output_dir: str | Path,
        run_checkpoint_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.experiment_config = experiment_config
        self.config: TrainConfig = experiment_config.train
        self.device = device
        self.run_output_dir = Path(run_output_dir)
        self.run_checkpoint_dir = Path(run_checkpoint_dir)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.use_cuda_amp = self.config.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda_amp)
        self.grid_generator = torch.Generator().manual_seed(experiment_config.seed + 31337)

    def _train_epoch(
        self,
        loader: Iterable[MultimodalBatch],
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "classification_loss": 0.0,
            "dense_loss": 0.0,
            "consistency_loss": 0.0,
            "accuracy": 0.0,
            "miou": 0.0,
        }
        sample_count = 0
        for step, cpu_batch in enumerate(loader, start=1):
            batch = cpu_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=self.config.amp,
            ):
                output = self.model(batch, generator=self.grid_generator)
                second_output = None
                if self.config.consistency_weight > 0 and getattr(
                    self.model, "supports_quadrature", False
                ):
                    second_output = self.model(
                        batch,
                        layout="random",
                        generator=self.grid_generator,
                    )
                losses = multitask_loss(
                    output,
                    batch,
                    dense_weight=self.config.dense_loss_weight,
                    consistency_weight=self.config.consistency_weight,
                    second_output=second_output,
                )
            if self.use_cuda_amp:
                self.scaler.scale(losses.total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.optimizer.step()

            batch_size = batch.batch_size
            sample_count += batch_size
            totals["loss"] += float(losses.total.detach().item()) * batch_size
            totals["classification_loss"] += (
                float(losses.classification.detach().item()) * batch_size
            )
            totals["dense_loss"] += float(losses.dense.detach().item()) * batch_size
            totals["consistency_loss"] += float(losses.consistency.detach().item()) * batch_size
            totals["accuracy"] += (
                float(
                    classification_accuracy(output.class_logits.detach(), batch.class_labels).item()
                )
                * batch_size
            )
            totals["miou"] += (
                float(
                    masked_binary_iou(
                        output.dense_logits.detach(),
                        batch.queries.targets,
                        batch.queries.mask,
                    ).item()
                )
                * batch_size
            )
            if self.config.log_every > 0 and step % self.config.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={losses.total.detach().item():.4f}",
                    flush=True,
                )
        if sample_count == 0:
            raise ValueError("Training loader produced no samples.")
        return {key: value / sample_count for key, value in totals.items()}

    def fit(
        self,
        train_loader: Iterable[MultimodalBatch],
        validation_loader: Iterable[MultimodalBatch],
    ) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        best_validation_loss = float("inf")
        best_epoch = 0
        best_checkpoint = self.run_checkpoint_dir / "best.pt"
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)
            validation_metrics = evaluate_loader(
                self.model,
                validation_loader,
                self.device,
                self.experiment_config.model,
                calibration_bins=self.experiment_config.evaluation.calibration_bins,
            )
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(epoch_record)
            atomic_write_json(self.run_output_dir / "history.json", history)
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"validation_loss={validation_metrics['loss']:.4f}",
                flush=True,
            )
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                save_checkpoint_atomic(
                    best_checkpoint,
                    {
                        "schema_version": 1,
                        "experiment_config": self.experiment_config.to_dict(),
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "validation_metrics": validation_metrics,
                    },
                )
        checkpoint = torch.load(best_checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint": str(best_checkpoint),
        }
