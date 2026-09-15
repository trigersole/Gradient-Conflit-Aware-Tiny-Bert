"""Measure task-vs-distillation gradient conflict during ordinary training.

The probe uses ``torch.autograd.grad`` and therefore does not write to
``parameter.grad``.  Call it after computing the individual losses and before
the usual backward pass of their weighted sum.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


SIGNALS = ("prediction", "hidden", "attention")


def shared_encoder_parameters(student: nn.Module) -> list[nn.Parameter]:
    """Return trainable encoder parameters for common Hugging Face models.

    Pass an explicit parameter iterable instead if the student uses a custom
    layout.  Selecting only the encoder deliberately excludes classifier and
    hidden-state projection heads, which are not shared by all four losses.
    """
    model = student.module if hasattr(student, "module") else student
    candidates = (
        getattr(model, "bert", None),
        getattr(model, "roberta", None),
        getattr(model, "distilbert", None),
        getattr(model, "albert", None),
        getattr(model, "deberta", None),
        getattr(model, "base_model", None),
    )
    encoder = next((candidate for candidate in candidates if candidate is not None), None)
    if encoder is None:
        raise ValueError(
            "Could not locate the student encoder. Pass your shared encoder "
            "parameters explicitly to GradientConflictProbe."
        )
    # Hugging Face BERT-like base models may include a pooler. The pooler feeds
    # classification/logit objectives but not intermediate-state objectives,
    # so including it would inflate only some gradient norms. Keep embeddings
    # and transformer blocks while excluding the pooler when possible.
    components = [
        getattr(encoder, name, None)
        for name in ("embeddings", "encoder", "transformer")
    ]
    components = [component for component in components if component is not None]
    source = components if components else [encoder]
    parameters = [
        parameter
        for component in source
        for parameter in component.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The selected student encoder has no trainable parameters.")
    return parameters


@dataclass
class _SignalStats:
    measured: int = 0
    negative: int = 0
    cosine_sum: float = 0.0
    negative_cosine_sum: float = 0.0

    def update(self, cosine: float) -> None:
        if not math.isfinite(cosine):
            return
        self.measured += 1
        self.cosine_sum += cosine
        if cosine < 0.0:
            self.negative += 1
            self.negative_cosine_sum += cosine

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "measured_batches": self.measured,
            "negative_batches": self.negative,
            "negative_percent": 100.0 * self.negative / self.measured if self.measured else None,
            "mean_cosine": self.cosine_sum / self.measured if self.measured else None,
            "mean_negative_cosine": (
                self.negative_cosine_sum / self.negative if self.negative else None
            ),
        }


@dataclass
class GradientConflictProbe:
    """Sample cos(g_task, g_signal) without changing the optimizer update.

    Args:
        parameters: Parameters shared by task, prediction, hidden and attention
            objectives. For TinyBERT this should normally be the student encoder.
        every_n_steps: Measure at 1-based steps divisible by this value.
        csv_path: Optional per-measurement CSV output path.
        gradient_mode: ``"raw"`` or ``"log_loss"``. Positive scalar loss
            normalization does not change cosine direction, but ``log_loss`` is
            useful if gradient norms are also being studied.
        epsilon: Numerical protection for norms and log-loss normalization.
    """

    parameters: Iterable[nn.Parameter]
    every_n_steps: int = 10
    csv_path: str | Path | None = None
    gradient_mode: str = "raw"
    epsilon: float = 1e-12
    _parameters: tuple[nn.Parameter, ...] = field(init=False, repr=False)
    _stats: dict[str, _SignalStats] = field(init=False, repr=False)
    _csv_file: Any = field(init=False, default=None, repr=False)
    _writer: csv.DictWriter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # Avoid double-counting if a caller accidentally supplies a shared
        # parameter through more than one module/parameter group.
        unique_parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in self.parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                unique_parameters.append(parameter)
                seen.add(id(parameter))
        self._parameters = tuple(unique_parameters)
        if not self._parameters:
            raise ValueError("parameters must contain at least one trainable parameter")
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if self.gradient_mode not in {"raw", "log_loss"}:
            raise ValueError("gradient_mode must be 'raw' or 'log_loss'")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._stats = {signal: _SignalStats() for signal in SIGNALS}

        if self.csv_path is not None:
            path = Path(self.csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = path.open("w", newline="", encoding="utf-8")
            fields = ["step", "epoch", "task_loss"]
            for signal in SIGNALS:
                fields.extend((f"{signal}_loss", f"cos_task_{signal}", f"norm_{signal}"))
            fields.append("norm_task")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._writer.writeheader()

    def should_measure(self, step: int) -> bool:
        """Return whether a 1-based optimizer/batch step should be sampled."""
        return step > 0 and step % self.every_n_steps == 0

    def _gradients(self, loss: Tensor) -> tuple[Tensor | None, ...]:
        if loss.ndim != 0:
            raise ValueError("Each loss must be a scalar tensor")
        gradients = torch.autograd.grad(
            loss,
            self._parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        if self.gradient_mode == "raw":
            return tuple(g.detach() if g is not None else None for g in gradients)

        # grad(log(L + eps)) = grad(L) / (L + eps).  abs/clamp keeps the
        # diagnostic finite if a custom objective unexpectedly becomes <= 0.
        denominator = loss.detach().abs().clamp_min(self.epsilon)
        return tuple(g.detach() / denominator if g is not None else None for g in gradients)

    @staticmethod
    def _dot_and_norms(
        left: tuple[Tensor | None, ...], right: tuple[Tensor | None, ...]
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = next(g.device for g in (*left, *right) if g is not None)
        # Float32 accumulation is stable for this diagnostic and is supported
        # consistently across CUDA, CPU and MPS devices.
        dot = torch.zeros((), device=device, dtype=torch.float32)
        left_sq = torch.zeros_like(dot)
        right_sq = torch.zeros_like(dot)
        for left_grad, right_grad in zip(left, right):
            if left_grad is not None:
                left_float = left_grad.float()
                left_sq += torch.sum(left_float * left_float)
            if right_grad is not None:
                right_float = right_grad.float()
                right_sq += torch.sum(right_float * right_float)
            if left_grad is not None and right_grad is not None:
                dot += torch.sum(left_grad.float() * right_grad.float())
        return dot, left_sq.sqrt(), right_sq.sqrt()

    def measure(
        self,
        losses: Mapping[str, Tensor],
        step: int,
        epoch: int | float | None = None,
    ) -> dict[str, float | int | None] | None:
        """Measure a batch, returning ``None`` on non-sampling steps.

        ``losses`` must contain ``task``, ``prediction``, ``hidden`` and
        ``attention`` scalar tensors from the same forward graph.
        """
        if not self.should_measure(step):
            return None
        required = {"task", *SIGNALS}
        missing = required.difference(losses)
        if missing:
            raise KeyError(f"Missing losses: {', '.join(sorted(missing))}")

        task_grads = self._gradients(losses["task"])
        row: dict[str, float | int | None] = {
            "step": step,
            "epoch": epoch,
            "task_loss": float(losses["task"].detach()),
        }
        task_norm_value: float | None = None

        for signal in SIGNALS:
            signal_grads = self._gradients(losses[signal])
            dot, task_norm, signal_norm = self._dot_and_norms(task_grads, signal_grads)
            task_norm_value = float(task_norm)
            denominator = task_norm * signal_norm
            cosine = float(dot / denominator) if float(denominator) > self.epsilon else math.nan
            # Bound harmless floating point overshoot to the cosine range.
            if math.isfinite(cosine):
                cosine = max(-1.0, min(1.0, cosine))
            self._stats[signal].update(cosine)
            row[f"{signal}_loss"] = float(losses[signal].detach())
            row[f"cos_task_{signal}"] = cosine
            row[f"norm_{signal}"] = float(signal_norm)

        row["norm_task"] = task_norm_value
        if self._writer is not None:
            self._writer.writerow(row)
            self._csv_file.flush()
        return row

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return aggregate conflict statistics for all sampled batches."""
        return {signal: stats.as_dict() for signal, stats in self._stats.items()}

    def format_summary(self) -> str:
        """Format the main result as a compact Markdown-style table."""
        lines = [
            "| Signal | Batches with negative cosine | Valid sampled batches |",
            "|---|---:|---:|",
        ]
        for signal, stats in self.summary().items():
            percent = stats["negative_percent"]
            display = "n/a" if percent is None else f"{percent:.1f}%"
            lines.append(
                f"| {signal.title()} | {display} | {stats['measured_batches']} |"
            )
        return "\n".join(lines)

    def write_summary(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)
            handle.write("\n")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self) -> "GradientConflictProbe":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
