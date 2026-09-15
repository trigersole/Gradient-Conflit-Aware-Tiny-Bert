"""Fixed-weight TinyBERT distillation with gradient-conflict measurement.

The default experiment uses an SST-2-fine-tuned compact BERT teacher and a
pretrained four-layer TinyBERT student. The teacher is never updated.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from gradient_conflict import GradientConflictProbe, shared_encoder_parameters


DEFAULT_TEACHER = "yoshitomo-matsubara/bert-base-uncased-sst2"
DEFAULT_STUDENT = "huawei-noah/TinyBERT_General_4L_312D"


@dataclass(frozen=True)
class LayerPair:
    student_hidden: int
    teacher_hidden: int
    student_attention: int
    teacher_attention: int


def make_layer_pairs(student_layers: int, teacher_layers: int) -> list[LayerPair]:
    """Evenly map every student layer to a deeper/equal teacher layer.

    Hidden-state tuples include embeddings at index zero; attention tuples do
    not. For the default 4-to-8 setup this maps teacher layers 2, 4, 6 and 8.
    """
    if student_layers <= 0 or teacher_layers < student_layers:
        raise ValueError("teacher must have at least as many layers as the student")
    pairs: list[LayerPair] = []
    previous = 0
    for student_layer in range(1, student_layers + 1):
        teacher_layer = round(student_layer * teacher_layers / student_layers)
        teacher_layer = max(previous + 1, min(teacher_layer, teacher_layers))
        previous = teacher_layer
        pairs.append(
            LayerPair(
                student_hidden=student_layer,
                teacher_hidden=teacher_layer,
                student_attention=student_layer - 1,
                teacher_attention=teacher_layer - 1,
            )
        )
    return pairs


def prediction_distillation_loss(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2


def hidden_distillation_loss(
    student_hidden_states: tuple[Tensor, ...],
    teacher_hidden_states: tuple[Tensor, ...],
    projections: nn.ModuleList,
    layer_pairs: list[LayerPair],
    attention_mask: Tensor,
) -> Tensor:
    token_mask = attention_mask.unsqueeze(-1).to(dtype=student_hidden_states[0].dtype)
    losses = []
    for projection, pair in zip(projections, layer_pairs):
        student_state = projection(student_hidden_states[pair.student_hidden])
        teacher_state = teacher_hidden_states[pair.teacher_hidden]
        squared_error = (student_state - teacher_state).square() * token_mask
        denominator = token_mask.sum().clamp_min(1.0) * teacher_state.shape[-1]
        losses.append(squared_error.sum() / denominator)
    return torch.stack(losses).mean()


def attention_distillation_loss(
    student_attentions: tuple[Tensor, ...],
    teacher_attentions: tuple[Tensor, ...],
    layer_pairs: list[LayerPair],
    attention_mask: Tensor,
) -> Tensor:
    token_mask = attention_mask.to(dtype=student_attentions[0].dtype)
    pair_mask = token_mask[:, None, :, None] * token_mask[:, None, None, :]
    losses = []
    for pair in layer_pairs:
        student_map = student_attentions[pair.student_attention]
        teacher_map = teacher_attentions[pair.teacher_attention]
        if student_map.shape[1] != teacher_map.shape[1]:
            # Non-default teachers with a different number of heads remain
            # usable through head-averaged maps. The canonical 12-head teacher
            # and 12-head TinyBERT student take the direct head-wise path.
            student_map = student_map.mean(dim=1, keepdim=True)
            teacher_map = teacher_map.mean(dim=1, keepdim=True)
        squared_error = (student_map - teacher_map).square() * pair_mask
        denominator = pair_mask.sum().clamp_min(1.0) * student_map.shape[1]
        losses.append(squared_error.sum() / denominator)
    return torch.stack(losses).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_sst2(tokenizer, max_length: int, max_train: int | None, max_eval: int | None):
    dataset = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True, desc="Tokenizing SST-2")
    dataset = dataset.rename_column("label", "labels")
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove = [column for column in dataset["train"].column_names if column not in keep]
    dataset = dataset.remove_columns(remove)
    train = dataset["train"]
    validation = dataset["validation"]
    if max_train is not None:
        train = train.select(range(min(max_train, len(train))))
    if max_eval is not None:
        validation = validation.select(range(min(max_eval, len(validation))))
    return train, validation


def make_loader(
    dataset: Dataset,
    collator: DataCollatorWithPadding,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator if shuffle else None,
    )


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        output = model(**batch)
        loss_sum += float(F.cross_entropy(output.logits, labels, reduction="sum"))
        correct += int((output.logits.argmax(dim=-1) == labels).sum())
        total += labels.numel()
    return {"loss": loss_sum / total, "accuracy": correct / total, "examples": total}


def optimizer_for(model: nn.Module, projections: nn.Module, learning_rate: float, weight_decay: float):
    named_parameters = list(model.named_parameters()) + [
        (f"projections.{name}", parameter) for name, parameter in projections.named_parameters()
    ]
    decay, no_decay = [], []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        target = no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay
        target.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use a GPU node or pass --allow-cpu for a tiny smoke test.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    train_dataset, validation_dataset = tokenize_sst2(
        tokenizer, args.max_length, args.max_train_samples, args.max_eval_samples
    )
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    train_loader = make_loader(
        train_dataset, collator, args.batch_size, True, args.num_workers, args.seed
    )
    validation_loader = make_loader(
        validation_dataset, collator, args.eval_batch_size, False, args.num_workers, args.seed
    )

    teacher_config = AutoConfig.from_pretrained(args.teacher_model)
    student_config = AutoConfig.from_pretrained(args.student_model, num_labels=2)
    if teacher_config.num_labels != 2:
        raise ValueError(f"Teacher must have 2 SST-2 labels, found {teacher_config.num_labels}")
    if teacher_config.num_hidden_layers < student_config.num_hidden_layers:
        raise ValueError("Teacher must have at least as many transformer layers as the student")
    if teacher_config.vocab_size != student_config.vocab_size:
        raise ValueError(
            "Teacher and student vocabularies differ. Use a tokenizer/checkpoint pair "
            "with the same vocabulary before distillation."
        )

    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        config=student_config,
        ignore_mismatched_sizes=True,
    )
    teacher.requires_grad_(False).eval().to(device)
    student.to(device)

    layer_pairs = make_layer_pairs(
        student_config.num_hidden_layers, teacher_config.num_hidden_layers
    )
    projections = nn.ModuleList(
        nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
        for _ in layer_pairs
    ).to(device)

    teacher_metrics = evaluate(teacher, validation_loader, device)
    print(f"Teacher validation accuracy: {100 * teacher_metrics['accuracy']:.2f}%", flush=True)
    if teacher_metrics["accuracy"] < args.minimum_teacher_accuracy:
        raise RuntimeError(
            f"Teacher accuracy {teacher_metrics['accuracy']:.3f} is below the required "
            f"{args.minimum_teacher_accuracy:.3f}; check checkpoint labels/tokenization."
        )

    optimizer = optimizer_for(student, projections, args.learning_rate, args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    probe = GradientConflictProbe(
        shared_encoder_parameters(student),
        every_n_steps=args.measure_every,
        csv_path=output_dir / "gradient_cosines.csv",
        gradient_mode=args.gradient_mode,
    )

    weights = {
        "task": args.task_weight,
        "prediction": args.prediction_weight,
        "hidden": args.hidden_weight,
        "attention": args.attention_weight,
    }
    global_step = 0
    started = time.time()
    try:
        for epoch in range(args.epochs):
            student.train()
            projections.train()
            for batch_index, batch in enumerate(train_loader, start=1):
                global_step += 1
                batch = move_batch(batch, device)
                labels = batch.pop("labels")
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    teacher_output = teacher(
                        **batch, output_hidden_states=True, output_attentions=True
                    )
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                    student_output = student(
                        **batch, output_hidden_states=True, output_attentions=True
                    )
                    losses = {
                        "task": F.cross_entropy(student_output.logits, labels),
                        "prediction": prediction_distillation_loss(
                            student_output.logits, teacher_output.logits, args.temperature
                        ),
                        "hidden": hidden_distillation_loss(
                            student_output.hidden_states,
                            teacher_output.hidden_states,
                            projections,
                            layer_pairs,
                            batch["attention_mask"],
                        ),
                        "attention": attention_distillation_loss(
                            student_output.attentions,
                            teacher_output.attentions,
                            layer_pairs,
                            batch["attention_mask"],
                        ),
                    }
                    total_loss = sum(weights[name] * loss for name, loss in losses.items())

                measurement = probe.measure(losses, step=global_step, epoch=epoch + 1)
                if measurement is not None:
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={float(total_loss.detach()):.4f} "
                        f"cos(T,P)={measurement['cos_task_prediction']:+.3f} "
                        f"cos(T,H)={measurement['cos_task_hidden']:+.3f} "
                        f"cos(T,A)={measurement['cos_task_attention']:+.3f}",
                        flush=True,
                    )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projections.parameters()), args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if args.max_steps is not None and global_step >= args.max_steps:
                    break
            metrics = evaluate(student, validation_loader, device)
            print(
                f"Epoch {epoch + 1}: student validation accuracy="
                f"{100 * metrics['accuracy']:.2f}%",
                flush=True,
            )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
    finally:
        probe.close()

    final_metrics = evaluate(student, validation_loader, device)
    conflict_summary = probe.summary()
    probe.write_summary(output_dir / "gradient_conflict_summary.json")
    student.save_pretrained(output_dir / "student")
    tokenizer.save_pretrained(output_dir / "student")
    torch.save(projections.state_dict(), output_dir / "hidden_projections.pt")

    results: dict[str, object] = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "teacher_validation": teacher_metrics,
        "student_validation": final_metrics,
        "layer_pairs": [asdict(pair) for pair in layer_pairs],
        "loss_weights": weights,
        "gradient_conflict": conflict_summary,
        "completed_steps": global_step,
        "runtime_seconds": time.time() - started,
        "seed": args.seed,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
        handle.write("\n")

    print("\n" + probe.format_summary(), flush=True)
    print(f"Student validation accuracy: {100 * final_metrics['accuracy']:.2f}%", flush=True)
    print(f"Results written to {output_dir.resolve()}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT)
    parser.add_argument("--output-dir", default="runs/tinybert-sst2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--gradient-mode", choices=("raw", "log_loss"), default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--minimum-teacher-accuracy", type=float, default=0.80)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 256 train examples, 128 validation examples and at most 10 steps.",
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 256
        args.max_eval_samples = args.max_eval_samples or 128
        args.max_steps = args.max_steps or 10
        args.measure_every = 1
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("epochs and batch sizes must be positive")
    if args.measure_every <= 0 or args.max_length <= 0:
        parser.error("measure-every and max-length must be positive")
    if args.temperature <= 0 or args.learning_rate <= 0:
        parser.error("temperature and learning-rate must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("warmup-ratio must be between 0 and 1")
    if any(
        weight < 0
        for weight in (
            args.task_weight,
            args.prediction_weight,
            args.hidden_weight,
            args.attention_weight,
        )
    ):
        parser.error("loss weights must be nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("max-train-samples must be positive")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        parser.error("max-eval-samples must be positive")
    if args.fp16 and args.allow_cpu and not torch.cuda.is_available():
        parser.error("--fp16 requires CUDA")
    return args


if __name__ == "__main__":
    train(parse_args())
