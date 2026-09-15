# TinyBERT gradient-conflict diagnostic

This workspace includes a drop-in diagnostic for the first experiment: train a
normal fixed-weight TinyBERT baseline while measuring
`cos(g_task, g_prediction)`, `cos(g_task, g_hidden)`, and
`cos(g_task, g_attention)` every tenth batch.

The implementation is in [`gradient_conflict.py`](gradient_conflict.py). It
uses `torch.autograd.grad`, so it does **not** populate or alter `.grad`; the
ordinary weighted-sum training update remains unchanged.

## Add it to the training loop

Create the probe once. Select only the shared student encoder parameters, not
the classifier or the student-to-teacher hidden projection layers:

```python
from gradient_conflict import GradientConflictProbe, shared_encoder_parameters

probe = GradientConflictProbe(
    parameters=shared_encoder_parameters(student),
    every_n_steps=10,
    csv_path="runs/baseline_seed_42/gradient_cosines.csv",
    gradient_mode="raw",
)
```

Then retain each component loss and call the probe **before** the normal
backward pass. `global_step` is 1-based here.

```python
for batch_index, batch in enumerate(train_loader):
    global_step += 1
    optimizer.zero_grad(set_to_none=True)

    # Existing TinyBERT forward/loss code:
    student_output = student(**batch, output_hidden_states=True,
                             output_attentions=True)
    with torch.no_grad():
        teacher_output = teacher(**batch, output_hidden_states=True,
                                 output_attentions=True)

    task_loss = compute_task_loss(student_output.logits, batch["labels"])
    prediction_loss = compute_prediction_loss(
        student_output.logits, teacher_output.logits, temperature
    )
    hidden_loss = compute_hidden_loss(
        student_output.hidden_states, teacher_output.hidden_states
    )
    attention_loss = compute_attention_loss(
        student_output.attentions, teacher_output.attentions
    )

    losses = {
        "task": task_loss,
        "prediction": prediction_loss,
        "hidden": hidden_loss,
        "attention": attention_loss,
    }
    measurement = probe.measure(losses, step=global_step, epoch=epoch)
    if measurement is not None:
        print(
            f"step={global_step} "
            f"P={measurement['cos_task_prediction']:+.3f} "
            f"H={measurement['cos_task_hidden']:+.3f} "
            f"A={measurement['cos_task_attention']:+.3f}"
        )

    # This is still the normal fixed-weight TinyBERT update.
    total_loss = (
        task_weight * task_loss
        + prediction_weight * prediction_loss
        + hidden_weight * hidden_loss
        + attention_weight * attention_loss
    )
    total_loss.backward()
    optimizer.step()
```

At the end of training, write and print the aggregate result:

```python
probe.write_summary("runs/baseline_seed_42/gradient_conflict_summary.json")
print(probe.format_summary())
probe.close()
```

The CSV includes losses, gradient norms, and the three cosine values per sampled
batch. The JSON includes the percentage of valid sampled batches with negative
cosine, mean cosine, and mean negative cosine conditional on conflict.

## Important details

- Measure before `total_loss.backward()`. This preserves the graph for the real
  update and keeps the measurement independent of the loss weights.
- With gradient accumulation, pass the micro-batch counter if "every tenth
  batch" is intended, or the optimizer-step counter if "every tenth update" is
  intended. Do not mix the two across runs.
- Under mixed precision, pass the original unscaled component losses. Dot
  products and norms are accumulated in float32.
- A zero-norm gradient produces `NaN` in the CSV and is excluded from the
  negative-percentage denominator. This prevents an unused signal from being
  mislabeled as non-conflicting.
- `gradient_mode="log_loss"` implements the proposal's
  `grad(log(L + epsilon))` normalization. It changes reported norms, but not
  cosine direction for positive losses.
