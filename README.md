# TinyBERT gradient-conflict experiment

This repository runs a complete fixed-weight TinyBERT-style SST-2 experiment
and measures these quantities every tenth training batch:

- `cos(g_task, g_prediction)`
- `cos(g_task, g_hidden)`
- `cos(g_task, g_attention)`

The diagnostic does not modify the training gradients or dynamically change
loss weights. This is the baseline experiment to establish whether the three
teacher signals conflict with the supervised task gradient.

## Default models

| Role | Hugging Face checkpoint | Layers | Hidden | Heads |
|---|---|---:|---:|---:|
| Fine-tuned teacher | `yoshitomo-matsubara/bert-base-uncased-sst2` | 12 | 768 | 12 |
| Pretrained student | `huawei-noah/TinyBERT_General_4L_312D` | 4 | 312 | 12 |

The teacher is already fine-tuned on SST-2 and is frozen throughout the run.
The pipeline checks its validation accuracy before training the student and
stops if it is below 80%. No teacher fine-tuning stage is performed. Both the
teacher checkpoint and the official TinyBERT project declare Apache-2.0 terms.

Student layers 1–4 are matched to teacher layers 3, 6, 9 and 12. Learned linear
layers project hidden states from 312 to 768 dimensions. Both models have 12
attention heads, so the attention loss compares corresponding heads directly.
Padding is excluded from hidden and attention losses.

## Files

- `train_tinybert.py`: dataset, models, four losses, training, evaluation and
  artifact saving.
- `gradient_conflict.py`: gradient cosine measurement and summary statistics.
- `submit.slurm`: one-GPU SLURM job, including an optional smoke-test mode.
- `requirements.txt`: Python dependencies.
- `tests/`: numerical checks for the probe, layer mapping and losses.

## Local or interactive GPU setup

Use Python 3.10 or newer. Install the PyTorch build recommended by your cluster
for its CUDA driver, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch
pip install -r requirements.txt
```

Verify that PyTorch sees the allocated GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## SLURM quick start

Clone the repository on the cluster and create the environment once on a login
or interactive node:

```bash
git clone https://github.com/trigersole/Gradient-Conflit-Aware-Tiny-Bert.git
cd Gradient-Conflit-Aware-Tiny-Bert
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch
pip install -r requirements.txt
```

Edit `submit.slurm` for your cluster. In particular, check its partition,
account, module names, time limit and GPU request. Some clusters use
`--gpus=1` instead of `--gres=gpu:1`.

Run the ten-step smoke test first:

```bash
mkdir -p logs
sbatch --export=ALL,SMOKE_TEST=1 submit.slurm
```

Watch it with:

```bash
squeue -u "$USER"
tail -f logs/tinybert-conflict-JOB_ID.out
```

After the smoke test succeeds, submit the full three-epoch run:

```bash
sbatch submit.slurm
```

The first run downloads the two model checkpoints and GLUE/SST-2 into
`$HF_HOME`. If compute nodes cannot access the internet, start an interactive
GPU/login session with internet access, set the same `HF_HOME`, and run the
smoke test once there so the model and dataset caches are populated.

## Direct command

The equivalent direct training command is:

```bash
python train_tinybert.py \
  --output-dir runs/seed-42 \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --temperature 4.0 \
  --measure-every 10 \
  --seed 42 \
  --fp16
```

For a quick functional check:

```bash
python train_tinybert.py --smoke-test --output-dir runs/smoke --fp16
```

## Losses and measurement

The ordinary student update is the fixed weighted sum

```text
L = w_task L_task + w_prediction L_prediction
    + w_hidden L_hidden + w_attention L_attention
```

All weights default to 1.0 and are configurable with `--task-weight`,
`--prediction-weight`, `--hidden-weight` and `--attention-weight`.
Prediction distillation uses temperature-scaled KL divergence. Task loss is
cross-entropy. Hidden and attention losses are padding-masked MSE values.

The gradient probe runs before the normal backward pass using
`torch.autograd.grad`, which leaves `.grad` untouched. It measures only student
embeddings and transformer-block parameters—not classifier, pooler or hidden
projection parameters. A zero-norm measurement is written as `NaN` and excluded
from the conflict-frequency denominator.

## Outputs

Each output directory contains:

```text
gradient_cosines.csv
gradient_conflict_summary.json
results.json
run_config.json
hidden_projections.pt
student/
```

`gradient_cosines.csv` contains the three cosine values, individual losses and
gradient norms for each sampled batch. `gradient_conflict_summary.json` reports
the negative-cosine percentage, mean cosine and conditional mean negative
cosine for each teacher signal. `results.json` also records teacher/student
validation accuracy, layer mapping, loss weights, runtime and seed.

The final console output includes a table such as:

```text
| Signal     | Batches with negative cosine | Valid sampled batches |
|------------|------------------------------:|----------------------:|
| Prediction |                          4.0% |                   200 |
| Hidden     |                         18.0% |                   200 |
| Attention  |                         35.0% |                   200 |
```

Run at least three seeds before treating the observed percentages as a result.
Keep the model checkpoints, loss weights, batch size, sampling interval and
dataset split identical between seeds.
