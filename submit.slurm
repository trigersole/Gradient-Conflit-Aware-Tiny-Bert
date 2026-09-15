#!/bin/bash
#SBATCH --job-name=tinybert-conflict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs

# Change these module names to those provided by your cluster.
module purge
module load python
module load cuda
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${SCRATCH:-${SLURM_SUBMIT_DIR}}/huggingface-cache"

nvidia-smi
EXTRA_ARGS=()
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke-test)
fi

srun python train_tinybert.py \
  --output-dir "runs/${SLURM_JOB_ID}" \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --measure-every 10 \
  --temperature 4.0 \
  --seed 42 \
  --fp16 \
  "${EXTRA_ARGS[@]}"
