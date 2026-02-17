#!/bin/bash
#SBATCH --job-name=viz2psych
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_viz2psych.out
#SBATCH --error=slurm/logs/%j_viz2psych.err

# ---------- Environment ----------
module load cuda/12.4.1
conda activate viz2psych

# Ensure the package is importable.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# ---------- Run ----------
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/run_all.py "$@"

echo "Job finished at $(date)"
