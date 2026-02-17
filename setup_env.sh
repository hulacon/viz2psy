#!/bin/bash
# Create the viz2psych conda environment on Talapas.
# Usage: bash setup_env.sh

set -euo pipefail

ENV_NAME="viz2psych"

echo "Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 -y

echo "Activating ${ENV_NAME} ..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing PyTorch with CUDA 12.4 support ..."
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo "Installing Python dependencies ..."
pip install resmem tqdm pandas Pillow

echo "Installing viz2psych in editable mode ..."
pip install -e /gpfs/projects/hulacon/bhutch/viz2psych

echo ""
echo "Done! Activate with:  conda activate ${ENV_NAME}"
echo "Then run:             sbatch slurm/score_images.sbatch"
