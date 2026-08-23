#!/bin/bash
# Slurm batch script: run scripts/predict_submission.py on Curta to build a Kaggle
# submission.csv from a trained RT-DETR checkpoint (much faster than CPU inference locally).
#
# Usage:
#   sbatch predict_detr.sbatch rtdetr-l_640 640
#   sbatch predict_detr.sbatch rtdetr-l_1024 1024
#   sbatch predict_detr.sbatch rtdetr-x_1024 1024
#
# Positional args: $1 = run name, $2 = imgsz (must match how that run was trained)

#SBATCH --job-name=detr_predict
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch predict_detr.sbatch <run-name> <imgsz>}"
IMGSZ="${2:?Usage: sbatch predict_detr.sbatch <run-name> <imgsz>}"

echo "Job $SLURM_JOB_ID on $(hostname), run=$RUN_NAME imgsz=$IMGSZ"
nvidia-smi || echo "WARNING: nvidia-smi failed — check this job actually landed on a GPU node"

PROJECT_DIR="$HOME/dogs_project/cv"
ENV_NAME="detr_env"

for i in $(seq "${CONDA_SHLVL:-0}"); do conda deactivate 2>/dev/null || true; done
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_EXE _CE_CONDA _CE_M

module load Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

python scripts/predict_submission.py \
    --run-name "$RUN_NAME" \
    --imgsz "$IMGSZ" \
    --data-dir "$PROJECT_DIR/data" \
    --runs-dir "$PROJECT_DIR/runs" \
    --out "$PROJECT_DIR/runs/$RUN_NAME/submission.csv"

echo "Job finished. Submission file: $PROJECT_DIR/runs/$RUN_NAME/submission.csv"
