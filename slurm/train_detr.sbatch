#!/bin/bash
# Slurm batch script to train one RT-DETR variant on Curta (FU Berlin HPC), unattended overnight.
#
# Verified against the real cluster (2026-07-30):
#   - `sinfo` confirmed a partition literally named "gpu" (g001-008) - matches --partition below.
#   - `conda` is not on PATH by default; loaded via the `Anaconda3` module instead.
#   - No separate CUDA module load is needed: the pip-installed PyTorch wheel (see below) bundles
#     its own CUDA runtime; only the GPU driver on the node matters, and that's already present on
#     GPU nodes.
#
# Usage (submit one job per model variant you want to compare):
#   sbatch train_detr.sbatch rtdetr-l_640  rtdetr-l.pt 640  4
#   sbatch train_detr.sbatch rtdetr-l_1024 rtdetr-l.pt 1024 2
#   sbatch train_detr.sbatch rtdetr-x_1024 rtdetr-x.pt 1024 2
#
# Positional args: $1 = run name, $2 = model checkpoint, $3 = imgsz, $4 = batch size
# (all optional, see defaults below). Curta's gpu partition has 11GB Pascal-era nodes (GTX 1080 Ti)
# that can't run AMP (see train_detr.py) — training runs in FP32, which needs ~2x the memory of the
# batch=8 default this was originally tuned for. Pick a conservative --batch per imgsz yourself
# instead of relying on Ultralytics' automatic batch-size-reduction-on-OOM: repeated failed
# allocation attempts inside one process fragment the CUDA allocator and can leave it unable to find
# a working cuDNN algorithm even at batch=1 ("RuntimeError: Unable to find a valid cuDNN algorithm
# to run convolution" — a real failure this hit in practice). nbs=64 (see train_detr.py) gradient-
# accumulates up to an effective batch of 64 regardless of the physical --batch here, so a small
# physical batch does not mean a proportionally worse-trained model, just more (slower) steps/epoch.

#SBATCH --job-name=detr_train
#SBATCH --partition=gpu
#SBATCH --qos=standard                 # <-- CHECK: run `sqos` on Curta; adjust if "standard" isn't
                                        #     valid for your account/partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:00:00                # slightly under 12h so it comfortably fits "overnight"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

RUN_NAME="${1:-rtdetr-l_1024}"
MODEL="${2:-rtdetr-l.pt}"
IMGSZ="${3:-1024}"
BATCH="${4:-4}"

echo "Job $SLURM_JOB_ID on $(hostname), run=$RUN_NAME model=$MODEL imgsz=$IMGSZ batch=$BATCH"
nvidia-smi || echo "WARNING: nvidia-smi failed — check this job actually landed on a GPU node"

PROJECT_DIR="$HOME/dogs_project/cv"
ENV_NAME="detr_env"

# --- one-time environment setup (idempotent: skipped if the env already exists) -----------------
# Wrapped in a flock: if you submit multiple runs at once (recommended, so they queue/train in
# parallel), they'd otherwise all race to `conda create`/`pip install` into the SAME shared env
# simultaneously, which can corrupt it. The lock makes every job's setup step wait its turn; once
# the env exists, later jobs' checks are near-instant and they proceed to training right away.
#
# Defensively reset any conda state inherited from the submitting shell: Slurm exports your
# interactive shell's environment into the job by default, so if you had `conda activate detr_env`
# (or any other env) active when you ran `sbatch`, conda's activate below can wrongly think it's
# already in the right env and skip updating PATH — the job then silently runs with whatever
# Python/torch the *submitting shell* had (this caused a real failure: job ran with the base
# Anaconda module's Python 3.12 + an incompatible torch build instead of detr_env's fixed
# torch==2.5.1). Unsetting these first guarantees a clean, reproducible environment every time,
# regardless of your shell state at submission time.
for i in $(seq "${CONDA_SHLVL:-0}"); do conda deactivate 2>/dev/null || true; done
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_EXE _CE_CONDA _CE_M

module load Anaconda3/2024.06-1

ENV_LOCK="$PROJECT_DIR/.env_setup.lock"
mkdir -p "$PROJECT_DIR"

(
    flock -x 200
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "Creating conda env $ENV_NAME (first run only, will take a few minutes)..."
        conda create -y -n "$ENV_NAME" python=3.11
    fi

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    pip install -q --upgrade pip
    # Pinned to 2.5.1: PyTorch dropped CUDA kernels for compute capability sm_61 (Pascal, e.g.
    # GTX 1080 Ti) in the 2.8 release — Curta's gpu partition has Pascal-era nodes, and anything
    # >=2.8 fails there with "CUDA error: no kernel image is available for execution on the device".
    # 2.5.1 is the newest version still under that cutoff AND the newest one actually published
    # under the cu121 wheel index (cu121 only goes up to 2.5.1; later torch releases moved to
    # cu124/cu126/etc, which is why an earlier attempt to pin 2.7.1 here failed with
    # "Could not find a version that satisfies the requirement torch==2.7.1").
    # If a job lands on a newer GPU (Ampere/Hopper etc.) this build still works fine, it's just not
    # the newest CUDA toolkit.
    pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    # -c (constraints file) forces pip to keep torch==2.5.1 even while resolving requirements.txt's
    # unpinned `ultralytics` — without this, ultralytics' own dependency resolution silently
    # upgrades torch to the latest release again (which lacks sm_61/Pascal kernels, see above),
    # undoing the pin from the line above. This was a real, repeated failure cause.
    echo "torch==2.5.1" > "$PROJECT_DIR/scripts/.torch_constraint.txt"
    pip install -q -r "$PROJECT_DIR/scripts/requirements.txt" -c "$PROJECT_DIR/scripts/.torch_constraint.txt"
) 200>"$ENV_LOCK"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- run training -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Reduces CUDA allocator fragmentation from any OOM-retry that still happens despite the
# conservative --batch above — segments can grow/shrink instead of requiring one contiguous block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_detr.py \
    --run-name "$RUN_NAME" \
    --model "$MODEL" \
    --imgsz "$IMGSZ" \
    --data-dir "$PROJECT_DIR/data" \
    --work-dir "$PROJECT_DIR/detr_work" \
    --out-dir "$PROJECT_DIR/runs" \
    --epochs 60 \
    --batch "$BATCH" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --max-hours 10.5 \
    --resume

echo "Job finished. Results in $PROJECT_DIR/runs/$RUN_NAME/"
