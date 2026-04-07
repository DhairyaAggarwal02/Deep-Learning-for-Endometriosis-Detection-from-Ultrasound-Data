#!/bin/bash
#SBATCH --job-name=endo_res50_5fold
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/st-ilker-1/dhairya/resnet50_5fold/logs/slurm-%j.out
#SBATCH --error=/scratch/st-ilker-1/dhairya/resnet50_5fold/logs/slurm-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"

# Activate venv
source /home/daggarwa/venvs/endo/bin/activate

echo "Python: $(which python)"
python -V

# -----------------------------
# Output dirs
# -----------------------------
BASE_OUT="/scratch/st-ilker-1/dhairya/resnet50_5fold"
mkdir -p "${BASE_OUT}/logs"

# Tag this run with key hyperparams for easy comparison
SEED=42
BATCH_SIZE=64
LR_BACKBONE=3e-5
MIXUP_ALPHA=0.2
LABEL_SMOOTHING=0.05
GRAD_CLIP=1.0

RUN_TAG="bs${BATCH_SIZE}_bb${LR_BACKBONE}_mix${MIXUP_ALPHA}_ls${LABEL_SMOOTHING}_gc${GRAD_CLIP}_seed${SEED}"
RUN_DIR="${BASE_OUT}/run_${SLURM_JOB_ID}_${RUN_TAG}"
OUT_DIR="${RUN_DIR}/out"
mkdir -p "${RUN_DIR}" "${OUT_DIR}"

# -----------------------------
# Persistent caches (OFFLINE SAFE)
# -----------------------------
export TORCH_HOME="/scratch/st-ilker-1/dhairya/torch_cache"
export XDG_CACHE_HOME="/scratch/st-ilker-1/dhairya/.cache"
export MPLCONFIGDIR="/scratch/st-ilker-1/dhairya/.mpl_cache"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Confirm GPU
nvidia-smi

# -----------------------------
# Paths
# -----------------------------
# IMPORTANT: this must be the folder that contains fold_1..fold_5 and test/
DATA_ROOT="/home/daggarwa/490_project/dhairya/endometriosis_cv"

# Point to the ResNet50 script
SCRIPT_PATH="/scratch/st-ilker-1/dhairya/resnet50/train_Res50_5fold.py"

echo "RUN_DIR: $RUN_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "DATA_ROOT: $DATA_ROOT"
echo "TORCH_HOME: $TORCH_HOME"
echo "RUN_TAG: $RUN_TAG"
echo "SCRIPT_PATH: $SCRIPT_PATH"

# -----------------------------
# Sanity check: pretrained weights must exist
# -----------------------------
WEIGHTS_FILE="${TORCH_HOME}/hub/checkpoints/resnet50-11ad3fa6.pth"
if [ ! -f "$WEIGHTS_FILE" ]; then
  echo "ERROR: Missing pretrained weights: $WEIGHTS_FILE"
  exit 1
fi
echo "Found pretrained weights: $WEIGHTS_FILE"

# -----------------------------
# Optional: log GPU stats every 60s in background
# -----------------------------
GPU_LOG="${RUN_DIR}/gpu_usage.csv"
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 60 > "$GPU_LOG" ) &
GPU_LOG_PID=$!
trap "kill $GPU_LOG_PID || true" EXIT

# -----------------------------
# Run 5-fold CV training + separate test eval
# -----------------------------
python "${SCRIPT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --img_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 16 \
  --max_epochs 200 \
  --phase1_epochs 15 \
  --early_patience 10 \
  --lr_patience 3 \
  --lr_factor 0.5 \
  --lr_head 1e-3 \
  --lr_backbone "${LR_BACKBONE}" \
  --weight_decay 1e-4 \
  --mixup_alpha "${MIXUP_ALPHA}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --grad_clip_norm "${GRAD_CLIP}"

echo "Finished at: $(date)"
echo "Outputs saved to: ${OUT_DIR}"
echo "GPU log saved to: ${GPU_LOG}"