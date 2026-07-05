#!/usr/bin/env bash
#SBATCH --job-name=distraj_baller2vec_ratio1
#SBATCH --output=logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/slurm_baller2vec_ratio1"}
GAMES=${GAMES:-140}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}

if [ -z "${PT_DIR:-}" ] || [ -z "${SAMPLE_ML_DIR:-}" ]; then
  echo "Set PT_DIR and SAMPLE_ML_DIR before submitting this job." >&2
  exit 2
fi

mkdir -p "${PROJECT_ROOT}/logs"

COMMON_ARGS=(
  --pt_dir "${PT_DIR}"
  --sample_ml_dir "${SAMPLE_ML_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --games "${GAMES}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --traj_model baller2vec
  --event_head disentangled
  --dynamic_weight 1.0
  --use_context
  --use_voronoi_spatial
)

if [ -n "${LIMIT_TRAIN_FILES:-}" ]; then
  COMMON_ARGS+=(--limit_train_files "${LIMIT_TRAIN_FILES}")
fi
if [ -n "${LIMIT_VAL_FILES:-}" ]; then
  COMMON_ARGS+=(--limit_val_files "${LIMIT_VAL_FILES}")
fi
if [ -n "${LIMIT_TEST_FILES:-}" ]; then
  COMMON_ARGS+=(--limit_test_files "${LIMIT_TEST_FILES}")
fi

if [ -n "${CHECKPOINT:-}" ]; then
  COMMON_ARGS+=(--checkpoint "${CHECKPOINT}" --eval_only)
else
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi

exec "${PYTHON}" -m torch.distributed.run   --nproc_per_node="${NPROC_PER_NODE}"   --master_port="${MASTER_PORT}"   "${PROJECT_ROOT}/main.py"   "${COMMON_ARGS[@]}"   "$@"
