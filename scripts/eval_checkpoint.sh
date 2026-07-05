#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/eval"}
GAMES=${GAMES:-140}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}

if [ -z "${PT_DIR:-}" ] || [ -z "${SAMPLE_ML_DIR:-}" ] || [ -z "${CHECKPOINT:-}" ]; then
  echo "Set PT_DIR, SAMPLE_ML_DIR, and CHECKPOINT before running." >&2
  exit 2
fi

COMMON_ARGS=(
  --pt_dir "${PT_DIR}"
  --sample_ml_dir "${SAMPLE_ML_DIR}"
  --checkpoint "${CHECKPOINT}"
  --eval_only
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

exec "${PYTHON}" "${PROJECT_ROOT}/main.py" "${COMMON_ARGS[@]}" "$@"
