#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/collect_reference_pose_recovery.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --run runs/pose-robustness-v2/imitation-flower42-canonical-lr1e4-300 \
 --task insert_flower --seed 211 212 213 --lead 80 \
 --output .local/reference-aligned/msaa0-flower-corrections-211-213
