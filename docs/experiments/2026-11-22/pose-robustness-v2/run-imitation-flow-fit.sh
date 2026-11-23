#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/audit_pose_flow_fit.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 --case insert_flower:42 \
 --run runs/pose-robustness-v2/imitation-flower42-reference-300/checkpoint-0100 \
 --run runs/pose-robustness-v2/imitation-flower42-reference-300 \
 --output runs/pose-robustness-v2/imitation-flower42-fixed-flow-fit.json
