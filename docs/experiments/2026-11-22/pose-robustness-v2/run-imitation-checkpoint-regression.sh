#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/audit_pose_checkpoint_regression.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --early-run runs/pose-robustness-v2/imitation-fruit42-reference-300/checkpoint-0100 \
 --late-run runs/pose-robustness-v2/imitation-fruit42-reference-300 \
 --early-gate runs/pose-robustness-v2/imitation-gate-reference-100-fruit42/gate.json \
 --late-gate runs/pose-robustness-v2/imitation-gate-reference-300-fruit42/gate.json \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 --case select_fruit:42 \
 --output runs/pose-robustness-v2/imitation-fruit42-checkpoint-regression
