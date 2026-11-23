#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/audit_pose_training_contract.py \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 \
 --norm-stats .local/checkpoints/vlabench-primitive-rlinf/norm_stats.json \
 --fork "$reference_fork" --case select_fruit:42 --case insert_flower:42 \
 --shortest-euler --output runs/pose-robustness-v2/canonical-training-contract-audit.json
