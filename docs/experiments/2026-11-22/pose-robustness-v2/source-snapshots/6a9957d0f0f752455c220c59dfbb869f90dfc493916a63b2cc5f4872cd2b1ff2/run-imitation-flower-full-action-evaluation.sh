#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/verify_pose_imitation_gate.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --run runs/pose-robustness-v2/imitation-flower42-full-action-100 \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 \
 --task insert_flower --seed 42 --repeats 3 --video --stop-on-failure \
 --output runs/pose-robustness-v2/imitation-gate-full-action100-flower42
