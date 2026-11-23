#!/usr/bin/env bash
# These two commands were executed as separate logged processes after the gates above.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
"$py" -u code/scripts/diagnose_pose_imitation.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --run runs/pose-robustness-v2/imitation-flower42-reference-300/checkpoint-0100 \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 --case insert_flower:42 \
 --output runs/pose-robustness-v2/imitation-flower42-reference100-predictions
"$py" -u code/scripts/verify_pose_imitation_gate.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --run runs/pose-robustness-v2/imitation-fruit42-reference-300/checkpoint-0100 \
 --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 --task select_fruit --seed 42 \
 --stage base --repeats 3 --stop-on-failure --execute 5 \
 --output runs/pose-robustness-v2/imitation-gate-published-reference-fruit42
