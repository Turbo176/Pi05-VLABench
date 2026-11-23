#!/usr/bin/env bash
# Integration check of the conservative candidate; this adds zero optimizer updates.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/check_reference_rl_handoff.py \
    --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
    --fork "$reference_fork" --run runs/pose-robustness-v2/two-task-conservative-90 \
    --episode .local/reference-aligned/msaa0-expert-replays-full/train-select_fruit-42.h5 \
    --episode .local/reference-aligned/msaa0-expert-replays-full/train-insert_flower-42.h5 \
    --paired-episodes runs/pose-robustness-v2/two-task-conservative-development/cases/select_fruit-1201-official/episodes.jsonl \
    --eval-seed 1201 --output runs/pose-robustness-v2/two-task-conservative-handoff.json
