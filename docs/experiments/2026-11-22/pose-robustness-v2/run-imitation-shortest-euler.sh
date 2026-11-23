#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/train_reference_aligned_fruit.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
  --fork "$reference_fork" \
  --reference-start .local/checkpoints/vlabench-primitive-rlinf \
  --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 \
  --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl \
  --render-samples 0 --shortest-euler --diagnostic-case select_fruit:42 --diagnostic-rollout \
  --gradient-accumulation 4 --actor-lora-rank 16 --vlm-lora-rank 8 \
  --freeze-action-heads --lr 0.0003 --gripper-loss-weight 5 \
  --no-image-augmentation --updates 200 --save-every 100 \
  --skip-evaluation --video \
  --output runs/pose-robustness-v2/imitation-fruit42-shortest-200
