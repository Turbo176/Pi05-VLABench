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
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras \
  --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl \
  --render-samples 0 --shortest-euler --multitask --recovery-share 0.5 \
  --gradient-accumulation 2 --micro-batch-size 3 \
  --actor-lora-rank 16 --vlm-lora-rank 8 --freeze-action-heads \
  --lr 0.00005 --gripper-loss-weight 5 --action-loss executed-poses \
  --no-image-augmentation --updates 400 --save-every 200 --skip-evaluation \
  --output runs/pose-robustness-v2/correction-multitask-400
