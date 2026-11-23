#!/usr/bin/env bash
# Evaluation-only recovery command. Execution is pending GPU service recovery.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/train_reference_aligned_fruit.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
  --fork "$reference_fork" \
  --load-run runs/pose-robustness-v2/imitation-fruit42-shortest-800 \
  --eval-stage sft --eval-task select_fruit --eval-seed 42 \
  --paired-episodes runs/pose-robustness-v2/imitation-fruit42-shortest-800/diagnostic-reset.jsonl \
  --render-samples 0 --max-steps 600 --trace-controls --video \
  --output runs/pose-robustness-v2/imitation-fruit42-shortest-800/training-reset-recovered
