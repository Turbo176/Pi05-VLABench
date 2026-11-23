#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u code/scripts/train_reference_aligned_fruit.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --reference-start .local/checkpoints/vlabench-primitive-rlinf \
 --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 \
 --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl \
 --diagnostic-case insert_flower:42 --render-samples 0 --shortest-euler \
 --actor-lora-rank 16 --vlm-lora-rank 8 --lr 0.0001 --no-image-augmentation \
 --gradient-accumulation 4 --micro-batch-size 2 --updates 300 --save-every 100 --skip-evaluation \
 --output runs/pose-robustness-v2/imitation-flower42-vlm-300
