#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
out=runs/pose-robustness-v2
common=(--base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors
 --fork "$reference_fork" --reference-start .local/checkpoints/vlabench-primitive-rlinf
 --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2
 --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras
 --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-105-134
 --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-tulip-181-205
 --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl
 --multitask --training-tasks select_fruit insert_flower --render-samples 0
 --gradient-accumulation 2 --micro-batch-size 4
 --actor-lora-rank 16 --vlm-lora-rank 0 --lr 0.000001 --no-image-augmentation
 --instruction-balanced)
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --smoke --skip-evaluation \
 --output "$out/two-task-instruction-balanced-smoke" > logs/pose-robustness-v2/two-task-instruction-balanced-smoke.log 2>&1
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --updates 180 --save-every 30 --skip-evaluation \
 --output "$out/two-task-instruction-balanced-180" > logs/pose-robustness-v2/two-task-instruction-balanced-180.log 2>&1
