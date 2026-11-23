#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
out=runs/pose-robustness-v2
while [[ ! -s $out/two-task-reference-1200/summary.json ]]; do
    if ! pgrep -f 'train_reference_aligned_fruit.py .*--output runs/pose-robustness-v2/two-task-reference-1200$' >/dev/null; then
        echo 'Main training ended without a completed checkpoint.' >&2
        exit 1
    fi
    sleep 10
done
common=(--base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors
        --fork "$reference_fork" --reference-start .local/checkpoints/vlabench-primitive-rlinf
        --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2
        --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras
        --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-105-134
        --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl
        --multitask --training-tasks select_fruit insert_flower --render-samples 0
        --gradient-accumulation 2 --micro-batch-size 4
        --actor-lora-rank 16 --vlm-lora-rank 0 --lr 0.000001 --no-image-augmentation)
echo "$(date -Is) Starting conservative-recipe smoke check"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --smoke --skip-evaluation \
    --output "$out/two-task-conservative-smoke" > logs/pose-robustness-v2/two-task-conservative-smoke.log 2>&1
echo "$(date -Is) Starting fresh 90-update shared two-task SFT"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --updates 90 --skip-evaluation \
    --output "$out/two-task-conservative-90" > logs/pose-robustness-v2/two-task-conservative-90.log 2>&1
echo "$(date -Is) Starting all eight fixed development cases"
exec "$py" -u code/scripts/evaluate_pose_development.py \
    --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
    --fork "$reference_fork" --run "$out/two-task-conservative-90" \
    --cases .local/reference-aligned/msaa0-development-cases \
    --protocol code/configs/evaluation-two-task-v2.json --stage sft --workers 3 \
    --output "$out/two-task-conservative-development" \
    > logs/pose-robustness-v2/two-task-conservative-development.log 2>&1
