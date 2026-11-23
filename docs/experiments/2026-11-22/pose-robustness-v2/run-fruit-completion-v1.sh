#!/usr/bin/env bash
# Run from /root/autodl-tmp/Pi05-VLABench after uploading code and configs.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
root=runs/pose-robustness-v2/fruit-completion-v1
parent=runs/pose-robustness-v2/two-task-conservative-90
base=.local/checkpoints/vlabench-primitive-rlinf/model.safetensors
mkdir -p "$root"
date --iso-8601=seconds > "$root/started-at.txt"
trap 'status=$?; printf "%s\n" "$status" > "$root/exit-status.txt"; date --iso-8601=seconds > "$root/stopped-at.txt"' EXIT
"$py" -u code/scripts/register_fruit_completion.py --project-root "$PWD" \
  > "$root/registration.log" 2>&1
"$py" -u code/scripts/check_reference_rl_handoff.py \
  --base "$base" --fork "$reference_fork" --run "$parent" \
  --episode .local/reference-aligned/msaa0-expert-replays-full/train-select_fruit-42.h5 \
  --paired-episodes runs/pose-robustness-v2/two-task-conservative-development/cases/select_fruit-1201-official/episodes.jsonl \
  --eval-seed 1201 --output "$root/handoff.json" > "$root/handoff.log" 2>&1
"$py" -u code/scripts/ppo_reference_aligned_fruit.py \
  --base "$base" --fork "$reference_fork" --sft-run "$parent" \
  --output "$root/ppo" --seeds 1301 1302 --iterations 1 \
  --max-steps 600 --shaping 0.5 --seed 372 --lr 0.000002 --critic-lr 0.0001 \
  > "$root/ppo.log" 2>&1
"$py" -u code/scripts/evaluate_pose_development.py \
  --base "$base" --fork "$reference_fork" --run "$parent" \
  --cases .local/reference-aligned/msaa0-development-cases \
  --protocol code/configs/evaluation-fruit-completion-v1.json \
  --tasks select_fruit --stage ppo --ppo-weights "$root/ppo/ppo-1.pt" --workers 3 \
  --output "$root/development-ppo" > "$root/development-ppo.log" 2>&1
for suite in ordinary fruit_language; do
  for stage in base sft ppo; do
    extra=()
    if [[ "$stage" == ppo ]]; then
      extra=(--ppo-weights "$root/ppo/ppo-1.pt")
    fi
    "$py" -u code/scripts/evaluate_pose_development.py \
      --base "$base" --fork "$reference_fork" --run "$parent" \
      --cases "$root/cases" --protocol code/configs/evaluation-fruit-completion-v1.json \
      --tasks select_fruit --suite "$suite" --stage "$stage" --workers 3 "${extra[@]}" \
      --output "$root/$suite-$stage" > "$root/$suite-$stage.log" 2>&1
  done
done
