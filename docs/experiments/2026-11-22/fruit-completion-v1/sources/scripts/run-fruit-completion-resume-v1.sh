#!/usr/bin/env bash
# Resume only the frozen evaluation after verified reuse of old reference records.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
root=runs/pose-robustness-v2/fruit-completion-v1
trap 'status=$?; printf "%s\n" "$status" > "$root/exit-status.txt"; date --iso-8601=seconds > "$root/stopped-at.txt"' EXIT
test -s "$root/reference-reuse.json"
test -s "$root/development-ppo/completed.json"
test -s "$root/ppo/ppo-1.pt"
date --iso-8601=seconds > "$root/resumed-at.txt"
cp code/scripts/run-fruit-completion-resume-v1.sh "$root/sources/scripts/"
for suite in ordinary fruit_language; do
  for stage in base sft ppo; do
    if [[ -s "$root/$suite-$stage/completed.json" ]]; then
      continue
    fi
    extra=()
    if [[ "$stage" == ppo ]]; then
      extra=(--ppo-weights "$root/ppo/ppo-1.pt")
    fi
    "$py" -u code/scripts/evaluate_pose_development.py \
      --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
      --fork "$reference_fork" --run runs/pose-robustness-v2/two-task-conservative-90 \
      --cases "$root/cases" --protocol code/configs/evaluation-fruit-completion-v1.json \
      --tasks select_fruit --suite "$suite" --stage "$stage" --workers 3 "${extra[@]}" \
      --output "$root/$suite-$stage" > "$root/$suite-$stage.log" 2>&1
  done
done
