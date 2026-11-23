#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
root=runs/pose-robustness-v2/fruit-completion-v1
until [[ -f "$root/exit-status.txt" ]]; do
  if ! pgrep -f '^bash code/scripts/run-fruit-completion(-resume)?-v1.sh$' >/dev/null; then
    echo 'Pipeline ended without an exit status.' >&2
    exit 1
  fi
  sleep 10
done
if [[ "$(cat "$root/exit-status.txt")" != 0 ]]; then
  echo 'Pipeline did not complete successfully; artifacts were not marked complete.' >&2
  exit 1
fi
trap 'status=$?; printf "%s\n" "$status" > "$root/artifacts-exit-status.txt"; date --iso-8601=seconds > "$root/artifacts-stopped-at.txt"' EXIT
"$py" -u code/scripts/summarize_fruit_completion.py --root "$root" \
  --protocol code/configs/evaluation-fruit-completion-v1.json \
  --reference-development runs/pose-robustness-v2/base-development/select_fruit/episodes.jsonl \
  --sft-development runs/pose-robustness-v2/two-task-conservative-development/episodes.jsonl \
  --output "$root/comparison.json" > "$root/comparison.log" 2>&1
"$py" -u code/scripts/plot_fruit_completion.py --comparison "$root/comparison.json" \
  --output "$root/figures" > "$root/figures.log" 2>&1
for stage in base sft ppo; do
  if [[ -s "$root/video-$stage/verification.json" ]]; then
    continue
  fi
  "$py" -u code/scripts/record_fruit_completion.py --root "$root" \
    --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
    --fork "$reference_fork" --run runs/pose-robustness-v2/two-task-conservative-90 \
    --stage "$stage" > "$root/video-$stage.log" 2>&1
done
