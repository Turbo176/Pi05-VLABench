#!/usr/bin/env bash
# Independently score one frozen language stage while ordinary evaluation runs.
set -euo pipefail
stage=${1:?Specify base or ppo}
[[ "$stage" == base || "$stage" == ppo ]]
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
root=runs/pose-robustness-v2/fruit-completion-v1
test ! -e "$root/fruit_language-$stage/completed.json"
cp code/scripts/run-fruit-language-overlap-v1.sh "$root/sources/scripts/"
extra=()
if [[ "$stage" == ppo ]]; then extra=(--ppo-weights "$root/ppo/ppo-1.pt"); fi
exec "$py" -u code/scripts/evaluate_pose_development.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
  --fork "$reference_fork" --run runs/pose-robustness-v2/two-task-conservative-90 \
  --cases "$root/cases" --protocol code/configs/evaluation-fruit-completion-v1.json \
  --tasks select_fruit --suite fruit_language --stage "$stage" --workers 3 "${extra[@]}" \
  --output "$root/fruit_language-$stage"
