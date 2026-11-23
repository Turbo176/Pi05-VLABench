#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
# This checkpoint supplies only the verified interface and normalization.
# The base evaluator does not restore any trained adapter weights.
# Keep final outcomes unopened until development-only checkpoint selection ends.
task=${1:-select_fruit}
mode=${2:-official}
suite=${3:-ordinary}
case_name="$task-$mode"
if [[ "$suite" == fruit_language && "$mode" == official ]]; then
  case_name=select_fruit-language-official
fi
exec "$py" -u code/scripts/train_reference_aligned_fruit.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
  --fork "$reference_fork" --load-run runs/pose-robustness-v2/correction-batch3-smoke \
  --eval-stage base --eval-task "$task" --prompt-mode "$mode" --render-samples 0 \
  --paired-episodes ".local/reference-aligned/msaa0-final-cases-v2/$case_name.jsonl" \
  --resume-evaluation \
  --output "runs/pose-robustness-v2/final-comparison/base-$suite-$task-$mode"
