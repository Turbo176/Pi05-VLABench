#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
run=runs/pose-robustness-v2/imitation-flower42-conservative-100
while [[ ! -s $run/summary.json ]]; do
 if ! pgrep -f 'train_reference_aligned_fruit.py .*--output runs/pose-robustness-v2/imitation-flower42-conservative-100$' >/dev/null; then
  echo 'Training exited before its checkpoint' >&2; exit 1
 fi
 sleep 5
done
exec "$py" -u code/scripts/verify_pose_imitation_gate.py \
 --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
 --run "$run" --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 \
 --task insert_flower --seed 42 --repeats 3 --video --stop-on-failure \
 --output runs/pose-robustness-v2/imitation-gate-conservative100-flower42
