#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
for updates in 100 300; do
 run=runs/pose-robustness-v2/imitation-flower42-vlm-300
 if (( updates == 100 )); then run="$run/checkpoint-0100"; fi
 while [[ ! -s $run/summary.json ]]; do
  if ! pgrep -f 'train_reference_aligned_fruit.py .*--output runs/pose-robustness-v2/imitation-flower42-vlm-300$' >/dev/null; then
   echo "Training ended before checkpoint $updates" >&2; exit 1
  fi
  sleep 5
 done
 "$py" -u code/scripts/verify_pose_imitation_gate.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" \
  --run "$run" --dataset .local/reference-aligned/lerobot-tritask-msaa0-v2 \
  --task insert_flower --seed 42 --repeats 3 --video --stop-on-failure \
  --output "runs/pose-robustness-v2/imitation-gate-vlm-$updates-flower42" \
  > "logs/pose-robustness-v2/imitation-gate-vlm-$updates-flower42.log" 2>&1 || {
   status=$?; if (( status != 2 )); then exit "$status"; fi
  }
done
