#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
out=runs/pose-robustness-v2
for updates in 30 90 180; do
 run="$out/two-task-instruction-balanced-180"
 if (( updates < 180 )); then printf -v checkpoint 'checkpoint-%04d' "$updates"; run="$run/$checkpoint"; fi
 while [[ ! -s $run/summary.json ]]; do
  if ! pgrep -f 'bash code/scripts/run-two-task-instruction-balanced.sh' >/dev/null; then
   echo "Training stopped before saving $updates" >&2; exit 1
  fi
  sleep 5
 done
 "$py" -u - "$run" "$updates" > "logs/pose-robustness-v2/two-task-balanced-development-$updates.log" 2>&1 <<'PY_REMOTE'
import runpy,sys,torch
run,updates=sys.argv[1:]
torch.cuda.set_per_process_memory_fraction(0.29)
sys.argv=['code/scripts/evaluate_pose_development.py',
 '--base','.local/checkpoints/vlabench-primitive-rlinf/model.safetensors',
 '--fork','.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387',
 '--run',run,'--cases','.local/reference-aligned/msaa0-development-cases',
 '--protocol','code/configs/evaluation-two-task-v3.json','--stage','sft','--workers','3',
 '--output',f'runs/pose-robustness-v2/two-task-balanced-development-{updates}']
runpy.run_path(sys.argv[0],run_name='__main__')
PY_REMOTE
done
