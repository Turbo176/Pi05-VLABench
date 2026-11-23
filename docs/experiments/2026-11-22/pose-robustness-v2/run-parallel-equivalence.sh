#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
exec "$py" -u - <<'PY'
import runpy, sys, torch
# Keep this verification process inside the memory left by the running trainer.
torch.cuda.set_per_process_memory_fraction(0.30)
sys.argv = ['code/scripts/evaluate_pose_development.py',
 '--base', '.local/checkpoints/vlabench-primitive-rlinf/model.safetensors',
 '--fork', '.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387',
 '--run', 'runs/pose-robustness-v2/two-task-smoke',
 '--cases', '.local/reference-aligned/msaa0-development-cases',
 '--protocol', 'code/configs/evaluation-two-task-v1.json',
 '--tasks', 'select_fruit', 'insert_flower', '--stage', 'base', '--workers', '3',
 '--verify-against', 'logs/pose-robustness-v2/reference-serial-development.jsonl',
 '--output', 'runs/pose-robustness-v2/parallel-equivalence']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
