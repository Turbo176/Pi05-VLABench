#!/usr/bin/env bash
# Start the registered development comparisons after serial equivalence is proven.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
out=runs/pose-robustness-v2
while [[ ! -s $out/parallel-equivalence/completed.json ]]; do
    if ! pgrep -f '^/root/autodl-tmp/Pi05-VLABench/.local/torch28-venv/bin/python -u -$' >/dev/null; then
        echo 'Serial-equivalence verification ended without a complete passing report.' >&2
        exit 1
    fi
    sleep 10
done
"$py" - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('runs/pose-robustness-v2/parallel-equivalence/completed.json').read_text())
assert r['serial_equivalence']['identical_cases'] == 8
assert r['cases'] == 8 and r['workers'] == 3
print('All eight serial/parallel trajectories agree exactly.', flush=True)
PY
for updates in 600 1200; do
    run="$out/two-task-reference-1200"
    if (( updates == 600 )); then run="$run/checkpoint-0600"; fi
    while [[ ! -s $run/summary.json ]]; do
        if ! pgrep -f 'train_reference_aligned_fruit.py .*--output runs/pose-robustness-v2/two-task-reference-1200$' >/dev/null; then
            echo "Training ended without checkpoint $updates." >&2
            exit 1
        fi
        sleep 10
    done
    echo "$(date -Is) Starting all eight development cases for update $updates"
    "$py" -u - "$run" "$updates" > "logs/pose-robustness-v2/two-task-development-$updates.log" 2>&1 <<'PY'
import runpy, sys, torch
run, updates = sys.argv[1:]
torch.cuda.set_per_process_memory_fraction(0.30)
sys.argv = ['code/scripts/evaluate_pose_development.py',
 '--base', '.local/checkpoints/vlabench-primitive-rlinf/model.safetensors',
 '--fork', '.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387',
 '--run', run, '--cases', '.local/reference-aligned/msaa0-development-cases',
 '--protocol', 'code/configs/evaluation-two-task-v1.json',
 '--tasks', 'select_fruit', 'insert_flower', '--stage', 'sft', '--workers', '3',
 '--output', f'runs/pose-robustness-v2/two-task-development-{updates}']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
done
echo "$(date -Is) Both complete development comparisons are saved"
