#!/usr/bin/env bash
# Executed from the remote project root, using the existing pinned environment.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
out=runs/pose-robustness-v2
mkdir -p logs/pose-robustness-v2
common=(--base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors
        --fork "$reference_fork" --reference-start .local/checkpoints/vlabench-primitive-rlinf
        --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2
        --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras
        --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-105-134
        --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl
        --multitask --training-tasks select_fruit insert_flower --render-samples 0
        --gradient-accumulation 2 --micro-batch-size 4
        --actor-lora-rank 16 --vlm-lora-rank 8 --lr 0.00001)
echo "$(date -Is) Starting two-task forward/backward and checkpoint check"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --smoke --skip-evaluation \
    --output "$out/two-task-smoke" > logs/pose-robustness-v2/two-task-smoke.log 2>&1
"$py" - <<'PY'
from collections import Counter
import json
from pathlib import Path
p = Path('runs/pose-robustness-v2/two-task-smoke')
r = json.loads((p / 'summary.json').read_text())
assert r['training_tasks'] == ['select_fruit', 'insert_flower']
assert r['train_frames'] == 9904 and len(r['train_task_seeds']) == 64
assert r['reload_max_action_difference'] == 0
assert r['first_actor_max_delta'] > 0 and r['first_vlm_max_delta'] > 0
for line in (p / 'updates.jsonl').read_text().splitlines():
    row = json.loads(line)
    assert Counter(s['task'] for s in row['samples']) == {'select_fruit': 4, 'insert_flower': 4}
print(json.dumps(dict(status='two_task_smoke_passed', peak_cuda_bytes=r['peak_cuda_bytes'],
                      actor_delta=r['first_actor_max_delta'], checkpoint_action_difference=0)), flush=True)
PY
echo "$(date -Is) Starting one shared two-task SFT run"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --updates 1200 \
    --save-every 600 --skip-evaluation --output "$out/two-task-reference-1200" \
    > logs/pose-robustness-v2/two-task-reference-1200.log 2>&1
echo "$(date -Is) Training complete; fixed development comparison is next"
