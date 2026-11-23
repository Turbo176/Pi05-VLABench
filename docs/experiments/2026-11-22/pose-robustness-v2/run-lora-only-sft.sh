#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
while [[ ! -s runs/pose-robustness-v2/lora-only-smoke/summary.json ]]; do
 if ! kill -0 21062 2>/dev/null; then echo 'Frozen-head smoke failed.' >&2; exit 1; fi
 sleep 5
done
"$py" - <<'PY'
import pathlib,json
info=json.loads(pathlib.Path('runs/pose-robustness-v2/lora-only-smoke/summary.json').read_text())
assert info['frozen_action_heads_unchanged'] and not info['train_action_heads']
assert info['reload_max_action_difference']==0 and info['first_actor_max_delta']>0
print(json.dumps(dict(smoke='passed',frozen_action_heads_unchanged=True)),flush=True)
PY
echo "$(date -Is) Starting 100-update LoRA-only follow-up"
"$py" -u code/scripts/train_reference_aligned_fruit.py --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" --reference-start .local/checkpoints/vlabench-primitive-rlinf --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl --render-samples 0 --multitask --gradient-accumulation 6 --actor-lora-rank 16 --vlm-lora-rank 8 --freeze-action-heads --lr 0.0001 --updates 100 --save-every 50 --skip-evaluation --output runs/pose-robustness-v2/lora-only-100 > logs/pose-robustness-v2/lora-only-100.log 2>&1
echo "$(date -Is) Frozen-head follow-up training complete"
