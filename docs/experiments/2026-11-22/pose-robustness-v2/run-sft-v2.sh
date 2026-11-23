#!/usr/bin/env bash
# Executed from the AutoDL project root in the existing, pinned environment.
set -euo pipefail
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
dataset=.local/reference-aligned/lerobot-tritask-msaa0-v2
out=runs/pose-robustness-v2
while [[ ! -s $dataset/verification.json ]]; do
    if [[ -z ${CONVERSION_PID:-} ]] || ! kill -0 "$CONVERSION_PID" 2>/dev/null; then
        echo 'Conversion stopped before its verification report was written.' >&2
        exit 1
    fi
    sleep 10
done
common=(--base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors
        --fork "$reference_fork" --reference-start .local/checkpoints/vlabench-primitive-rlinf
        --lerobot "$dataset" --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl
        --multitask --render-samples 0 --gradient-accumulation 6
        --actor-lora-rank 16 --vlm-lora-rank 8 --lr 0.000003)
echo "$(date -Is) Starting balanced multitask smoke check"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --smoke --save-every 1 \
    --output "$out/smoke-multitask" > logs/pose-robustness-v2/smoke-multitask.log 2>&1
echo "$(date -Is) Checking optimizer and RNG resume from update one"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --smoke \
    --resume-run "$out/smoke-multitask/checkpoint-0001" \
    --output "$out/smoke-multitask-resumed" > logs/pose-robustness-v2/smoke-multitask-resumed.log 2>&1
"$py" - <<'PY'
from collections import Counter
import json
from pathlib import Path
import torch
root = Path('runs/pose-robustness-v2')
full = torch.load(root / 'smoke-multitask/sft.pt', weights_only=True, map_location='cpu')
resumed = torch.load(root / 'smoke-multitask-resumed/sft.pt', weights_only=True, map_location='cpu')
assert full['policy'].keys() == resumed['policy'].keys()
assert all(torch.equal(value, resumed['policy'][name]) for name, value in full['policy'].items())
for line in (root / 'smoke-multitask/updates.jsonl').read_text().splitlines():
    row = json.loads(line)
    assert Counter(sample['task'] for sample in row['samples']) == {
        'select_fruit': 2, 'insert_flower': 2, 'take_out_cool_drink': 2}
result = dict(updates=3, resumed_from_update=1, identical_policy_tensors=len(full['policy']),
              max_parameter_difference=0.0, balanced_windows_per_update=6,
              checkpoint_reload_max_action_difference=full['metadata'].get('reload_max_action_difference'))
# The final restore check is written to summary.json after the checkpoint itself.
result['checkpoint_reload_max_action_difference'] = json.loads(
    (root / 'smoke-multitask/summary.json').read_text())['reload_max_action_difference']
(root / 'resume-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
PY
echo "$(date -Is) Starting 300-update multitask SFT"
"$py" -u code/scripts/train_reference_aligned_fruit.py "${common[@]}" --updates 300 \
    --save-every 100 --skip-evaluation --output "$out/sft-300" \
    > logs/pose-robustness-v2/sft-300.log 2>&1
echo "$(date -Is) SFT completed; development evaluation is a separate step"
