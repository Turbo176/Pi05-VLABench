#!/usr/bin/env bash
# Run the comparison after the single-GPU smoke test and a completed SFT run.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo 'Usage: run_policy_comparison.sh PYTHON BASE_SAFETENSORS NORM_STATS SFT_CHECKPOINT NEW_OUTPUT_DIR' >&2
    exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=$1
base=$2
stats=$3
sft=$4
result=$5
mkdir -- "$result"

for stage in sft handoff; do
    "$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --checkpoint "$sft" \
        --stage "$stage" --suite development --output "$result/$stage-development.jsonl"
done
"$python" - "$result" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
sft, handoff = [[json.loads(line) for line in (path / f'{name}-development.jsonl').read_text().splitlines()]
                for name in ('sft', 'handoff')]
keys = ('task', 'seed', 'success', 'steps', 'stop_reason', 'initial_state_sha256', 'initial_rgb_sha256',
        'action_chunks_sha256', 'stats_sha256', 'checkpoint_sha256')
if len(sft) != 3 or len(handoff) != 3 or any(a[k] != b[k] for a, b in zip(sft, handoff) for k in keys):
    raise RuntimeError('SFT and its RLinf handoff produced different evaluation traces')
(path / 'handoff-verified.json').write_text(json.dumps({'episodes': 3, 'verified_keys': keys}, indent=2) + '\n')
PY
"$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --checkpoint "$sft" \
    --stage sft --suite ppo_development --output "$result/fruit-reward-development.jsonl"
shaping=$("$python" - "$result/fruit-reward-development.jsonl" <<'PY'
from pathlib import Path
import json
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
if len(rows) != 4:
    raise RuntimeError('Expected four fruit development episodes')
print(0 if any(row['success'] for row in rows) else 0.25)
PY
)
# A successful development evaluation does not guarantee sparse rewards on PPO training starts.
# If a complete zero-reward batch stops the sparse run, keep its logs and run the shaped variant.
ppo="$result/ppo"
if ! "$python" -u "$scripts/ppo_pi05.py" --sft "$sft" --base "$base" --stats "$stats" \
    --output "$ppo" --seeds 1211 1212 1213 1214 --iterations 1 --shaping "$shaping" 2>&1 | tee "$result/ppo.log"; then
    if [[ $shaping != 0 ]] || ! grep -q 'No success reward in' "$result/ppo.log"; then
        exit 1
    fi
    ppo="$result/ppo-shaped"
    "$python" -u "$scripts/ppo_pi05.py" --sft "$sft" --base "$base" --stats "$stats" \
        --output "$ppo" --seeds 1211 1212 1213 1214 --iterations 1 --shaping 0.25 2>&1 | tee "$result/ppo-shaped.log"
fi
exec bash "$scripts/run_policy_evaluations.sh" "$python" "$base" "$stats" "$sft" "$ppo/ppo-1.pt" "$result"
