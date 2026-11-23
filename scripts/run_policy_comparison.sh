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

"$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --action-space "${PI05_ACTION_SPACE:-absolute}" --checkpoint "$sft" \
    --stage sft --suite ppo_development --output "$result/fruit-reward-development.jsonl"
"$python" - "$result/fruit-reward-development.jsonl" <<'PY'
from pathlib import Path
import json
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
expected = {('select_fruit', seed) for seed in range(1201, 1205)}
if len(rows) != len(expected) or {(row['task'], row['seed']) for row in rows} != expected:
    raise RuntimeError('Expected the four designated fruit development episodes')
successes = sum(row['success'] for row in rows)
print(f'Fruit development successes: {successes}/{len(rows)}', flush=True)
if not successes:
    raise SystemExit('PPO paused: the SFT policy did not succeed on any fruit development start')
PY
for stage in sft handoff; do
    "$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --action-space "${PI05_ACTION_SPACE:-absolute}" --checkpoint "$sft" \
        --stage "$stage" --suite development --output "$result/$stage-development.jsonl"
done
"$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --action-space "${PI05_ACTION_SPACE:-absolute}" --checkpoint "$sft" \
    --stage handoff --suite ppo_development --output "$result/handoff-fruit-reward-development.jsonl"
"$python" - "$result" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
sft, handoff = [[json.loads(line) for line in (path / f'{name}-development.jsonl').read_text().splitlines()]
                for name in ('sft', 'handoff')]
keys = ('task', 'seed', 'success', 'steps', 'stop_reason', 'initial_state_sha256', 'initial_rgb_sha256',
        'action_chunks_sha256', 'stats_sha256', 'checkpoint_sha256')
fruit_sft, fruit_handoff = [[json.loads(line) for line in (path / f'{name}-reward-development.jsonl').read_text().splitlines()]
                            for name in ('fruit', 'handoff-fruit')]
for before, after, expected in ((sft, handoff, 3), (fruit_sft, fruit_handoff, 4)):
    if len(before) != expected or len(after) != expected or any(a[k] != b[k] for a, b in zip(before, after) for k in keys):
        raise RuntimeError('SFT and its RLinf handoff produced different evaluation traces')
(path / 'handoff-verified.json').write_text(json.dumps({'episodes': 7, 'verified_keys': keys}, indent=2) + '\n')
PY
# A successful development evaluation does not guarantee sparse rewards on PPO training starts.
# If a complete zero-reward batch stops the sparse run, keep its logs and run the shaped variant.
ppo="$result/ppo"
if ! "$python" -u "$scripts/ppo_pi05.py" --sft "$sft" --base "$base" --stats "$stats" \
    --output "$ppo" --seeds 1211 1212 1213 1214 --iterations 1 --shaping 0 2>&1 | tee "$result/ppo.log"; then
    if ! grep -q 'No success reward in' "$result/ppo.log"; then
        exit 1
    fi
    ppo="$result/ppo-shaped"
    "$python" -u "$scripts/ppo_pi05.py" --sft "$sft" --base "$base" --stats "$stats" \
        --output "$ppo" --seeds 1211 1212 1213 1214 --iterations 1 --shaping 0.25 2>&1 | tee "$result/ppo-shaped.log"
fi
exec bash "$scripts/run_policy_evaluations.sh" "$python" "$base" "$stats" "$sft" "$ppo/ppo-1.pt" "$result"
