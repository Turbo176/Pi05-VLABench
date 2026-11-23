#!/usr/bin/env bash
# Resume matched evaluations from any completed prefix.
set -euo pipefail
if [[ $# != 6 ]]; then
    echo 'Usage: run_policy_evaluations.sh PYTHON BASE_SAFETENSORS NORM_STATS SFT_CHECKPOINT PPO_CHECKPOINT OUTPUT_DIR' >&2
    exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=$1
base=$2
stats=$3
sft=$4
ppo=$5
result=$6
mkdir -p -- "$result"

run_stage() {
    local suite=$1 stage=$2
    shift 2
    local -a weights=()
    case $stage in
        sft) weights=(--checkpoint "$sft") ;;
        ppo) weights=(--checkpoint "$ppo") ;;
    esac
    "$python" -u "$scripts/evaluate_pi05.py" --base "$base" --stats "$stats" --action-space "${PI05_ACTION_SPACE:-absolute}" "${weights[@]}" \
        --stage "$stage" --suite "$suite" --workers "${PI05_EVAL_WORKERS:-1}" \
        --resume --video-dir "$result/videos" \
        --output "$result/$suite-$stage.jsonl" "$@"
}

if [[ ! -f "$result/ordinary-first-seed-summary.json" ]]; then
    for stage in base sft ppo; do
        run_stage ordinary "$stage" --max-seeds-per-task 1
    done
    "$python" "$scripts/summarize_pi05.py" "$result/ordinary-base.jsonl" "$result/ordinary-sft.jsonl" \
        "$result/ordinary-ppo.jsonl" --output "$result/ordinary-first-seed-summary.json"
fi

for suite in ordinary fruit_language; do
    if [[ -f "$result/$suite-summary.json" ]]; then
        continue
    fi
    if [[ ${PI05_PARALLEL_EVAL:-0} == 1 ]]; then
        pids=()
        for stage in base sft ppo; do
            run_stage "$suite" "$stage" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=$?
        done
        if (( status != 0 )); then
            exit "$status"
        fi
    else
        for stage in base sft ppo; do
            run_stage "$suite" "$stage"
        done
    fi
    "$python" "$scripts/summarize_pi05.py" "$result/$suite-base.jsonl" "$result/$suite-sft.jsonl" \
        "$result/$suite-ppo.jsonl" --suite "$suite" --output "$result/$suite-summary.json"
done
