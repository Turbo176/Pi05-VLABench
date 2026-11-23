#!/usr/bin/env bash
# Run a pinned pose-policy comparison using prepared reset manifests.
set -euo pipefail
if (( $# < 7 )); then
    echo 'Usage: run_reference_pose_suite.sh PYTHON OPENPI_FORK BASE_SAFETENSORS SFT_RUN PPO_WEIGHTS CASES OUTPUT [SEED ...]' >&2
    exit 2
fi
python_bin=$1
fork=$2
base=$3
sft_run=$4
ppo_weights=$5
cases=$6
output=$7
shift 7
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stage_args=()
seed_args=()
protocol_args=()
if [[ -n ${POSE_EVAL_PROTOCOL:-} ]]; then protocol_args=(--protocol "$POSE_EVAL_PROTOCOL"); fi
case "${POSE_EVAL_STAGE:-all}" in
    all) stages=(base sft ppo) ;;
    base|sft|ppo) stages=("$POSE_EVAL_STAGE") ;;
    *) echo 'POSE_EVAL_STAGE must be base, sft, ppo, or all' >&2; exit 2 ;;
esac
for seed in "$@"; do seed_args+=(--eval-seed "$seed"); done
export JAX_PLATFORMS=cpu
mkdir -p "$output"

run_group() {
    local stage=$1 suite=$2 task=$3 mode=$4 case_file=$5
    local dest="$output/$stage-$suite-$task-$mode"
    echo "Evaluating $stage $suite $task $mode"
    "$python_bin" -u "$script_dir/train_reference_aligned_fruit.py" \
        --base "$base" --fork "$fork" --load-run "$sft_run" \
        --eval-stage "$stage" --eval-task "$task" --prompt-mode "$mode" \
        --paired-episodes "$cases/$case_file.jsonl" --output "$dest" --resume-evaluation \
        "${stage_args[@]}" "${seed_args[@]}"
}

for stage in "${stages[@]}"; do
    stage_args=()
    if [[ $stage == ppo ]]; then stage_args=(--ppo-weights "$ppo_weights"); fi
    for task in select_fruit insert_flower take_out_cool_drink; do
        run_group "$stage" ordinary "$task" official "$task-official"
    done
    cat "$output/$stage-ordinary-"*/episodes.jsonl > "$output/$stage-ordinary.jsonl"
    if (( $# == 0 )); then
        run_group "$stage" fruit_language select_fruit official select_fruit-language-official
        for mode in paraphrase alternate_target; do
            run_group "$stage" fruit_language select_fruit "$mode" "select_fruit-$mode"
        done
        cat "$output/$stage-fruit_language-"*/episodes.jsonl > "$output/$stage-fruit_language.jsonl"
    fi
done

if [[ ${POSE_EVAL_STAGE:-all} != all ]]; then exit 0; fi

if (( $# == 0 )); then
    for suite in ordinary fruit_language; do
        "$python_bin" "$script_dir/summarize_pi05.py" --pose --suite "$suite" "${protocol_args[@]}" \
            "$output/base-$suite.jsonl" "$output/sft-$suite.jsonl" "$output/ppo-$suite.jsonl" \
            --output "$output/$suite-summary.json"
    done
else
    "$python_bin" "$script_dir/summarize_pi05.py" --pose \
        "$output/base-ordinary.jsonl" "$output/sft-ordinary.jsonl" "$output/ppo-ordinary.jsonl" \
        --output "$output/ordinary-subset-summary.json"
fi
