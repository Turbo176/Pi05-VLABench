#!/usr/bin/env bash
# Call only after development-only SFT selection, zero-update handoff and PPO.
set -euo pipefail
if (( $# != 3 )); then
    echo 'Usage: run-two-task-final.sh SFT_RUN PPO_WEIGHTS FRESH_OUTPUT' >&2
    exit 2
fi
sft=$1
ppo=$2
out=$3
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
if [[ -e $out ]]; then echo 'Use a fresh output folder.' >&2; exit 1; fi
mkdir -p "$out"
run_stage() {
    local stage=$1
    local extra=()
    if [[ $stage == ppo ]]; then extra=(--ppo-weights "$ppo"); fi
    for suite in ordinary fruit_language; do
        "$py" -u code/scripts/evaluate_pose_development.py \
            --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
            --fork "$reference_fork" --run "$sft" --stage "$stage" "${extra[@]}" \
            --cases .local/reference-aligned/msaa0-final-cases-v2 \
            --protocol code/configs/evaluation-two-task-v2.json --suite "$suite" --workers 3 \
            --output "$out/$stage-$suite" > "$out/$stage-$suite.log" 2>&1
    done
}
pids=()
for stage in base sft ppo; do run_stage "$stage" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then
    echo 'A stage failed. Retain every partial record; inspect before resuming.' >&2
    exit 1
fi
for suite in ordinary fruit_language; do
    "$py" code/scripts/summarize_pi05.py --pose --suite "$suite" \
        --protocol code/configs/evaluation-two-task-v2.json \
        "$out/base-$suite/episodes.jsonl" "$out/sft-$suite/episodes.jsonl" "$out/ppo-$suite/episodes.jsonl" \
        --output "$out/$suite-summary.json"
done
echo "$(date -Is) All three stages and both final suites completed"
