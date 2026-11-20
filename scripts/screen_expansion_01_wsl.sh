#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# == 0 ]]; then
    batch=runs/stage2-expansion-01
    tasks=(select_fruit insert_flower take_out_cool_drink)
    seeds=(53 54 55)
else
    if [[ $# -lt 3 || ! $1 =~ ^runs/stage2-[a-z0-9-]+$ || ! $2 =~ ^(select_fruit|insert_flower|take_out_cool_drink|all)$ ]]; then
        echo "Usage: $0 [runs/stage2-BATCH TASK|all SEED ...]" >&2
        exit 2
    fi
    batch=$1
    tasks=("$2")
    if [[ $2 == all ]]; then
        tasks=(select_fruit insert_flower take_out_cool_drink)
    fi
    shift 2
    seeds=("$@")
    for seed in "${seeds[@]}"; do
        if [[ ! $seed =~ ^(0|[1-9][0-9]*)$ ]]; then
            echo "Invalid seed: $seed" >&2
            exit 2
        fi
    done
fi
mkdir -p -- "$batch"
for seed in "${seeds[@]}"; do
    for task in "${tasks[@]}"; do
        destination="$batch/${task}-${seed}-screen"
        if [[ -s "$destination.status" ]]; then
            printf 'Already recorded: %s seed %s\n' "$task" "$seed"
            continue
        fi
        if [[ -e "$destination" || -e "$destination.log" ]]; then
            echo "Incomplete attempt at $destination. Keep its artifacts and use a new batch path for retry." >&2
            exit 2
        fi
        printf 'Screening %s seed %s\n' "$task" "$seed"
        status=0
        timeout -k 15s 240s bash scripts/run_wsl.sh collect --task "$task" --seed "$seed" --budget 600 --screen --output "$destination" > "$destination.log" 2>&1 || status=$?
        printf '%s\n' "$status" > "$destination.status.tmp"
        mv -- "$destination.status.tmp" "$destination.status"
        printf '%s seed %s exit %s\n' "$task" "$seed" "$status"
        if [[ -f "$destination/expert.json" ]]; then
            grep -E '"instruction"|"success"|"steps"|"wall_seconds"' "$destination/expert.json"
        else
            tail -8 "$destination.log"
        fi
    done
done
