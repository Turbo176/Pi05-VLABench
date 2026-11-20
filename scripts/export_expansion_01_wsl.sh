#!/usr/bin/env bash
# Rebuild a v2.1 training split from the Stage 2 pilot and the first fixed-seed expansion.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

batch=runs/stage2-expansion-01
output=data/stage2-expansion-01-v21
if [[ -e "$output" ]]; then
    echo "Existing export at $output" >&2
    exit 2
fi

training=(
    --episode train=runs/select_fruit-42/episode.h5
    --episode train=runs/stage2/insert_flower-42-retry/episode.h5
    --episode train=runs/stage2/take_out_cool_drink-43-screen-success/episode.h5
)

render_success() {
    local screening=$1 episode=$2
    if [[ ! -f "$episode/render_screen.json" || ! -f "$episode/episode.h5" ]]; then
        bash scripts/run_wsl.sh render-screen "$screening" --output "$episode"
    fi
    training+=(--episode "train=$episode/episode.h5")
}

for screening_status in "$batch"/{select_fruit,insert_flower,take_out_cool_drink}-{53,54,55}-screen.status; do
    if [[ ! -f "$screening_status" ]]; then
        echo "Awaiting screening outcome at $screening_status" >&2
        exit 2
    fi
done

render_success runs/stage2/followup-take_out_cool_drink-52/screening.h5 "$batch/take_out_cool_drink-52-rgb"
for seed in 53 54 55; do
    for task in select_fruit insert_flower take_out_cool_drink; do
        screening="$batch/${task}-${seed}-screen"
        if [[ $(< "$screening.status") == 0 ]]; then
            render_success "$screening/screening.h5" "$batch/${task}-${seed}-rgb"
        fi
    done
done

bash scripts/convert_lerobot_wsl.sh --openpi --action-horizon 50 "${training[@]}" --output "$output"
