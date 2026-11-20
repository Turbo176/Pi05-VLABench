#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$PWD/.local/huggingface"
if [[ "${1:-}" == --openpi ]]; then
    shift
    source_dir="$PWD/.local/lerobot-openpi"
    revision=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
    if [[ ! -d "$source_dir/.git" ]] || [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
        echo "Check out the pinned LeRobot revision at $source_dir before using --openpi." >&2
        exit 1
    fi
    export PYTHONPATH="$source_dir:/var/tmp/pi05-vlabench-$(id -u)/openpi-dataset-deps"
    if [[ "${1:-}" == --policy-io-samples ]]; then
        shift
        exec "/var/tmp/pi05-vlabench-$(id -u)/lerobot/bin/python" -u scripts/verify_policy_io.py dataset "$@"
    fi
    set -- --format v2.1 "$@"
fi
exec "/var/tmp/pi05-vlabench-$(id -u)/lerobot/bin/python" -u scripts/convert_lerobot.py "$@"
