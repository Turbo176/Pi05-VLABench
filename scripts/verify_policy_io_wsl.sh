#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 LEROBOT_SPLIT REPORT_JSON EPISODE_H5 [EPISODE_H5 ...]" >&2
    exit 2
fi
root=$1
report=$2
shift 2
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/samples.npz"; rmdir -- "$scratch"' EXIT
bash scripts/convert_lerobot_wsl.sh --openpi --policy-io-samples "$root" "$scratch/samples.npz"
export PYTHONPATH="$PWD/.local/VLABench:$PWD/.local/rrt-algorithms"
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/.local/cache"
"/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python" -u scripts/verify_policy_io.py simulator \
    "$scratch/samples.npz" "$report" "$@"
