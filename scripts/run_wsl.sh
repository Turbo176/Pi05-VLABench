#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source_root=${PI05_SIM_SOURCE_ROOT:-"$PWD/.local/VLABench"}
export PYTHONPATH="$source_root:$PWD/.local/rrt-algorithms"
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/.local/cache"
runtime="/var/tmp/pi05-vlabench-$(id -u)"
script=scripts/select_fruit.py
if [[ ${1:-} == evaluate ]]; then
    script=scripts/evaluate_policy.py
    shift
fi
# MuJoCo and Mesa can exhaust WSL memory when two simulator processes overlap.
exec flock --exclusive --no-fork "$runtime/simulator.lock" \
    "$runtime/venv/bin/python" -X faulthandler -u "$script" "$@"
