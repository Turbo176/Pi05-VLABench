#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/.local/VLABench:$PWD/.local/rrt-algorithms"
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/.local/cache"
exec "/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python" -u scripts/select_fruit.py "$@"
