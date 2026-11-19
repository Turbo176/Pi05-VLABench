#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PIP_CACHE_DIR="$ROOT/.local/pip-cache"
export TMPDIR="$ROOT/.local/tmp"
mkdir -p "$PIP_CACHE_DIR"
linux_dir="/var/tmp/pi05-vlabench-$(id -u)"
mkdir -m 700 -p "$linux_dir/tmp"
export TMPDIR="$linux_dir/tmp"
python3 -m venv "$linux_dir/venv"
"$linux_dir/venv/bin/python" -m pip install --disable-pip-version-check --retries 2 --timeout 25 numpy==1.25.0 scipy==1.14.0 mujoco==3.2.2 dm_control==1.0.22 gym==0.26.2 open3d-cpu==0.18.0 scikit-learn==1.5.2 opencv-python-headless==4.10.0.84 colorlog==6.9.0 imageio-ffmpeg==0.5.1 h5py==3.11.0 igraph==0.11.8 openai==1.109.1 'rrt-algorithms @ git+https://github.com/motion-planning/rrt-algorithms.git@14e47963780affe3ff3bb9fc33a4b14e1a733e85'
