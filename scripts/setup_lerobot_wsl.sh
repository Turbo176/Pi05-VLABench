#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
linux_dir="/var/tmp/pi05-vlabench-$(id -u)"
export PIP_CACHE_DIR="$PWD/.local/pip-cache"
export TMPDIR="$linux_dir/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv "$linux_dir/lerobot"
python_bin="$linux_dir/lerobot/bin/python"
"$python_bin" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple pip==25.3
# CPU wheels from PyTorch's official index. Keep this independent of the simulator venv.
"$python_bin" -m pip install --disable-pip-version-check --timeout 25 --retries 2 \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu'
"$python_bin" -m pip install --disable-pip-version-check --timeout 25 --retries 2 \
  --index-url https://pypi.org/simple 'lerobot==0.4.4' 'h5py==3.11.0' 'numpy==2.2.6' \
  'torchcodec==0.5' 'opencv-python-headless==4.11.0.86' 'pyarrow==22.0.0' 'datasets==4.4.2' 'rerun-sdk==0.26.1' 'imageio-ffmpeg==0.5.1'
"$python_bin" -m pip check
# OpenPI's old reader expects datasets 3.x column indexing. Install its locked
# versions outside the v3 exporter environment and add them only for --openpi.
"$python_bin" -m pip install --disable-pip-version-check --timeout 25 --retries 2 \
  --index-url https://pypi.org/simple --no-deps --upgrade --target "$linux_dir/openpi-dataset-deps" \
  'datasets==3.6.0' 'pyarrow==20.0.0'
