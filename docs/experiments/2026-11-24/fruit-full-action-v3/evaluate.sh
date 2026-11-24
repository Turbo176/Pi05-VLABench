#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/Pi05-VLABench
r=runs/fruit-full-ppo-reward-compare-20261124
trap 'rc=$?; printf "%s\n" "$rc" > "$r/evaluate.exit-code"' EXIT
source .local/reference-env.sh
export PI05_REFERENCE_FORK="$reference_fork"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/$r/source:$PWD/code/scripts:$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
"$py" "$r/evaluate.py"
