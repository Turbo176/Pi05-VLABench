#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/Pi05-VLABench
r=runs/fruit-full-ppo-reward-compare-20261124
trap 'rc=$?; printf "%s\n" "$rc" > "$r/train.exit-code"' EXIT
source .local/reference-env.sh
export PI05_REFERENCE_FORK="$reference_fork"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/$r/source:$PWD/code/scripts:$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
while ! test -f runs/fruit-train47-handoff-20261124/development-handoff.json; do
 if test -f runs/fruit-train47-handoff-20261124/follow-v2.exit-code; then
  test "$(cat runs/fruit-train47-handoff-20261124/follow-v2.exit-code)" = 0
 fi
 sleep 3
done
while test "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 28672; do sleep 3; done
"$py" "$r/prelaunch.py"
for arm in potential milestone; do
 if test "$arm" = potential; then reward=discounted-terminal-potential-v2; shaping=.5; else reward=grasp-lift-completion-v3; shaping=0; fi
 set +e
 "$py" "$r/source/train_fruit_ppo.py" --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors --fork "$reference_fork" --sft-run runs/fruit-full-train47-20261124/storage/sft --output "$r/storage/$arm" --seeds 1302 1308 1303 1301 --reward-version "$reward" --shaping "$shaping" --iterations 2 --critic-warmup-epochs 2 --lr .000001 --critic-lr .0001 --target-kl .02 --minibatch 4 --max-steps 600 --seed 372 --trace-controls --inference-only-checkpoints > "$r/$arm-train.log" 2>&1
 result=$?
 printf '%s\n' "$result" > "$r/$arm-train.exit-code"
 set -e
 test "$result" = 0
done
