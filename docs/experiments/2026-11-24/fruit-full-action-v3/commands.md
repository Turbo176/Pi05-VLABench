# 🚀 Reproduce the full-action fruit experiment

Use the prepared Linux environment in the [GPU guide](../../../single-gpu.md). Weights, gated tokenizer files, simulator assets and complete LeRobot datasets must already exist. The runtime uses MuJoCo 3.2.2, PyTorch 2.8.0+cu128, the pinned VLABench OpenPI fork and RLinf source. These commands reproduce the recorded settings with fresh output paths.

The server placed the repository's shared scripts under `code/scripts`. From a normal checkout, use `scripts` instead. The phase-specific source directories retain the exact training implementations; they are small source snapshots, not copies of upstream projects.

```bash
source .local/reference-env.sh
export PI05_REFERENCE_FORK="$reference_fork"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
report=docs/experiments/2026-11-24/fruit-full-action-v3
base=.local/checkpoints/vlabench-primitive-rlinf/model.safetensors
export PYTHONPATH="$PWD/$report/implementation/sft:$PWD/scripts:$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
export PI05_POSE_IMAGE_CACHE=64
```

## 📈 SFT

The executed run starts from the published VLABench-finetuned reference, not from a single-episode diagnostic or the earlier SFT180. Its verified split has 45 complete expert demonstrations and two expert corrections. Action statistics use only the 47 training episodes; state statistics remain unchanged.

```bash
parent=runs/repeated-fruit-full-action/sft
"$py" "$report/implementation/sft/train_reference_aligned_fruit.py" \
  --base "$base" --fork "$reference_fork" \
  --reference-start .local/checkpoints/vlabench-primitive-rlinf \
  --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 \
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras \
  --paired-episodes "$report/evidence/development/cases.jsonl" \
  --multitask --training-tasks select_fruit --full-action-expert \
  --fp32-master-optimizer --refit-action-quantiles --render-samples 0 \
  --gradient-accumulation 4 --micro-batch-size 2 \
  --actor-lora-rank 0 --vlm-lora-rank 0 --lr 0.00001 \
  --no-image-augmentation --updates 3000 --save-every 1000 \
  --skip-evaluation --action-loss all --shortest-euler \
  --inference-only-snapshots --output "$parent"
```

The final SFT saves full optimizer and FP32 master state; intermediate snapshots omit that state. Peak measured training CUDA allocation was about 18.9 GB in this runtime. Plan checkpoint storage separately: the final resumable checkpoint is about 5.8 GB.

## 🔍 Handoff and PPO

Switch to the RL implementation and keep exactly the same SFT checkpoint and normalization:

```bash
export PYTHONPATH="$PWD/$report/implementation/rl/source:$PWD/scripts:$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
"$py" -m pytest "$report/implementation/rl/tests" -q
```

The retained 33 CPU tests check FP32 master updates and resume, actor isolation during critic warmup, optimizer rollback, reward events and physical-control/chunk accounting. They are separate from full-model GPU validation. The recorded zero-update handoff also compares three real observation windows and all 16 autonomous development trajectories.

```bash
for arm in potential milestone; do
  if [[ "$arm" == potential ]]; then
    reward=discounted-terminal-potential-v2
    shaping=0.5
  else
    reward=grasp-lift-completion-v3
    shaping=0
  fi
  "$py" "$report/implementation/rl/source/train_fruit_ppo.py" \
    --base "$base" --fork "$reference_fork" --sft-run "$parent" \
    --output "runs/repeated-fruit-full-action/$arm" \
    --seeds 1302 1308 1303 1301 --reward-version "$reward" --shaping "$shaping" \
    --iterations 2 --critic-warmup-epochs 2 --lr 0.000001 \
    --critic-lr 0.0001 --target-kl 0.02 --minibatch 4 --max-steps 600 \
    --seed 372 --trace-controls --inference-only-checkpoints
done
```

In the recorded run, the potential arm saved both iterations. The milestone arm saved iteration one, then rejected iteration two because its whole-rollout selected-prefix diagnostic exceeded 0.02. That exit is retained as a guarded training rejection. There is no milestone iteration-two checkpoint. PPO exports support inference only and cannot resume optimizer state.

The exact executed invocations are also retained in `train.sh`. Development screening and model locking are in `evaluate.py`; their original server paths identify the recorded run. Use fresh output paths and the frozen manifests when adapting them to another workspace. Never use the reserved test set to choose another checkpoint.

## 📊 Paired evaluation

The manifest used for this experiment is `evidence/test/cases.jsonl`. Copy it to a fresh cases directory with the filename expected by the evaluator:

```bash
cases=runs/repeated-fruit-full-action/cases
mkdir -p "$cases"
cp "$report/evidence/test/cases.jsonl" "$cases/select_fruit-official.jsonl"
cp "$report/evidence/test/protocol.json" "$cases/protocol.json"
```

Read the chosen PPO path from `evidence/test/model-lock.json`. The reference must use its **original statistics** and reference-compatible configuration with `--stage base`; this stage restores no trained SFT parameters. The reference normalization SHA-256 is `bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f`. The new SFT and PPO use `2f89fa57106178d3451bd4582f6454085ada19686de382dec37b55ac7594a6da`.

The recorded evaluation read construction metadata from the historical SFT180 file, without restoring its trained tensors. A fresh checkout can reconstruct the same metadata from this report instead of downloading that unrelated checkpoint:

```bash
reference_config=runs/repeated-fruit-full-action/reference-config
"$py" - "$report" "$reference_config" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
import torch
report, destination = map(Path, sys.argv[1:])
configuration = json.loads((report / 'evidence/reference-configuration.json').read_text())
source = report / 'evidence/reference-norm_stats.json'
assert hashlib.sha256(source.read_bytes()).hexdigest() == configuration['metadata']['norm_stats_sha256']
destination.mkdir(parents=True, exist_ok=False)
shutil.copyfile(source, destination / 'norm_stats.json')
torch.save({'policy': {}, 'metadata': configuration['metadata']}, destination / 'sft.pt')
PY
```

This small file is valid only for `--stage base`; it contains no trained policy tensors. Its configuration-file hash differs from the historical container file, while the model construction settings, base weights and normalization stay the same.

```bash
chosen=runs/repeated-fruit-full-action/potential/ppo-2.pt
for stage in base sft ppo; do
  configuration="$parent"
  extra=()
  if [[ "$stage" == base ]]; then configuration="$reference_config"; fi
  if [[ "$stage" == ppo ]]; then extra=(--ppo-weights "$chosen"); fi
  "$py" "$report/implementation/rl/source/evaluate_pose_development.py" \
    --base "$base" --fork "$reference_fork" --run "$configuration" \
    --cases "$cases" --protocol "$cases/protocol.json" --suite ordinary \
    --tasks select_fruit --workers 2 --stage "$stage" "${extra[@]}" \
    --output "runs/repeated-fruit-full-action/test-$stage"
done
```

The completed test driver checks all model and normalization hashes before inference. The GPU run overlaps at most two model processes with two simulator workers each; the loop above uses the same per-model settings sequentially.

## ✅ Verify retained results

```bash
cd "$report"
python analyze.py --evidence evidence --output results --verify-only
```

This recomputes every outcome using the standard library and checks exported evidence/source hashes. Remove `--verify-only` to regenerate plots with NumPy and Matplotlib. Reloading and comparing the excluded model tensors is a separate PyTorch check, retained in `audit_saved_actor.py` and `audit_saved_available_actor.py`. Recorded videos execute fresh policy rollouts and must match the corresponding scored episode's complete action hash.
