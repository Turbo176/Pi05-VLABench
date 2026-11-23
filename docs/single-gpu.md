# GPU training and evaluation

The completed experiment uses the [pinned VLABench OpenPI fork and RLinf](upstream.md), a shared fruit/flower SFT checkpoint, and fruit PPO from that exact parent. It uses the pose interface in [system design](system-design.md). The older joint-space training route is retired.

## Prepared environment

The executed AutoDL workspace was `/root/autodl-tmp/Pi05-VLABench`. Repository scripts/configs were uploaded into `code/scripts` and `code/configs`. The ignored `.local/reference-env.sh` selected the Python runtime, pinned checkouts, simulator assets, and `reference_fork`; it contains machine-specific paths and must be prepared for a new machine.

The [recorded environment](experiments/2026-11-22/fruit-completion-v1/environment.json) used Python 3.12.3, PyTorch 2.8.0+cu128, MuJoCo 3.2.2, dm-control 1.0.22, NumPy 1.26.4 and PyArrow 18.1.0. Model weights, gated tokenizer files and complete datasets must be obtained separately under their upstream terms. These commands assume that environment and those inputs already exist; they are not an installer.

```bash
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
```

## Shared SFT recipe

The actual SFT invocation used the following inputs and arguments. Use a fresh output path for a new run. An earlier two-update, shared-data [GPU smoke check](experiments/2026-11-22/pose-robustness-v2/two-task-smoke.json) verified backward updates and exact checkpoint restoration with a different VLM-adapter recipe; it is not a performance result for this final recipe.

```bash
"$py" code/scripts/train_reference_aligned_fruit.py \
  --base .local/checkpoints/vlabench-primitive-rlinf/model.safetensors \
  --fork "$reference_fork" \
  --reference-start .local/checkpoints/vlabench-primitive-rlinf \
  --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 \
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras \
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-105-134 \
  --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl \
  --multitask --training-tasks select_fruit insert_flower --render-samples 0 \
  --gradient-accumulation 2 --micro-batch-size 4 \
  --actor-lora-rank 16 --vlm-lora-rank 0 --lr 0.000001 --no-image-augmentation \
  --updates 90 --skip-evaluation \
  --output runs/pose-robustness-v2/two-task-conservative-90
```

This trains action-expert rank-16 LoRA and action heads while freezing vision/language. The [saved summary](experiments/2026-11-22/fruit-completion-v1/sft-parent/summary.json) records the exact episode pairs, normalization, weight hashes, update count and costs. The [training log](experiments/2026-11-22/fruit-completion-v1/sft-parent/updates.jsonl) contains all 90 updates.

## Zero-update handoff and PPO

These are the executed handoff and PPO commands. The handoff compares seeded predictions, shared decoding, recomputed log probabilities and one closed-loop rollout before optimization.

```bash
parent=runs/pose-robustness-v2/two-task-conservative-90
root=runs/pose-robustness-v2/fruit-completion-v1
base=.local/checkpoints/vlabench-primitive-rlinf/model.safetensors
"$py" code/scripts/check_reference_rl_handoff.py \
  --base "$base" --fork "$reference_fork" --run "$parent" \
  --episode .local/reference-aligned/msaa0-expert-replays-full/train-select_fruit-42.h5 \
  --paired-episodes runs/pose-robustness-v2/two-task-conservative-development/cases/select_fruit-1201-official/episodes.jsonl \
  --eval-seed 1201 --output "$root/handoff.json"
"$py" code/scripts/ppo_reference_aligned_fruit.py \
  --base "$base" --fork "$reference_fork" --sft-run "$parent" \
  --output "$root/ppo" --seeds 1301 1302 --iterations 1 \
  --max-steps 600 --shaping 0.5 --seed 372 --lr 0.000002 --critic-lr 0.0001
```

The handoff output and training directory must be fresh. The [frozen PPO configuration](../configs/ppo-fruit-completion-v1.json) records trainable parameters, reward and execution budget. The [saved-weight audit](experiments/2026-11-22/fruit-completion-v1/saved-checkpoint-update-audit.json) confirms an actor update separately from the critic.

## Paired evaluation

The same evaluator handles the published reference, SFT and PPO. This loop matches the evaluated policy stages and suites; copy the four retained JSONL manifests from the [experiment's cases directory](experiments/2026-11-22/fruit-completion-v1/cases/select_fruit-official.jsonl) into `$root/cases` for a fresh run. `--workers 3` was used in the completed experiment. Output directories must be fresh.

```bash
for suite in ordinary fruit_language; do
  for stage in base sft ppo; do
    extra=()
    if [[ "$stage" == ppo ]]; then
      extra=(--ppo-weights "$root/ppo/ppo-1.pt")
    fi
    "$py" code/scripts/evaluate_pose_development.py \
      --base "$base" --fork "$reference_fork" --run "$parent" \
      --cases "$root/cases" --protocol code/configs/evaluation-fruit-completion-v1.json \
      --tasks select_fruit --suite "$suite" --stage "$stage" --workers 3 "${extra[@]}" \
      --output "$root/$suite-$stage"
  done
done
```

The recorded reference values passed a separate [reuse audit](experiments/2026-11-22/fruit-completion-v1/reference-reuse.json): reset manifests matched and four fresh repeats reproduced complete action hashes. The first ten numerical seeds form the final subset; the [original twenty reference records](experiments/2026-11-22/fruit-completion-v1/reference-original-20.jsonl) remain available. A new experiment can run the reference stage directly.

The ordinary outcomes are 4/10, 5/10 and 5/10. PPO did not improve aggregate success. See the [full results](experiments/2026-11-22/fruit-completion-v1/README.md), including language regressions. This repository cleanup reran CPU verification and syntax checks; it did not repeat GPU training or simulation.
