# GPU training and evaluation

The current fruit full-action experiment uses 3,000 SFT updates and PPO from the exact same checkpoint. Use its [verified commands](experiments/2026-11-24/fruit-full-action-v3/commands.md) and [phase-specific sources](experiments/2026-11-24/fruit-full-action-v3/implementation/rl/source/train_fruit_ppo.py). They include the 33-test CPU check, FP32 master optimizer, current reward variants and reference normalization.

The recipes below document the earlier shared fruit/flower SFT90/SFT180 experiments; their results remain separate. They use the pose interface in [system design](system-design.md).

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

## Fruit reward follow-up

The two reward-smoke commands below ran on November 24, 2026 after an instance clone. Both completed full-model critic warmup, eight actor optimizer steps, checkpoint saving and paired development evaluation. Development scores were 10/16 for legacy and 5/16 for corrected, versus 8/16 for their SFT90 parent. The separate eight-start expansion used SFT180, selected at 9/16 on development, and the preferred legacy reward. Its first iteration saved a valid checkpoint; the second stopped during simulator rollout before another actor update. The forty-scene test is complete: reference 20/40, SFT180 17/40 and PPO1 16/40. The historical experiment above retains its original entry point and recipe.

The follow-up entry point is `code/scripts/train_fruit_ppo.py`. Upload that file alongside the existing scripts, preserving the frozen source snapshots under the historical run. The smoke comparison uses SFT90; the expansion uses the selected SFT180, with the same frozen vision/language features. The cloned environment ran without new weights, simulator assets or Python packages. [Expanded-run commands and report](experiments/2026-11-24/fruit-recovery-v2/README.md)

First run the CPU tests from a source checkout with NumPy and PyTorch installed:

```bash
python -m unittest discover -s tests -v
python scripts/review_fruit_experiment.py
```

On a GPU-less AutoDL instance, importing simulator helpers with the GPU `egl` backend can fail with `Cannot initialize a headless EGL display`. For import checks only, set `MUJOCO_GL=disable` in that shell. This disables rendering; it does not validate images, video or a visual-policy rollout. Start a fresh shell with the normal `egl` environment for GPU execution.

On the server, source the environment as above. Verify available disk space and the actual GPU, then rerun `check_reference_rl_handoff.py` from the previous section with a **fresh** output path. After an instance clone, wait for both disk copies to finish before writing to the destination. The existing development cases remain debugging inputs. Inspect SFT control traces on those cases before treating reward as the sole cause of failure.

Before increasing PPO compute, check the SFT/data baseline. The recorded 90 updates sampled 720 observation/action windows: 360 fruit and 360 flower. All 47 fruit episodes were sampled, but each supplied only 3–13 windows (mean 7.66). The before/after validation action errors did not improve. This supports a bounded training/coverage investigation; it does not prove that more updates or demonstrations will increase success. Extend the same recipe with intermediate checkpoints and select on validation/development results. Any changed task mixture, extra data or normalization needs a separately recorded experiment.

The old ten-scene test contains five pear, four banana and one apple targets, with no orange target. A fresh forty-scene manifest was reserved, ten naturally reset targets per fruit, without policy calls or environment controls during scene selection. Its SHA-256 is `5156c7f5bdd32e3405f7a7747969b0f2faa813c30e3f498ea0f70ffead21f36d`. Reference and SFT180 tests ran in parallel, with their outcomes kept unopened until PPO selection was frozen. All three policies then completed the same forty-scene test. Those outcomes are now inspected and cannot serve as an untouched final test for subsequent tuning. Multiple test scenes do not substitute for independent PPO training seeds.

For a bounded reward-only comparison, use both rewards with the same new optimizer, warmup, starts and maximum interaction budget:

```bash
parent=runs/pose-robustness-v2/two-task-conservative-90
base=.local/checkpoints/vlabench-primitive-rlinf/model.safetensors
next=runs/fruit-potential-v2
for reward in legacy-distance-v1 discounted-terminal-potential-v2; do
  "$py" code/scripts/train_fruit_ppo.py \
    --base "$base" --fork "$reference_fork" --sft-run "$parent" \
    --output "$next/$reward" --seeds 1301 1302 --iterations 1 \
    --critic-warmup-epochs 2 --max-steps 600 --minibatch 8 \
    --shaping 0.5 --seed 372 --lr 0.000002 --critic-lr 0.0001 \
    --target-kl 0.02 --reward-version "$reward" --trace-controls
done
```

Each arm has one warmup collection round and one PPO collection round, at most **2,400 controls** in total. Success may end episodes early, so actual controls can differ and must be reported. Two passes over the warmup data add optimizer steps, not environment interactions. Both arms restart from SFT90, with fresh actor/critic optimizers and the same random seeds. This is a short stability check, not a convergence experiment. Comparing either arm with the old run also changes warmup and update guards; it is not a reward-only comparison with the old checkpoint.

`run.json` records arguments and source/input hashes. `rollouts.jsonl` and `transitions.jsonl` separate official success from shaping rewards. `warmup.json` records actor equality and critic changes. `updates.jsonl` records each minibatch's loss, gradient, clipping, prefix-score diagnostics and actual actor change. `iteration-N.json` records interaction cost and GPU memory. `ppo-N.pt` retains the policy/optimizer schema used by the existing evaluator. Both smoke checkpoints completed paired sixteen-scene development evaluation.

`--trace-controls` also saves raw policy action chunks, decoded pose targets, measured input states and the official controller's per-control IK/pose records under `traces/`. These files support the separate investigation of nonconverged controls and jerky motion without changing the controller or adding action smoothing.

Keep these output directories separate from `fruit-completion-v1`. Use development results to debug, then freeze fresh held-out initialization manifests and the evaluation rule before running final comparisons. Previously examined test outcomes are not an untouched test set for the tuned reward.
