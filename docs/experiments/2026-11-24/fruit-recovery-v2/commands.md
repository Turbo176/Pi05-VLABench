# Reproduction commands

Run from the prepared Linux workspace described in [the GPU guide](../../../single-gpu.md). The commands below identify the actual input paths and arguments. Use fresh output paths when repeating them. Weights, assets, environments and full datasets are not stored in Git.

```bash
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/code/scripts:$PWD/.local/lerobot-converter-arrow18:$PYTHONPATH"
base=.local/checkpoints/vlabench-primitive-rlinf/model.safetensors
study=runs/fruit-recovery-20261124
```

## SFT continuation and parent selection

```bash
"$py" code/scripts/train_reference_aligned_fruit.py \
  --base "$base" --fork "$reference_fork" \
  --reference-start .local/checkpoints/vlabench-primitive-rlinf \
  --resume-run runs/pose-robustness-v2/two-task-conservative-90 \
  --lerobot .local/reference-aligned/lerobot-tritask-msaa0-v2 \
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-small-extras \
  --extra-lerobot .local/reference-aligned/lerobot-msaa0-flower-105-134 \
  --paired-episodes .local/reference-aligned/msaa0-development-cases/select_fruit.jsonl \
  --multitask --training-tasks select_fruit insert_flower --render-samples 0 \
  --gradient-accumulation 2 --micro-batch-size 4 \
  --actor-lora-rank 16 --vlm-lora-rank 0 --lr 0.000001 --no-image-augmentation \
  --updates 360 --save-every 90 --skip-evaluation \
  --output "$study/sft-continuation-360"
```

The four-scene screen selected the 180-update candidate over 360. The expanded sixteen-scene development comparison then selected SFT180 at 9/16 over SFT90 at 8/16. The chosen parent is `$study/sft-continuation-360/checkpoint-0180`; `checkpoint-0270` was saved but was not a selection candidate.

## Expanded PPO

The matched two-start reward smoke commands are retained in [the GPU guide](../../../single-gpu.md#fruit-reward-follow-up). Their development comparison preferred the legacy reward. This separate expansion starts from the selected SFT180:

```bash
parent="$study/sft-continuation-360/checkpoint-0180"
"$py" code/scripts/train_fruit_ppo.py \
  --base "$base" --fork "$reference_fork" --sft-run "$parent" \
  --output "$study/ppo-expanded-selected/legacy-distance-v1" \
  --seeds 1303 1304 1305 1306 1307 1308 1312 1314 --iterations 3 \
  --critic-warmup-epochs 2 --max-steps 600 --minibatch 16 --shaping 0.5 \
  --seed 372 --lr 0.000002 --critic-lr 0.0001 --target-kl 0.02 \
  --reward-version legacy-distance-v1 --trace-controls
```

The maximum training budget is 19,200 controls, including critic warmup. Read the actual cost from the rollout log and iteration reports. Every saved iteration is evaluated on the same sixteen development scenes. Selection prefers more official successes, then fewer controls, then the earlier iteration. PPO checkpoint identity is frozen before opening final test outcomes.

## Paired final evaluation

The test manifest contains forty scenes, ten naturally reset targets per fruit. Prepare a cases directory containing `select_fruit-official.jsonl` from the retained test manifest, then use the retained `test-protocol.json`. Set `chosen` to the checkpoint recorded in `final-selection.json` once development selection completes.

```bash
report=docs/experiments/2026-11-24/fruit-recovery-v2
mkdir -p "$study/reserved-test"
cp "$report/evidence/test-cases.jsonl" "$study/reserved-test/select_fruit-official.jsonl"
cp "$report/evidence/test-protocol.json" "$study/reserved-test/protocol.json"
chosen="$study/ppo-expanded-selected/legacy-distance-v1/ppo-1.pt"
for stage in base sft ppo; do
  extra=()
  if [[ "$stage" == ppo ]]; then extra=(--ppo-weights "$chosen"); fi
  "$py" code/scripts/evaluate_pose_development.py \
    --base "$base" --fork "$reference_fork" --run "$parent" \
    --cases "$study/reserved-test" --protocol "$study/reserved-test/protocol.json" \
    --tasks select_fruit --suite ordinary --stage "$stage" --workers 3 "${extra[@]}" \
    --output "$study/repeated-test-$stage"
done
```

The recorded run overlaps the already-frozen reference/SFT evaluations with PPO training and keeps their outcomes sealed during PPO selection. This loop uses the same evaluator settings with fresh output paths. The reference is the published VLABench model already fine-tuned upstream.

## Analysis and recording

`analyze.py` checks all forty paired scene identities and controls, recomputes success counts and Wilson intervals, and can audit the saved actor/critic tensors and optimizer counters using `--sft-weights`, `--ppo-weights`, and `--ppo-updates`. Its outcome analysis has no need to download weights; tensor auditing requires the supplied local checkpoint files.

From a source checkout, the retained outcome and file checks use only the Python standard library:

```bash
cd docs/experiments/2026-11-24/fruit-recovery-v2
python analyze.py --verify-only --cases evidence/test-cases.jsonl \
  --base evidence/test-reference.jsonl --sft evidence/test-sft.jsonl \
  --ppo evidence/test-ppo.jsonl --selection evidence/final-selection.json --output results
```

To regenerate the figures from the actual raw outcomes and logs, run the same command without `--verify-only` and add `--sft-updates evidence/sft-parent/updates.jsonl --ppo-updates evidence/ppo/updates.jsonl`. This needs NumPy and Matplotlib. Adding the exact local `--sft-weights` and `--ppo-weights` additionally rechecks saved tensor deltas and optimizer counters with PyTorch. Keep a copy of the retained figures when reproducing them with another Matplotlib version; image bytes can vary across versions.

`record_fruit_completion.py` accepts `--ppo-weights` and `--label` for the selected checkpoint. It re-executes the policy, records a fresh simulation, and rejects a recording whose full action hash differs from the scored episode. Display resolution changes do not alter the policy cameras.

The main recordings use a fresh presentation directory and seed 6005. Copy `evidence/test-cases.jsonl` to `"$study/presentation-main/cases/select_fruit-official.jsonl"` and the retained `evidence/video-plan.json` to `"$study/presentation-main/video-plan.json"` before recording:

```bash
"$py" code/scripts/record_fruit_completion.py --root "$study/presentation-main" \
  --base "$base" --fork "$reference_fork" --run "$parent" --stage ppo \
  --ppo-weights "$chosen" --scored-episodes "$study/final-test-ppo/episodes.jsonl" \
  --label fruit-ppo-v2-iteration1 --display-image-size 512
```

For reference and SFT, select `--stage base --label published-reference-already-finetuned` or `--stage sft --label shared-multitask-sft180`, point `--scored-episodes` at the matching test file, and omit `--ppo-weights`. Each stage writes a separate directory and checks all scored action and policy-image hashes. MP4s preserve the complete 10 Hz rollout. The curated GIF previews retain normal speed at 10 fps and stop after at most 15 seconds.
