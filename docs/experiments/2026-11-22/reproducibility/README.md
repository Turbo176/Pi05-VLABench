# Pose-policy reproducibility audit

## Observed failure

On November 22, two complete PPO rollouts used the same flower seed 1001, checkpoint, normalization, initial physics, initial policy images, and sampling seed. The first five controls matched exactly. At the next policy call, the right camera changed and the predicted actions diverged. One rollout failed after 600 controls; the other succeeded after 156 controls. [Original repeat audit](flower-ppo-original/summary.json)

A separate probe removed the neural policy and executed the same first five action targets six times. The full MuJoCo integration state, including solver warm starts, matched exactly before and after these controls. The inspected model parameters and actuator-command hashes also matched. Some camera channels differed by one uint8 level, including repeated renders of an unchanged physical state. [Fixed-action probe](fixed-first-chunk/summary.json) · [Executed probe source](fixed_first_chunk.py)

## Renderer check

Each profile rendered four cameras 16 times after the same fixed controls. This is a graphics repeatability test, not a task success evaluation.

| Profile | Render sets differing from the first set |
| --- | ---: |
| Default, four MSAA samples | 16/16 |
| Shadows disabled | 9/16 |
| Reflections disabled | 10/16 |
| Shadows and reflections disabled | 14/16 |
| MSAA samples set to zero | 0/16 |

Only the offscreen antialiasing setting changed in the last row. Cameras, image size, task physics, IK, actuator control, and success scoring stayed the same. The sample count is set before creating the first render context and is now recorded in the dataset and rollout metadata. [Raw graphics results](render-profiles/summary.json) · [Executed probe source](render_profiles.py)

With this setting, the unchanged PPO checkpoint repeated the entire 600-control flower rollout twice with identical policy inputs, predicted actions, and actuator-command hashes. Both trials failed. [Full closed-loop repeat audit](flower-ppo-msaa0/summary.json) · [Per-call records](flower-ppo-msaa0/repeats.jsonl)

The earlier reference/SFT/PPO results remain historical observations under the original rendering profile. They cannot establish an improvement under the corrected profile. All compared stages must use the same new manifests and renderer settings.

## Data and training checks

The selected SFT pilot made 90 optimizer updates with one sampled window per update: 30 fruit, 30 flower, and 30 fridge windows. It sampled 38 of the 63 training episodes. Its training helper called `compute_loss` without `train=True`, leaving RLinf's optional training image augmentation disabled. The revised helper enables it explicitly, records that setting, and supports accumulation of balanced task gradients before each optimizer update. Legacy training can explicitly disable augmentation.

Fruit validation seeds 55 and 2027 had successful original joint-control demonstrations but unsuccessful converted-pose action replays. These failures were already present in the [earlier label audit](../../2026-11-21/failure-diagnosis/reference-aligned-fruit/label-audit.jsonl). We reran the official expert through the same pose controller at their original held-out initializations. Seed 55 succeeded in 141 controls and seed 2027 in 151. They remain validation episodes. [Source replacements](validation-source-overrides.json)

The corrected dataset re-executes every admitted pose action in fresh MuJoCo and records the actual pre-action images and measured state. A failed replay is retained as a partial file and stops admission. Original source files remain intact. The training and evaluation loaders reject inconsistent renderer metadata. PPO collection inherits the SFT checkpoint's renderer setting.

The paired summary now checks the base-weight hash, rendering profile, cameras' initial image hashes, normalization, instruction, sampling scheme, and control budget. It separately verifies that PPO descends from the evaluated SFT checkpoint. The reference policy may use configuration metadata from an earlier SFT run without being mislabeled as having loaded those trained parameters.

## Commands verified in the existing environment

Use the pinned VLABench OpenPI fork and RLinf runtime. The following commands run from the project root. Output paths must be fresh.

```bash
python scripts/prepare_reference_pose_cases.py --task insert_flower --seeds 1001 \
    --render-samples 0 --output "$CASES/flower-1001-msaa0.jsonl"
python scripts/audit_pose_reproducibility.py \
    --base "$REFERENCE/model.safetensors" --fork "$OPENPI_FORK" --load-run "$SFT_RUN" \
    --ppo-weights "$PPO_RUN/ppo-1.pt" --stage ppo --task insert_flower --seed 1001 \
    --paired-episodes "$CASES/flower-1001-msaa0.jsonl" --render-samples 0 --output "$AUDIT_OUT"
python scripts/collect_reference_fruit_pose_expert.py --task select_fruit --seed 55 2027 \
    --budget 600 --render-samples 0 --cache "$REPAIRED_VALIDATION"
python scripts/rerender_pose_dataset.py --dataset "$ORIGINAL_POSE_DATASET" \
    --render-samples 0 --workers 4 --source-overrides "$SOURCE_OVERRIDES" --output "$REPLAY_OUT"
```

No model weights, simulator assets, or dataset episodes are included in this report.
