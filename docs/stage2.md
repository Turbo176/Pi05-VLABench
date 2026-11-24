# Demonstrations and LeRobot data

The current fruit policy uses **47 training episodes / 6,986 frames** and **four validation episodes / 589 frames**. The earlier shared fruit/flower experiment used **64 training episodes / 9,904 frames** and **five validation episodes / 744 frames**: 47 fruit episodes, including two expert corrections, plus 17 flower episodes. The [data card](system-design.md#data-card) separates these checkpoints; the [transform/controller audit](experiments/2026-11-22/pose-robustness-v2/training-contract-audit.json) documents the common pose interface.

## Capture, replay, export

Use the isolated simulator from [local setup](stage1.md). The pose capture tools run in the prepared VLABench/OpenPI environment described in the [GPU guide](single-gpu.md). They record actual commands, pre-action images and measured state. A fresh simulation executes the saved commands before an episode is admitted for training. Failed and partial attempts remain outside the training split.

`collect_reference_fruit_pose_expert.py` captures one or more official expert starts; `collect_pose_expert_batch.py` screens seeds and admits replay-verified episodes. `rerender_pose_dataset.py` re-executes pose commands and records the observations produced by that controller. The current data and evaluation use `render_samples=0`; the [renderer audit](experiments/2026-11-22/reproducibility/README.md) explains this setting.

The following source-repair command was executed at the original validation starts. These episodes remain validation data:

```bash
python scripts/collect_reference_fruit_pose_expert.py --task select_fruit --seed 55 2027 \
  --budget 600 --render-samples 0 --cache "$REPAIRED_VALIDATION"
```

LeRobot conversion uses `convert_lerobot.py --reference-pose` with repeated `--episode SPLIT=PATH` arguments. `--format v2.1` selects the format used by the executed SFT reader. The converter checks successful termination, unique task/seed pairs and initializations across splits, finite arrays, per-frame language, timestamps, and image/state/action readback. `reference_pose_lerobot.py` reads complete ten-action windows without crossing episode boundaries.

At 10 Hz, observation row `t` precedes command `t`. The three policy cameras are forward, right, and wrist. Pose actions contain XYZ in meters, Euler coordinates in radians, and the gripper coordinate. Normalization and pose decoding use the shared upstream transforms. Expert target geometry, rewards, success and terminal labels are excluded from policy observations. [Full policy/control contract](system-design.md#policy-and-control-contract)

## Expert replay gallery

The three README expert videos come from the earlier joint-space collection: nine actuator targets, four stored RGB views, and measured joint positions at 10 Hz. They are expert action replays in fresh simulation, separate from learned-policy rollouts and from the pose training totals above.

| Official task | Expert replay evidence |
| --- | --- |
| `select_fruit` | [Replay](stage1-action-replay.json) |
| `insert_flower` | [Replay](stage2-insert-flower-replay.json) |
| `take_out_cool_drink` | [Four-view replay](stage2-fridge-rgb-replay.json) |

The complete older training export contains 60 episodes / 16,520 frames, with 20 episodes per task. [Collection summary](stage2-expansion-06.json) and [LeRobot v2.1 readback](stage2-three-task-train-v21.json) retain the final checks. Intermediate expansion batches are superseded by this summary.

A minimal local capture, replay and export was executed as follows:

```bash
bash scripts/setup_lerobot_wsl.sh
bash scripts/run_wsl.sh collect --task insert_flower --seed 42 --output runs/stage2/insert_flower-42-retry
bash scripts/run_wsl.sh replay runs/stage2/insert_flower-42-retry/episode.h5 --hd
bash scripts/convert_lerobot_wsl.sh \
  --episode train=runs/select_fruit-42/episode.h5 \
  --output data/stage2-single-episode
```

This last command is an export check, not the full SFT training dataset. The independent converter environment lives at `/var/tmp/pi05-vlabench-$(id -u)/lerobot`. Source captures, assets, caches and generated datasets remain outside Git.

Flower and fridge require additional official assets. The subset downloader commands used for the final gallery include:

```powershell
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(flowers/bloom_flower/(rose/rose_0|tulip/tulip_draft_1|chrysanthemum/chrysanthemum_3|sunflower/sunflower_0|daisy/daisy_0|peony/peony_4)|containers/vases/(vase_3|vase_5))/' --manifest .local/assets-insert-flower.json
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/containers/vases/(vase_1|vase_4)/' --manifest .local/assets-insert-flower-vases.json
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(containers/fridge/small_fridge/|drinks/(sota_drink/(cola/bottled_cola_1/|spirit/bottled_spirit_1/)|juice/boxed_drink_1/|energy_drink/(monster|jumpstart|redbull)/))' --manifest .local/assets-take-out-cool-drink.json
```

Direct transfers are tried first. The reduced asset variety does not establish performance on the full benchmark. The [initial expert screening](stage2-expert-screening.json) also records unsuccessful candidate tasks; the three gallery videos are successful examples rather than expert success-rate estimates.
