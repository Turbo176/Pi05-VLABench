# Stage 2 dataset pilot

The pilot uses the VLABench Franka, MuJoCo, and the upstream expert skills. Each capture records nine actuator targets and aligned four-camera RGB at 10 Hz. The LeRobot v3.0 export stores the RGB losslessly as image features for frame-by-frame validation. `task` contains the original natural-language instruction.

## Local setup

Run the [stage 1 setup](stage1.md) first. Install LeRobot 0.4.4 into a separate CPU environment from WSL.

```bash
bash scripts/setup_lerobot_wsl.sh
```

The simulator venv is `/var/tmp/pi05-vlabench-1000/venv`. The LeRobot venv is `/var/tmp/pi05-vlabench-1000/lerobot`. Asset manifests and caches are under `.local/`. Captures and LeRobot datasets are under `runs/` and `data/` respectively.

Fetch the extra official assets from Windows PowerShell at the project root.

```powershell
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(flowers/bloom_flower/(rose/rose_0|tulip/tulip_draft_1|chrysanthemum/chrysanthemum_3|sunflower/sunflower_0|daisy/daisy_0|peony/peony_4)|containers/vases/(vase_3|vase_5))/' --manifest .local/assets-insert-flower.json
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/containers/vases/(vase_1|vase_4)/' --manifest .local/assets-insert-flower-vases.json
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(counters/counter_0|stoves/basic_sleek_induc|dishes/dish_without_plate_0|tablewares/pans/(pan_0|pan_24)|condiment/(bbq_sauce/bbq_sauce1|ketchup/ketchup_7|shaker/shaker_3|salad_dressing|hotsauce)|tube/nametag)/' --manifest .local/assets-add-condiment.json
```

The extra asset ranges total about 64 MiB. The pinned VLABench `insert_flower` config expects `vase_seen` and `vase_unseen`. The local wrapper registers these names with the original Vase class and the `vase_1` and `vase_4` assets used by the pinned in-distribution and cross-category tracks. The other two vase models were in the initial trial asset download. Asset variety is intentionally small for this pilot.

## Expert screening

| Official task | Unique completed seeds | Successes | Other observations |
| --- | ---: | ---: | --- |
| `select_fruit` | 3 | 3 | Seeds 42–44, with seed 42 repeated for an acquisition regression check. |
| `insert_flower` | 9 | 3 | Seeds 42, 46, and 50 succeeded. The six unsuccessful runs all reported a failed `lift` stage before the episode ended. |
| `add_condiment` | 5 | 0 | Seeds 42–45 and 51 all reported a failed `pick` stage. |
| `get_coffee` | 0 | — | Seeds 42 and 43 timed out in `press` after 300 seconds during full capture. Seed 43 also timed out in the reduced screening mode after 220 seconds. |
| `take_out_cool_drink` | 4 | 2 | Seeds 43 and 46 succeeded. Seed 45 failed during reset and reproduced `mjWARN_BADQACC` on a reset-only retry. Seeds 42 and 44 reported failed drink grasp and placement. |

Each row counts distinct initializations in the local asset profile. The success counts apply to the expert screening set. The `--screen` mode logs controls and official success while skipping RGB collection. A successful screen must be captured again with four views before export. See [screening records](stage2-expert-screening.json) for seeds, instructions, and reported skill outcomes.

### Composite task asset profile

`take_out_cool_drink` uses the official small fridge and a reduced drink asset set. VLABench resolves the background to the default scene. The asset transfer used 14,062,452 bytes after the direct Drive request timed out.

```powershell
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(containers/fridge/small_fridge/|drinks/(sota_drink/(cola/bottled_cola_1/|spirit/bottled_spirit_1/)|juice/boxed_drink_1/|energy_drink/(monster|jumpstart|redbull)/))' --manifest .local/assets-take-out-cool-drink.json
```

The coffee pilot used the official Nespresso machine and two official mugs (`mug_0`, `mug_17`). The local wrapper makes those mugs available under `mug_seen` and `mug_unseen`.

```powershell
python scripts/vlabench_assets.py objects --proxy-fallback --match '^obj/meshes/(coffee_machines/nespresso|tablewares/cups/(mug_0|mug_17))/' --manifest .local/assets-get-coffee.json
```

## Capture and conversion

Use one simulator capture at a time on the local WSL host. Concurrent capture hit the WSL memory limit. The capture wrapper now materializes upstream observations when a skill reads a field, including the planning point cloud. It writes the aligned RGB and measured joint positions at each action tick. See the stage 1 runbook for the action units and camera order.

```bash
bash scripts/run_wsl.sh collect --task insert_flower --seed 42 --output runs/stage2/insert_flower-42-retry
bash scripts/run_wsl.sh replay runs/stage2/insert_flower-42-retry/episode.h5 --hd
bash scripts/convert_lerobot_wsl.sh \
  --episode train=runs/select_fruit-42/episode.h5 \
  --output data/stage2-single-episode
```

The converter accepts repeated `--episode SPLIT=PATH` flags. It checks for a successful and unique terminal step, shape and timestamp consistency, duplicate task/seed pairs, and duplicate initial simulator state across splits. Each split has a separate LeRobot dataset root. The validator reopens each split with `LeRobotDataset` and checks every frame's four images, state, action, timestamp, language, and indices against the source. It rejects unfinished demonstrations and duplicate initializations before writing.

Available policy observations are the four `observation.images.*` fields, `observation.state`, and the instruction returned by `dataset[i]['task']`. `action` is the supervised label. `episode.h5` and the local audit manifest retain provenance and diagnostics. The training adapter must use the observation allowlist. The model camera mapping is covered in the [SFT to PPO source audit](sft-rl-compatibility.md).

The seed 46 flower replay re-executed 203 controls with zero difference in the simulator state. With `--check-rgb`, the replay also compared 204 complete four-view frames against the capture, including the final image.

```bash
bash scripts/run_wsl.sh replay runs/stage2/insert_flower-46-screen-success/episode.h5 --check-rgb
```

## OpenPI dataset reader

The pinned OpenPI imports a LeRobot v2.1 reader at revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. Its lockfile uses Hugging Face Datasets 3.6.0 and PyArrow 20.0.0. Datasets 4.x changed the return type of column access and the old reader failed at `torch.stack`. The setup script installs the two locked packages into a separate WSL directory for `--openpi`.

The expanded pilot uses two successful tasks in each split. Training uses fruit seed 42 and flower seed 42. Validation uses fruit seed 43 and flower seed 46. Test uses fruit seed 44 and flower seed 50. The flower splits were selected from successful expert runs. Treat this as a format validation dataset. Future closed-loop evaluation needs a fixed set of seeds chosen before running any policies.

From WSL, collect the source episodes into fresh directories. Check out the pinned reader into `.local/lerobot-openpi` for the v2.1 export.

```bash
bash scripts/run_wsl.sh collect --task select_fruit --seed 43 --output runs/stage2/select_fruit-43-pilot
bash scripts/run_wsl.sh collect --task select_fruit --seed 44 --output runs/stage2/select_fruit-44-pilot
bash scripts/run_wsl.sh collect --task insert_flower --seed 46 --output runs/stage2/insert_flower-46-screen-success
bash scripts/run_wsl.sh collect --task insert_flower --seed 50 --output runs/stage2/insert_flower-50-screen-success

source_episodes=(
  --episode train=runs/select_fruit-42/episode.h5
  --episode train=runs/stage2/insert_flower-42-retry/episode.h5
  --episode validation=runs/stage2/select_fruit-43-pilot/episode.h5
  --episode validation=runs/stage2/insert_flower-46-screen-success/episode.h5
  --episode test=runs/stage2/select_fruit-44-pilot/episode.h5
  --episode test=runs/stage2/insert_flower-50-screen-success/episode.h5
)
bash scripts/convert_lerobot_wsl.sh --openpi --action-horizon 50 \
  "${source_episodes[@]}" --output data/stage2-expanded-v21
bash scripts/convert_lerobot_wsl.sh \
  "${source_episodes[@]}" --output data/stage2-expanded-v3
```

The 50-action check exercises the same `delta_timestamps` query as OpenPI's data loader. Near the end of each episode, it verifies that the reader repeats the final action and marks padding at that episode's boundary. See the [v2.1 readback](stage2-lerobot-v21.json) and [v3.0 readback](stage2-lerobot-v3.json).

## Three-task training and validation

The fridge task adds an articulated object with door opening and closing. Its training episode uses seed 43 and its validation episode uses seed 46. Successful seeds have the same short instruction about wanting a cool drink. The language generalization tests need prompts that identify an unambiguous target. The expanded test split covers fruit and flower.

```bash
bash scripts/run_wsl.sh collect --task take_out_cool_drink --seed 43 --budget 600 \
  --output runs/stage2/take_out_cool_drink-43-screen-success
three_task_train=(
  --episode train=runs/select_fruit-42/episode.h5
  --episode train=runs/stage2/insert_flower-42-retry/episode.h5
  --episode train=runs/stage2/take_out_cool_drink-43-screen-success/episode.h5
)
bash scripts/convert_lerobot_wsl.sh --openpi --action-horizon 50 \
  "${three_task_train[@]}" --output data/stage2-three-task-train-v21
bash scripts/convert_lerobot_wsl.sh \
  "${three_task_train[@]}" --output data/stage2-three-task-train-v3
```

Fridge seed 43 passed a fresh-simulator replay of all 477 commands at 10 Hz. The MuJoCo state matched exactly, and all 478 four-view frames matched the original capture. See the [replay record](stage2-fridge-rgb-replay.json).

The three-task training split contains 790 frames. Both [v2.1](stage2-three-task-train-v21.json) and [v3.0](stage2-three-task-train-v3.json) passed frame-by-frame verification. The v2.1 reader also passed nine checks of 50-step action windows.

```bash
bash scripts/run_wsl.sh collect --task take_out_cool_drink --seed 46 --budget 600 \
  --output runs/stage2/take_out_cool_drink-46-screen-success
three_task_validation=(
  --episode validation=runs/stage2/select_fruit-43-pilot/episode.h5
  --episode validation=runs/stage2/insert_flower-46-screen-success/episode.h5
  --episode validation=runs/stage2/take_out_cool_drink-46-screen-success/episode.h5
)
bash scripts/convert_lerobot_wsl.sh --openpi --action-horizon 50 \
  "${three_task_validation[@]}" --output data/stage2-three-task-validation-v21
bash scripts/convert_lerobot_wsl.sh \
  "${three_task_validation[@]}" --output data/stage2-three-task-validation-v3
```

The three-task validation split contains 832 frames. See the [v2.1](stage2-three-task-validation-v21.json) and [v3.0](stage2-three-task-validation-v3.json) validation records. Across the train, validation, and test splits, all eight episodes have distinct task/seed pairs and initialization hashes.
