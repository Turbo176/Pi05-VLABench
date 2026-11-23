# Stage 2 dataset pilot

> Original local joint-space dataset. The shared fruit/flower SFT uses the separate pose export in the [current data card](system-design.md#data-card); its 64 training episodes are not added to the older 60-episode collection below.

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

| Official task | Unique attempted seeds | Successes | Other observations |
| --- | ---: | ---: | --- |
| `select_fruit` | 3 | 3 | Seeds 42–44, with seed 42 repeated for an acquisition regression check. |
| `insert_flower` | 10 | 3 | Seeds 42, 46, and 50 succeeded. Seed 52 failed in the follow-up. |
| `add_condiment` | 5 | 0 | Seeds 42–45 and 51 all reported a failed `pick` stage. |
| `get_coffee` | 2 | 0 | Seeds 42 and 43 timed out in `press` after 300 seconds during full capture. Seed 43 also timed out in the reduced screening mode after 220 seconds. |
| `take_out_cool_drink` | 6 | 3 | Seeds 43, 46, and 52 succeeded. Seed 45 reproduced `mjWARN_BADQACC` on a reset-only retry. Seeds 42 and 44 reported failed drink grasp and placement. |

Each row counts distinct attempted seeds in the local asset profile, including reset errors and timeouts. Repeat runs of a seed count once. These measurements describe the official expert on a small screening set. The initial three-task training split has one successful episode per task and serves as a data pipeline check. The `--screen` mode logs controls and official success while skipping RGB collection. Before export, replay the saved controls in a fresh simulator and record four RGB views. See [screening records](stage2-expert-screening.json) for seeds, instructions, and reported skill outcomes.

The follow-up used seed 52 for both candidates with a 600-control-step budget and RGB capture disabled. The flower run ended unsuccessfully at 173 steps. The fridge run succeeded at 501 steps. The first training expansion includes the successful fridge trace after four-view action replay.

```bash
for task in insert_flower take_out_cool_drink; do
  bash scripts/run_wsl.sh collect --task "$task" --seed 52 --screen --budget 600 \
    --output "runs/stage2/followup-${task}-52"
done
```

### What failed in insert_flower

The stage log marks `lift` success by end-effector position tolerance. It does not directly check whether the flower was lifted. We restored the saved simulator states at skill boundaries for seeds 42–50 and checked them with the official task condition. Each flower rose 24.15–26.18 cm during the first lift. At the end of the six unsuccessful runs, the flower reference point still missed the vase's acceptance bounds.

| Axis outside the final acceptance bounds | Seeds | Distance beyond the bounds |
| --- | --- | --- |
| Height | 43, 45, 47, 49 | 3.07, 9.85, 0.61, 6.73 cm |
| Horizontal | 44, 48 | 4.28, 2.01 cm along x |

The data point to insertion height and horizontal alignment as failure modes in this sample. The measurements do not pinpoint the underlying contact or controller cause. Reproduce the diagnostic from WSL using the original captures.

```bash
export PYTHONPATH="$PWD/.local/VLABench:$PWD/.local/rrt-algorithms"
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/.local/cache"
/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python scripts/inspect_insert_flower.py \
  .local/insert-flower-diagnosis.json \
  runs/stage2/insert_flower-{42-retry,43-pilot,44-pilot,45-screen,46-screen,47-screen,48-screen,49-screen,50-screen}
```

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


## First training expansion

Screening ran serially with the pinned expert on seeds 53 through 55. Each attempt had a 600-step limit and a 240-second process timeout.

| Task | Expert successes in this batch | Total training episodes | Total training frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | 3/3 | 4 | 597 |
| `insert_flower` | 1/3 | 2 | 335 |
| `take_out_cool_drink` | 0/3 | 2 | 978 |
| Total | 4/9 | 8 | 1,910 |

The three fruit successes targeted two pears and one banana. Both tulip attempts failed, and the rose attempt succeeded. The saved physics states show that seed 53 lifted the tulip 25.88 cm and stopped 2.14 cm short of the vase acceptance region in height. Seed 54 never raised the tulip during the first lift stage. The successful seed 55 raised the rose 26.02 cm before inserting it. Fridge seed 53 exhausted its step budget during door closing. Seed 54 crashed during the second `move_offset` stage. Seed 55 reached the end of the expert sequence without satisfying the official condition. The second fridge training trace came from the earlier successful screen at seed 52. See the [run report](stage2-expansion-01.json).

Each of the five added demonstrations reached official success when its saved controls ran in a fresh simulator. The reset and per-step physics state, measured joints, actions, and timestamps matched the original trace exactly. Each replay recorded four RGB views. The LeRobot v2.1 reader verified all 1,910 frames and checked 24 positions for 50-step action windows and padding at episode boundaries. The eight training, three validation, and two test episodes have distinct task/seed pairs and initialization hashes. Use this export for data validation while collecting the first 20 successful trajectories per task.

Run from the project root in WSL after generating the three original training captures and the seed 52 screen above.

```bash
bash scripts/screen_expansion_01_wsl.sh
bash scripts/export_expansion_01_wsl.sh
/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python scripts/report_expansion_01.py
```

The commands store the raw attempts and replays in `runs/stage2-expansion-01/` and the verified dataset in `data/stage2-expansion-01-v21/`. The screen command skips completed attempts. Use a fresh path if an attempt was interrupted before its exit status was written.


## Second training expansion

| Task | Expert successes at seeds 56 through 58 | Total training episodes | Total training frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | 3/3 | 7 | 1,020 |
| `insert_flower` | 1/3 | 3 | 509 |
| `take_out_cool_drink` | 2/3 | 5 | 2,454 |
| Total | 6/9 | 15 | 3,983 |

The six successes and a separate successful retry of fridge seed 54 passed fresh action replay. The initial fridge crash remains in the first batch. Flower seed 57 raised a `TypeError` while dm_control copied the MuJoCo state for inverse kinematics. The underlying cause remains unconfirmed. See the [expansion report](stage2-expansion-02.json).

Fruit seed 56 reached official success in all 151 replayed steps. Its intermediate simulator states differed from the original screen. The RGB and measured state in LeRobot come from the actual replay before each saved action. The replay report records the maximum state difference. The export command explicitly opts in to this episode. All other added traces matched the recorded simulator state exactly.

The v2.1 reader verified all 3,983 frames and 45 positions for 50-step action windows. Across training, validation, and test, the 20 episodes have distinct task/seed pairs and initialization hashes. The current training support covers four fruits and three rose episodes. All five fridge episodes use the same instruction. Fridge holds 2,454 of the 3,983 frames, which should inform later task sampling.

The [flower contact measurements](stage2-flower-contacts.json) show gripper-to-flower contact at the first lift boundary in the original rose demo (seed 42), with the rose rising 25.91 cm. The new unsuccessful tulip seeds 53 and 56 also had contact and lifted the flower over 25 cm. Both stopped above the official vase acceptance region. Tulip seed 54 had no such contact and the flower did not rise at that boundary.

From the project root in WSL, reproduce the retry and extend the existing first export. The scratch directory keeps LeRobot's temporary image reads on the Linux filesystem. A direct export under `/mnt/d` returned `ENOMEM` during image embedding.

```bash
bash scripts/screen_expansion_01_wsl.sh runs/stage2-fridge-recheck take_out_cool_drink 54
/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python scripts/extend_training_wsl.py \
  data/stage2-expansion-01-v21/verification.json runs/stage2-expansion-02 56 58 \
  --scratch-root "/var/tmp/pi05-vlabench-$(id -u)" \
  --recovered-screen runs/stage2-fridge-recheck/take_out_cool_drink-54-screen/screening.h5 \
  --allow-state-drift select_fruit:56
```

Review the retry status before export. The first fridge attempt at seed 54 crashed during the second `move_offset` stage. The measured retry used the same pinned simulator source under the Linux filesystem with the source asset path linked to the existing asset tree. Its outcome can vary across runs.


## Third training expansion

| Task | Expert successes at seeds 59 through 62 | Total training episodes | Total training frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | No new attempts | 7 | 1,020 |
| `insert_flower` | 2/4 | 5 | 868 |
| `take_out_cool_drink` | 3/4 | 8 | 3,966 |
| Total | 5/8 | 20 | 5,854 |

All five additions passed official success and matched the saved simulator state during action replay. Flower seed 59 raised `mjWARN_BADQACC` during the upstream reset. Seed 60 ran all four skills and missed the vase condition. Both successful flower additions used rose targets. The v2.1 reader verified all frames and 60 positions for 50-step action windows. The 25 training, validation, and test episodes have distinct task/seed pairs and initial state hashes. See the [batch results](stage2-expansion-03.json).

```bash
/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python scripts/extend_training_wsl.py \
  data/stage2-expansion-02-v21/verification.json runs/stage2-expansion-03 59 62 \
  --task insert_flower --task take_out_cool_drink --scratch-root "/var/tmp/pi05-vlabench-$(id -u)"
```

## Fourth training expansion

| Task | Expert successes at seeds 63 through 72 | Total training episodes | Total training frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | 10/10 | 17 | 2,566 |
| `insert_flower` | 3/10 | 8 | 1,382 |
| `take_out_cool_drink` | 1/10 | 9 | 4,478 |
| Total | 14/30 | 34 | 8,426 |

The flower data includes its first training tulip at seed 65. Fridge seed 71 successfully removed a Monster drink and closed the door. All 14 additions exactly matched the recorded simulator state during action replay. LeRobot v2.1 verified every frame and checked 102 positions for 50-step action windows. The 39 episodes across training, validation, and test have distinct task and seed pairs and initial state hashes. The [batch report](stage2-expansion-04.json) records the data and normalization hash.

```bash
/var/tmp/pi05-vlabench-$(id -u)/venv/bin/python scripts/extend_training_wsl.py \
  data/stage2-expansion-03-v21/verification.json runs/stage2-expansion-04 63 72 \
  --scratch-root "/var/tmp/pi05-vlabench-$(id -u)"
```

We checked the last saved physics state for fridge seeds 63 through 67 using the two official completion conditions. Seeds 64, 65, and 66 closed the door with the target drink still inside. Seed 67 removed the target and left the door open. Seed 63 missed both conditions. The [diagnostic log](stage2-fridge-conditions.json) includes the measurements at all skill boundaries and seed 59 as a successful control.

This batch used a local copy of the same pinned VLABench checkout and the already downloaded assets on the WSL filesystem. `PI05_SIM_SOURCE_ROOT` selects the filesystem location. It defaults to the project checkout. The conversion stages one source episode at a time on the Linux filesystem and discards the pinned writer's in-memory image table after each episode is saved. Readback opens the finished Parquet files through the pinned reader.


## Fifth training expansion

| Task | Expert successes at seeds 73 through 78 | Total training episodes | Total training frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | 3/3 | 20 | 3,031 |
| `insert_flower` | 2/6 | 10 | 1,732 |
| `take_out_cool_drink` | 1/6 | 10 | 4,995 |
| Total | 6/15 | 40 | 9,758 |

The fruit screen stopped at seed 75 after reaching 20 replay-verified training episodes. All six additions reproduced their recorded simulator states exactly. The batch added the first chrysanthemum and Jumpstart demonstrations. Eight of the 10 flower episodes target roses, and the 10 fridge episodes share one official instruction across five drink types. LeRobot v2.1 verified every frame and checked 120 positions for 50-step action windows. The training, validation, and test data have 45 distinct task and seed pairs and 45 distinct initialization hashes. The test data covers fruit and flower. See the [batch report](stage2-expansion-05.json).

```bash
runtime="/var/tmp/pi05-vlabench-$(id -u)"
export PI05_SIM_SOURCE_ROOT=$(cat .local/active-sim-native)
"$runtime/venv/bin/python" scripts/extend_training_wsl.py \
  data/stage2-expansion-04-v21/verification.json runs/stage2-expansion-05 73 78 \
  --scratch-root "$runtime"
PYTHONPATH=.local/openpi-norm-src:.local/openpi-norm-deps \
  "$runtime/lerobot/bin/python" scripts/compute_policy_norm_stats.py \
  data/stage2-expansion-05-v21/train data/stage2-expansion-05-v21/train-stats
"$runtime/venv/bin/python" scripts/audit_expansion.py \
  runs/stage2-expansion-05/dataset_report.json data/stage2-expansion-05-v21/verification.json \
  docs/stage2-expansion-05.json
```

The expansion runner can append through the pinned LeRobot v2.1 writer to a scratch copy of the verified training split. The four-episode smoke run copied three episodes, appended fruit seed 73, and verified all 952 frames and 12 action windows. The source metadata checksums matched before and after. Full export readback still checks every episode.

A three-process screening probe on seeds 79–82 exhausted the local 7.4 GiB WSL memory. The kernel logs recorded out-of-memory kills. The next screen uses one simulator process and retains the probe logs in `runs/stage2-expansion-06/`. The serial attempts and action replays use `runs/stage2-expansion-06-serial/`.


## Sixth training expansion

| Task | Expert successes in this batch | Total training episodes | Verified frames |
| --- | ---: | ---: | ---: |
| `select_fruit` | No new attempts | 20 | 3,031 |
| `insert_flower` | 10/42 | 20 | 3,482 |
| `take_out_cool_drink` | 10/23 | 20 | 10,007 |
| Total | 20/65 | 60 | 16,520 |

The serial run stopped screening fridge at seed 101 and flower at seed 120 when each reached 20 successful demonstrations. The two flower physics errors and the fridge run that exhausted its action budget remain in the attempt log. Each successful addition exactly reproduced its simulator state trace. The LeRobot v2.1 reader verified 16,520 frames and checked 180 positions for 50-step action windows. The 60 training, three validation, and two pilot test episodes have distinct task and seed pairs and distinct initialization hashes. The [batch report](stage2-expansion-06.json) includes target counts and ranges of initial target positions recorded in the simulator.

Four fruit types, three flower types, and five drinks appear as targets. Fourteen flower episodes use roses, five use tulips, and one uses a chrysanthemum. The fridge episodes share one official instruction and make up 60.6% of training frames. This pilot supports the first GPU verification. The closed-loop results will determine whether to expand coverage.

```bash
runtime="/var/tmp/pi05-vlabench-$(id -u)"
export PI05_SIM_SOURCE_ROOT=$(cat .local/active-sim-native)
"$runtime/venv/bin/python" scripts/extend_training_wsl.py \
  data/stage2-expansion-05-v21/verification.json runs/stage2-expansion-06-serial 79 130 \
  --task insert_flower --task take_out_cool_drink --scratch-root "$runtime"
PYTHONPATH=.local/openpi-norm-src:.local/openpi-norm-deps \
  "$runtime/lerobot/bin/python" scripts/compute_policy_norm_stats.py \
  data/stage2-expansion-06-serial-v21/train data/stage2-expansion-06-serial-v21/train-stats
"$runtime/venv/bin/python" scripts/audit_expansion.py \
  runs/stage2-expansion-06-serial/dataset_report.json data/stage2-expansion-06-serial-v21/verification.json \
  docs/stage2-expansion-06.json
```
