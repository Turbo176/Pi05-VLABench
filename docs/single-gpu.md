# Single-GPU π0.5 experiments

The experiment runs pinned RLinf `Pi0` for supervised flow-matching and `Pi0RL` for post-training on the same official OpenPI base weights. The simulation uses VLABench and MuJoCo. See [pinned revisions](upstream.md) and the [input contract](policy-io.md).

## Current pose-policy route

The November 22 experiments use the published VLABench fine-tuned π0.5 reference, three RGB cameras (forward, right, wrist), measured end-effector pose and gripper state, and the actual task instruction. The policy predicts ten seven-coordinate Cartesian targets and executes five through the unchanged official adapter and inverse kinematics at 10 Hz. Its render profile disables offscreen MSAA for reproducible policy images. These settings differ from the historical 50-by-9 joint-space pilot documented below.

The selected fruit/flower training data contain 64 replay-verified episodes / 9,904 frames; five validation episodes / 744 frames remain separate. The real two-task smoke check passed backward, actor updates, checkpoint restoration and exact restored actions, peaking at 19.46 GB allocated CUDA memory. The 1,200-update shared SFT run completed in 1,991.8 seconds in its update loop; its offline action errors do not establish a closed-loop improvement. [Current experiment and measured outcomes](experiments/2026-11-22/pose-robustness-v2/README.md)

Actual storage on the cloned AutoDL instance:

| Content | Location |
| --- | --- |
| Project root | `/root/autodl-tmp/Pi05-VLABench` |
| Uploaded repository code | Project root + `/code` |
| Project interpreter | Project root + `/.local/torch28-venv/bin/python` |
| Pinned simulator, OpenPI fork and RLinf source | Project root + `/.local/upstream` |
| Reference weights | Project root + `/.local/checkpoints/vlabench-primitive-rlinf` |
| LeRobot exports and expert traces | Project root + `/.local/reference-aligned` |
| Package and model caches | Project root + `/.local/cache` |
| Experiment outputs | Project root + `/runs/pose-robustness-v2` |

The interpreter uses the image's `/root/miniconda3/bin/python3.12` and existing CUDA libraries; the project dependency paths are supplied by the ignored `.local/reference-env.sh`. The local repository is `D:\Robotics\Pi05-VLABench`, with private RGB data and checkpoint backups in its ignored `.local/autodl-backup` folder. The separate RoboCasa project is outside this project.

The [executed smoke and SFT command](experiments/2026-11-22/pose-robustness-v2/run-two-task-sft.sh) runs from the remote project root and explicitly references `code/scripts`. [Serial/parallel equivalence](experiments/2026-11-22/pose-robustness-v2/parallel-equivalence.json) checks the faster evaluator against all eight serial development trajectories. Do not substitute the joint-space commands below for this pose-policy experiment.

## Historical joint-space pilot

## Environment and input data

The AutoDL runtime uses Python 3.12.3, PyTorch 2.8.0+cu128, MuJoCo 3.2.2, dm_control 1.0.22, and EGL rendering. Keep dependencies, model weights, datasets, and the OpenPI cache outside Git. Add RLinf, the OpenPI `src` and `packages/openpi-client/src` directories, VLABench, and RRT Algorithms to `PYTHONPATH`. Set `VLABENCH_ROOT` to the assets-bearing `VLABench` package directory, `MUJOCO_GL=egl`, and `JAX_PLATFORMS=cpu` during conversion. The pinned OpenPI tokenizer uses the public `gs://big_vision/paligemma_tokenizer.model` object. Cache it at `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`.

The training reader uses the LeRobot v2.1 export plus its generated `verification.json`. It samples tasks evenly, chooses a random episode and frame for that task, and constructs a 50-action window within that episode. It encodes only the forward and wrist images, prompt, and measured joint positions. The 9 absolute actuator targets are normalized with training-only quantiles and padded to 32 model dimensions. The loss uses the nine actuator dimensions. Each rollout executes at most ten actions before reading the scene again.

`pack_policy_views.py` is an optional two-view transfer format. The local check decoded all 60 packed episodes and matched their RGB hashes. A separate reconstruction check matched all policy inputs and action labels at three frames in each of the three sampled tasks. The full four-view export stays the canonical LeRobot dataset.

## Measured integration checks

| Test | Observation |
| --- | --- |
| Single GPU with three real task traces | The reader loaded all three tasks, each with a 50 × 9 action window. |
| SFT runtime on randomly initialized π0.5 | Measured gradient L2 of 7.60 and maximum action-policy weight change of 5.00e-5. Peak allocated CUDA memory was 9.47 GiB. |
| Pi0 to Pi0RL on randomly initialized π0.5 | Identical sampled action tensors with a fixed observation and noise seed after checkpoint reload. |
| Official simulator with joint holds | Three tasks reset, rendered, and completed 12 control steps each. |
| PPO integration on randomly initialized π0.5 | 22 fruit control steps, matching rollout and recomputed log probabilities, and maximum action-policy weight change of 3.00e-6. Official fruit success was false. |

These measurements cover integration fixtures. The comparison below uses actual pretrained weights.

## Run the experiment

Use the [simulator and asset setup](stage1.md#setup-and-run) and the [three-task training export](stage2.md). The tested package versions are in the [GPU environment snapshot](experiments/2026-11-21/gpu-python-freeze.txt). The runtime uses PyTorch 2.8.0+cu128 from the hosting image and an isolated Python environment for the project packages. Download the official `gs://openpi-assets/checkpoints/pi05_base` Orbax checkpoint from OpenPI before conversion. Keep the Orbax directory and converted weights in separate directories. The converted weight file requires 13.4 GB.

Set `PYTHON`, `BASE_JAX`, `BASE_PT`, `DATASET`, and `RUN` to absolute paths. `DATASET` is the parent of the `train`, `train-stats`, and `verification.json` entries. `RUN` must be new. From the repository root, configure the source paths to point at the pinned checkouts and official assets used for this run.

```bash
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:$RLINF:$VLABENCH:$RRT_ALGORITHMS"
export VLABENCH_ROOT="$VLABENCH/VLABench"
export OPENPI_DATA_HOME="$PWD/.local/cache/openpi"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

```bash
"$PYTHON" -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
  --input-model "$BASE_JAX" --input-norm-stats "$BASE_JAX/assets/franka/norm_stats.json" \
  --output-model "$BASE_PT" --output-norm-stats "$BASE_PT/upstream-franka-norm-stats.json" \
  --action-dim 32 --action-horizon 50 --max-token-len 200

STATS="$DATASET/train-stats/norm_stats.json"
"$PYTHON" scripts/train_pi05.py --train "$DATASET/train" --manifest "$DATASET/verification.json" \
  --stats "$STATS" --base "$BASE_PT/model.safetensors" --output "$RUN" \
  --updates 900 --seed 240 --lr 5e-5

bash scripts/run_policy_comparison.sh "$PYTHON" "$BASE_PT/model.safetensors" \
  "$STATS" "$RUN/sft.pt" "$RUN/comparison"
```

SFT updates the two action projections and two timestep MLP layers. PPO starts from the resulting policy parameters, trains the same four modules and a value head, and records the policy gradient and parameter difference. Initial short tests use batch size one. The additional value head has independent parameters. The development comparison runs both policy classes from matching initial states and checks matching action hashes. The comparison stops before PPO if SFT succeeds on none of the four fruit development starts. These starts are used to select a checkpoint. The final comparison uses a separate test suite.

The supplied `pi05_base` Franka statistics have eight dimensions, with one gripper coordinate. VLABench uses nine absolute targets. The base comparison runs the untouched weights through this project's nine-coordinate, train-normalized adapter. Report it as **base weights with the VLABench adapter**. The numbers characterize the transfer baseline under the shared evaluation protocol.

For fruit PPO, the task success condition supplies a terminal reward. The optional shaping term measures progress toward reaching the fruit and moving it toward the plate. PPO uses these simulator positions only for reward calculation. The released policy inputs and official success metric are the same across stages. Report the reward variant and PPO environment steps with the held-out results.

## Resume evaluation

Use the same input weights, statistics and output directory to continue a run. The evaluation command checks the logged checkpoint and preprocessing identity before appending cases. `PI05_PARALLEL_EVAL=1` runs all three model stages in separate processes. Use the serial default when GPU memory cannot hold all three. `PI05_EVAL_WORKERS=3` overlaps three simulator rollouts per model stage and shares one loaded policy. It was checked against serial runs on fruit seeds 1002–1004 and reproduced the initial state, camera images, model actions, and outcomes. Full-suite summaries require matching case sets and reject missing cases.

```bash
bash scripts/run_policy_evaluations.sh "$PYTHON" "$BASE_PT/model.safetensors" \
  "$STATS" "$RUN/sft.pt" "$RUN/comparison/ppo/ppo-1.pt" "$RUN/comparison"
```

The PPO path above assumes a completed sparse-reward run. Use `ppo-shaped/ppo-1.pt` if the comparison entered the sparse-reward retry branch. If the matching check fails, keep the original diagnostics and inspect the initial state, four camera views and instruction before comparing policy outcomes.
