# Policy inputs and actions

## Current VLABench primitive pose policy

The active fruit-first run uses the published VLABench OpenPI fork, not the older joint-space mapping below. These same transforms serve the verified LeRobot pose export, SFT and the RLinf actor. [Current experiment](experiments/2026-11-22/fruit-completion-v1/README.md)

| Model slot | Actual VLABench view | Mask |
| --- | --- | --- |
| `base_0_rgb` | `forward` | `true` |
| `left_wrist_0_rgb` | `right` scene camera | `true` |
| `right_wrist_0_rgb` | `franka/Franka_wrist_cam` | `true` |

The slot names are inherited from the upstream transform; they do not name two physical wrist cameras. Policy renders are 480 × 480 and use the pinned resize/transform path to 224 × 224. The pose LeRobot export stores the corresponding verified 224 × 224 views. Display videos separately render four views at 512 × 512 each; the resulting 1024 × 1024 mosaic is never fed back as a policy input. Recorded policy RGB and complete control hashes must still match scored executions.

The observation whitelist is the three RGB arrays, `observation/state`, and the actual instruction in `prompt`. State is `[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper]`. The official adapter subtracts the published robot origin `[0, -0.4, 0.78]` meters from measured world position. It adds the actual robot-frame origin back to predicted position before IK. Original Euler angles use radians; this run retains the original unwrapped Euler-delta convention. The published `discrete_state_input=False` profile is preserved. Measured state remains available to the shared delta transforms and decoder; explicit state-token conditioning is not claimed.

Stored labels are absolute seven-coordinate end-effector pose/gripper targets. The shared OpenPI data transform subtracts the observation's first six state coordinates, applies the fixed published quantile statistics, and pads to 32 model coordinates. Decoding uses those exact statistics and adds measured state back. Gripper is an absolute command, not a distance: the official adapter maps predictions `>= 0.1` to two finger targets of 0.04 m and lower predictions to two zero targets. The unchanged official inverse kinematics produces seven arm joint targets, followed by the two finger targets.

All policy stages predict ten controls and execute at most five before observing again, at 10 Hz. The budget is 600 controls; official success or environment termination ends a rollout earlier. PPO uses action-chunk transitions, gamma 0.99 and GAE lambda 0.95, with zero terminal bootstrap for this finite-horizon task. Simulator target geometry is reward-only and never enters observations. Expert actions are absent at policy inference. Success and termination metadata remain evidence and labels, not model inputs.

The parent SFT normalization SHA-256 is `bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f`. The fresh handoff verifies predictions, shared decoding, sampled/recomputed log probabilities and an entire successful execution before PPO updates.

## Historical joint-space mapping

The same mapping processes LeRobot v2.1 samples and observations from a fresh VLABench simulator. This check runs in the existing WSL environments with local assets and applies the project adapter before OpenPI model transforms.

| Policy slot | VLABench camera | Mask |
| --- | --- | --- |
| `base_0_rgb` | `forward` | `true` |
| `left_wrist_0_rgb` | `franka/Franka_wrist_cam` | `true` |
| `right_wrist_0_rgb` | Zero-filled placeholder | `false` |

The single wrist view uses the slot convention in the pinned OpenPI Libero adapter. The export retains all four source views. The adapter emits 256 × 256 HWC RGB bytes, the original language prompt, and float32 measured joint positions. State and action order is joints 1–7 in radians and finger joints 1–2 in meters. The dataset branch adds absolute actuator targets as training labels. At inference, `actuator_targets` takes the first nine coordinates of each unnormalized 32-coordinate output and preserves the control units.

## Reproduce the local check

Use the [three-task v2.1 train split](stage2.md#three-task-training-and-validation) and provide the source episodes in its stored order. This runs the pinned LeRobot reader and MuJoCo in separate processes.

```bash
bash scripts/verify_policy_io_wsl.sh data/stage2-three-task-train-v21/train \
  .local/policy-io-smoke.json \
  runs/select_fruit-42/episode.h5 \
  runs/stage2/insert_flower-42-retry/episode.h5 \
  runs/stage2/take_out_cool_drink-43-screen-success/episode.h5
```

At the first, middle, and last pre-action frame in each episode, the active RGB pixels, blank image, camera masks, float32 states, and instructions matched exactly. Each of the nine samples also matched its 50-step float32 action window and episode-boundary padding. All 790 original float64 expert actions went through the 32-to-9-coordinate mapping and were executed at 10 Hz. All three replays finished successfully and matched the original MuJoCo state at every step. The [report](policy-io-smoke.json) includes the sample indices and observed state and command ranges.

## Training normalization

The pilot used `openpi.shared.normalize` from OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`. It read just the numeric LeRobot columns and fit nine state and action coordinates on 790 states and 39,500 action labels. Each training frame contributes a 50-step window, with the last action repeated at episode boundaries as in the loader. The validation and test exports are separate from the input directory. Reload the same `norm_stats.json` for SFT and rollout with quantile normalization enabled.

```bash
PYTHONPATH=.local/openpi-norm-src:.local/openpi-norm-deps \
  /var/tmp/pi05-vlabench-$(id -u)/lerobot/bin/python scripts/compute_policy_norm_stats.py \
  data/stage2-three-task-train-v21/train data/stage2-three-task-train-v21/train-stats
```

For this local run, `.local/openpi-norm-src/openpi/shared/normalize.py` holds the pinned OpenPI file and `.local/openpi-norm-deps` holds `numpydantic==1.6.9` from the pinned OpenPI lockfile. The existing LeRobot environment provides NumPy, PyArrow, and Pydantic. In a full OpenPI environment with `pyarrow` and `h5py`, run the script with that environment's `python` in place of the local command. The pilot's `train-stats/norm_stats.json` has SHA-256 `f6baad064029dafad7b902c3bda32bd21d306f8c17a3a6338b6f3840c59c183e`.

The first expansion also passed this fit and reload check on 1,910 states and 95,500 action labels. Its stats file at `data/stage2-expansion-01-v21/train-stats/norm_stats.json` has SHA-256 `ca9975e879b41a7c1d5a4987b0010337adab305e050a27e78c2d1cc7f5fd114f`. Refit the stats on the selected training export before model training.

The second expansion passed the fit and reload check on 3,983 states and 199,150 action labels. Its stats file at `data/stage2-expansion-02-v21/train-stats/norm_stats.json` has SHA-256 `b4a1d347ff44b666d56062c03a65c20a7c55b4e15c238a8c83c3ade5284cd0a0`.

The fifth expansion passed the fit and reload check on 9,758 states and 487,900 action labels. Its stats file at `data/stage2-expansion-05-v21/train-stats/norm_stats.json` has SHA-256 `b729f62d460f36c9f4c3e35f0a2a55f92f25d15d93502109ab1bdd0be4efeab4`.

The pinned LeRobot reader mapped 120 samples from all 40 episodes in the fifth expansion through `lerobot_inputs`. The samples cover the first, middle, and final frame of each episode, including the padded action window at the end.

```bash
bash scripts/convert_lerobot_wsl.sh --openpi --policy-io-samples \
  data/stage2-expansion-05-v21/train "/var/tmp/pi05-vlabench-$(id -u)/expansion-05-policy-inputs.npz"
```

The 60-episode export passed the fit and reload check on 16,520 states and 826,000 action labels. Its `train-stats/norm_stats.json` has SHA-256 `fc89b1b8f11586cd09f963ae9ceb8ef774fa44204905e43ddaa2614a3f3744e2`.

The pinned reader also mapped 180 samples from the final 60 episodes through the same input adapter.

```bash
bash scripts/convert_lerobot_wsl.sh --openpi --policy-io-samples \
  data/stage2-expansion-06-serial-v21/train "/var/tmp/pi05-vlabench-$(id -u)/expansion-06-policy-inputs.npz"
```

## GPU verification after expert and dataset validation

After expert screening and dataset expansion, refit training normalization and register the shared adapter in the OpenPI SFT and RLinf rollout paths. Compare the final tensors after resizing, normalization, padding, and tokenization. The 50-step label horizon is verified locally. Select and measure the execution length during the GPU smoke test.
