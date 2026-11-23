# Fruit SFT with the published VLABench interface

This pilot adapts the original π0.5 base checkpoint on `select_fruit`. It uses the official VLABench simulator at `cf588fe60c0c7282174fe979f5913170cfe69017`, OpenPI transforms at `788ef1f20e655de5bb40869fb7b7fcdd49732387`, and the local RLinf PyTorch backend at `a29c0a6c37b02881b8d4af0900fa97f35e44b34c`. See the [published reference results](../vlabench-reference/README.md) for the separate pre-finetuned reference.

## Data and model interface

The pilot starts with the 44 successful fruit demonstrations verified during LeRobot v2.1 export. It re-executes the original float64 controls and checks the reset, measured state, timestamp, post-step physics, and official terminal success. For each state before a control, it renders forward, right, and wrist views at 480 × 480. The OpenPI resize produces three 224 × 224 images stored in the derived HDF5 cache.

The seven action coordinates are robot-frame XYZ in metres, XYZ Euler angles in radians, and a binary gripper command. The conversion uses forward kinematics on saved expert joint targets. Each ten-step chunk is relative to the measured pose at its first frame. The episode's final action pads windows near termination. Pose conversion can change the joint-space trajectory when the evaluator solves inverse kinematics again, so a separate replay audit requires the converted training labels to reach official success.

Seeds 55, 69, 2012, and 2027 are reserved for offline action validation and excluded from fitting. Development reset hashes for seeds 1201–1204 are checked against the expert and the published reference traces. The model receives only the three images and instruction. The measured pose is used to define the relative action and decode the predicted absolute pose. Action decoding matches the reference adapter's five-control execution at 10 Hz, with official inverse kinematics and a 600-control budget.

All 44 saved joint-control replays reached official success. The 6,596 post-step physics states matched exactly. Replaying the converted pose and binary gripper labels through the reference adapter succeeded in 28 of 44 episodes. The SFT split kept 26 eligible episodes with 3,907 frames after reserving validation seeds. The four validation episodes contain 581 frames. [Saved joint-control replays](control-replay.json) · [Converted label replays](label-audit.jsonl)

In a four-seed follow-up on failed conversions, the pose targets with the original finger commands succeeded on seeds 55, 72, and 75. The original arm commands with binary finger targets failed on all four seeds. The benchmark adapter keeps the published binary grip command for both SFT pilots. [Gripper replay measurements](gripper-audit.log)

## Verified GPU preflight

On one RTX 4080 SUPER, a real flow-matching update changed actor parameters by up to `1.0061e-4`. The actor gradient norm was `10.44` before clipping. After zeroing the trainable parameters and restoring the checkpoint and optimizer state, the same observation and noise gave identical actions. The normalized network output was identical when the measured state was perturbed. The maximum action normalization round-trip error was `2.38e-7`. [Smoke measurements](smoke.json)

## Action validation

| Policy | XYZ MAE (cm) | Euler MAE (rad) | Gripper MAE |
| --- | ---: | ---: | ---: |
| Base weights with data normalization | 3.14 | 0.297 | 0.633 |
| SFT with the upstream Euler deltas | 2.04 | 0.216 | 0.157 |

This table uses 24 windows across the four offline validation starts. Each row covers the first five targets per window. The 600-update SFT took 146.5 seconds for optimizer steps and used 9.55 GiB peak PyTorch allocation. The checkpoint reloaded with exact action agreement for a fixed input and noise. [Training summary](summary.json) · [Optimizer steps](updates.jsonl)

The training targets had 39,070 predicted control slots. For 13.6% of slots, the roll delta magnitude exceeded π due to the Euler angle branch boundary. The alternate run passes equivalent Euler deltas in `[-π, π)` into normalization and training. [Angle audit](angle-audit.json) · [Alternate run summary](short-angle-summary.json)

## Paired development rollouts

All four seeds ran under the same official success condition and 600-control budget. The two local SFT runs started from the original base weights. The published VLABench reference starts from upstream fine-tuned weights.

| Policy | Official successes | IK nonconvergence events |
| --- | ---: | ---: |
| Base with upstream action representation | [0/4](base-episodes.jsonl) | 745 |
| SFT, 600 updates, upstream Euler deltas | [0/4](sft-600-episodes.jsonl) | 606 |
| SFT, 600 updates, shortest equivalent Euler deltas | [0/4](short-angle-600-episodes.jsonl) | 0 |
| SFT, 2,400 updates, shortest equivalent Euler deltas | [0/4](short-angle-2400-episodes.jsonl) | 701 |
| Native OpenPI SFT, 1,200 updates | [0/4](native-1200-episodes.jsonl) | 1,060 |
| Published VLABench reference, native JAX | [3/4](../vlabench-reference/episodes.jsonl) | 199 |

The short-angle SFT reduced Euler MAE from 0.0061 to 0.0040 radians relative to the base evaluated with the same short-angle normalization. XYZ MAE fell from 3.14 to 2.01 cm. The four rollouts reached the step cap and the target lifted at most 9 mm. [Short-angle optimizer steps](short-angle-updates.jsonl)

## Backend check

The published reference finished seed 1201 in 134 controls in native OpenPI. Its converted weights in RLinf reached 600 controls with 97 nonconverged IK targets. The converted policy lifted the banana by 17.6 cm and moved it 62.6 cm from its initial position. The episode failed the official success predicate. A separate first-window test shared the three images, prompt, robot pose, normalization and action noise. The XYZ MAE between implementations was 0.067 mm. The first-window check ran JAX on CPU and PyTorch on CUDA, while the closed-loop runs used CUDA. [Converted rollout](converted-reference-1201.jsonl) · [First-window check](backend-first-window.json)

A native JAX smoke run fitted the action-expert LoRA and action/time projections from the original base. One optimizer step produced a gradient norm of `13.96` and a maximum actor delta of `1.0002e-4`. The restored adapter produced identical actions for the fixed input and noise. [Native preflight](native-smoke.json)

## Demonstrations collected with the pose interface

The official pick and place skills were rerun in the 14 excluded training scenes. They planned with the existing simulator and executed Cartesian targets through the evaluator's inverse kinematics and binary gripper command. All 14 scenes reached the official goal and replayed to success through that interface. The LeRobot v2.1 export contains 40 training episodes with 6,054 frames and four validation episodes with 581 frames. LeRobot readback verified every frame and the ten-action windows at three positions per episode. The policy consumes three RGB arrays, a measured gripper pose and an instruction. The training loader reads action targets separately. [New pose replays](direct-recovery-audit.jsonl) · [Dataset verification](lerobot-full-verification.json)

One training step on the small LeRobot export verified the opt-in `--discrete-state` variant. It tokenizes the robot state together with the instruction. The fixed-input audit in the RLinf actor used the step's SFT checkpoint. The three sampled windows produced identical normalized actions and PPO replay log probabilities before any RL update. [LeRobot SFT step](lerobot-smoke.json) · [RL actor audit](rl-handoff-smoke.json)

## LeRobot SFT with robot state tokens

The RTX 4080 SUPER run fit 40 fruit episodes from the LeRobot v2.1 export for 1,800 updates. Four offline validation episodes supplied 24 action windows. XYZ mean absolute error fell from 3.63 cm before SFT to 1.47 cm after SFT. The gripper error fell from 0.620 to 0.148. Training took 521 seconds and used 9.55 GiB peak PyTorch allocation. The checkpoint reload reproduced its fixed-input actions exactly. [Training summary](lerobot-1800/summary.json)

The base and SFT checkpoints both reached official success in **0/4** development episodes. Both ran on the same four resets, used the same normalization and sampler seeds, and had the same 600-control budget. Inverse kinematics returned 1,533 nonconvergence events for the base and 457 for SFT across 2,400 controls per checkpoint. On seed 1201, SFT first grasped the banana at step 131, lifted it by 83.1 cm and opened the gripper at step 276. The banana was 88.9 cm horizontally from the plate center when the gripper opened. The other three episodes registered zero target grasps. [Base records](lerobot-1800/base-episodes.jsonl) · [SFT records](lerobot-1800/episodes.jsonl) · [Banana trace summary](lerobot-1800/banana-trace-summary.json)

The trained checkpoint produced identical normalized predictions and PPO log probabilities on all three fixed-input windows after loading into the RLinf actor. Four full rollouts matched the SFT actuator command hashes and outcomes. The actor performed zero PPO updates. [Fixed-input audit](lerobot-1800/rl-handoff-1800.json) · [Matched actor rollouts](lerobot-1800/rl-handoff-episodes.jsonl)

## Three-task pose data preflight

The same three-camera pose interface replayed successful expert episodes for fruit seed 42, flower seed 42, and fridge seed 43 through the published five-control adapter. The training split of the smoke export contains 790 verified frames. Flower seed 55 supplied 155 separate validation frames. Original binary pose label replays also passed for flower seed 55 and fridge seed 52. [LeRobot export audit](lerobot-1800/verification.json) · [Adapter replay records](lerobot-1800/adapter.log)

Repeating flower seed 1005 six times across two AutoDL processes gave one physics hash and one instruction. The left camera produced two hashes at 256 × 256 and three at 480 × 480. All three policy cameras stayed byte-identical at both sizes. New rollout records preserve the full-camera hash and also hash the cameras used for inference. The paired summary requires matching policy-camera hashes when both inputs provide them. Reports with older records still require matching full-camera hashes. [Reset audit](lerobot-1800/flower-1005-resets.jsonl)

## Reproduce the pilot

Use the repository's pinned simulator and RLinf runtime together with the published OpenPI fork. Prepend the fork's `src` and `packages/openpi-client/src` to `PYTHONPATH` and include this project's `scripts` directory. `ORIGINAL_VERIFICATION` and `EXPANSION_VERIFICATION` are the two source LeRobot verification manifests. Run the packing command where the original HDF5 trajectories live. The compact controls bundle can be copied to the machine with the simulator. The export commands below expect the expansion recordings at the paths in their manifest.

```bash
python scripts/pack_fruit_replay_controls.py \
    --verification "$ORIGINAL_VERIFICATION" --output "$CONTROL_BUNDLE"

mkdir -p "$CACHE"
for seed in 42 53 54 55 56 57 58 {63..75}; do
    python scripts/export_reference_fruit_views.py \
        --verification "$ORIGINAL_VERIFICATION" --controls "$CONTROL_BUNDLE" \
        --seed "$seed" --output-dir "$CACHE" || exit 1
done
for seed in {2010..2033}; do
    python scripts/export_reference_fruit_views.py \
        --verification "$EXPANSION_VERIFICATION" \
        --seed "$seed" --output-dir "$CACHE" || exit 1
done
python scripts/check_reference_fruit_label_replay.py "$CACHE"/*.h5 > "$LABEL_AUDIT"

python scripts/train_reference_aligned_fruit.py \
    --base "$BASE_SAFETENSORS" --fork "$REFERENCE_OPENPI" --cache "$CACHE" \
    --paired-episodes "$REFERENCE_EPISODES" --smoke --output runs/reference-aligned-smoke
python scripts/train_reference_aligned_fruit.py \
    --base "$BASE_SAFETENSORS" --fork "$REFERENCE_OPENPI" --cache "$CACHE" \
    --paired-episodes "$REFERENCE_EPISODES" --label-audit "$LABEL_AUDIT" \
    --updates 600 --output runs/reference-aligned-fruit-600 --compare-base --video
```

`REFERENCE_EPISODES` points to the published reference's paired `episodes.jsonl` in the [reference report](../vlabench-reference/README.md). `BASE_SAFETENSORS` points to the project's conversion of the original OpenPI base weights. The base and SFT development rollouts share the same normalization, reset checks, instruction, Torch noise seed, and control budget.

### LeRobot state-token run

`DATASET` is the verified LeRobot pose export. `RUN` is a fresh output directory. The second command uses the trained run to load the identical preprocessing and LoRA architecture with the base weights.

```bash
python scripts/train_reference_aligned_fruit.py \
    --base "$BASE_SAFETENSORS" --fork "$REFERENCE_OPENPI" --lerobot "$DATASET" \
    --paired-episodes "$REFERENCE_EPISODES" --shortest-euler --discrete-state \
    --updates 1800 --output "$RUN" --video
python scripts/train_reference_aligned_fruit.py \
    --base "$BASE_SAFETENSORS" --fork "$REFERENCE_OPENPI" --load-run "$RUN" \
    --paired-episodes "$RUN/episodes.jsonl" --eval-stage base --output "$BASE_EVAL" --trace-controls
```

The smoke export was replayed through the same adapter with all three official tasks:

```bash
python scripts/check_reference_pose_adapter.py \
    --fork "$REFERENCE_OPENPI" --dataset "$TRITASK_SMOKE_DATASET"
```
