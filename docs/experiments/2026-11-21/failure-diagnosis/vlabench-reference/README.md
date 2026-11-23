# Published VLABench π0.5 reference

The [published checkpoint](https://huggingface.co/VLABench/pi05-primitive-10task/tree/f658a8bec801aab960fe0ceab370a123b31c1837) uses the [VLABench OpenPI fork](https://github.com/Shiduo-zh/openpi/tree/788ef1f20e655de5bb40869fb7b7fcdd49732387) and configuration `pi05_ft_vlabench_primitive`. The simulator revision was `cf588fe60c0c7282174fe979f5913170cfe69017`. The download included `params/`, `assets/`, and checkpoint metadata, totaling 12,440,879,191 bytes. It used a direct connection to the Hugging Face mirror after a direct connection to `huggingface.co` timed out. Each large shard was checked against its LFS SHA-256 and each small file against its Git blob hash.

The four development resets are the same as the [expert trials](../fruit-development-expert.json). Before stepping, the runner checks the initial physics hash and the instruction. The robot runs at 10 Hz for at most 600 controls per episode. The rollout uses the simulator's task predicate to determine success. Target displacement and lift are recorded after each step for diagnosis. The policy receives three RGB views, measured gripper pose and state, and the task instruction.

The published adapter reads the forward, right, and wrist cameras at 480 × 480 and the upstream model transform resizes the images. The policy predicts ten relative gripper poses with the corresponding gripper command. It executes five poses before replanning, converts each pose to a joint target with official inverse kinematics, and takes one environment step per target. The published configuration tokenizes the instruction without adding discretized robot state. The absolute action transform uses the measured starting pose to decode the model's relative action. The earlier end-effector SFT pilot used two views, tokenized robot state, executed ten poses per plan, and held the current joints if inverse kinematics failed. This reference test changes these together with the checkpoint.

A [stationary-action fixture](io-fixture.jsonl) exercised the upstream adapter on all four matched resets. Each trial executed six controls and requested two action chunks. A separate [GPU preflight](gpu-preflight.log) restored the existing OpenPI base weights and ran the same adapter for six controls. The preflight verifies loading and simulator integration.

## Results

The published reference completed 3/4 fruit development episodes. The official expert completed 4/4 and the expanded-data SFT checkpoint completed 0/4. The reference and SFT matched initial physics, the initial RGB render used for auditing, and instructions for all four resets. The two policies used their respective observation and action adapters. The four cases were used for development.

| Seed | Instruction | Official outcome | Simulator controls |
| ---: | --- | --- | ---: |
| 1201 | Put the banana into the plate_seen | Success | 134 |
| 1202 | Put the pear into the plate_seen | Success | 144 |
| 1203 | Put the orange into the plate_seen | Step budget | 600 |
| 1204 | Put the pear into the plate_seen | Success | 104 |

On the orange trial, the reference reached the 600-step cap. Inverse kinematics reported 199 nonconverged targets and the target lifted at most 2.6 mm. The three successful trials had zero nonconverged targets. [Per-episode measurements](episodes.jsonl) · [Verified checkpoint files](download-verified.json)

## Adapter checks

We replayed seed 1201 with the same published weights, reset, sampler seed, and step budget. Each trial changed one interface setting. Executing ten controls per prediction ended at the budget. Masking the right-hand scene view and continuing to execute five controls also ended at the budget. The three-camera, five-control run finished in 134 controls. These two single-seed checks inform the adapter settings for the next SFT run. [Adapter trials](adapter-ablation.jsonl)

We also replayed all 132 original controls for training seed 42. The live robot states exactly matched the original recording. At frame 34 we rendered the three upstream views and encoded ten expert actions through the published model configuration. Moving the labels from absolute poses to relative poses and back introduced at most 2.4e-7 error. [Training-window check](training-window-preflight.json)

## Run the paired evaluation

Use the pinned VLABench simulator and OpenPI fork with JAX 0.5.3, its CUDA 12 plugin, and the simulator's MuJoCo EGL setup. Put the published fork's `src` and `packages/openpi-client/src` first among OpenPI paths on `PYTHONPATH`. The evaluator expects the verified download manifest at `download-verified.json` inside the checkpoint directory.

```bash
python scripts/evaluate_vlabench_reference.py \
  --fork "$REFERENCE_OPENPI" \
  --checkpoint "$REFERENCE_CHECKPOINT" \
  --paired-episodes docs/experiments/2026-11-21/failure-diagnosis/fruit-development-expert.json \
  --output-dir runs/vlabench-reference-fruit-development --video
```
