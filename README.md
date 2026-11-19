# Pi05-VLABench

A staged project for multitask π0.5 imitation learning, closed-loop generalization evaluation, and PPO post-training on VLABench.

## Verified locally

The official `select_fruit` expert picked and placed an apple with the official Franka robot in MuJoCo. Seed 42 produced one successful 132-step episode at 10 Hz. A separate process rebuilt the episode, fed all 132 saved actuator commands into the simulator, and reached the official success condition. The maximum difference across the per-step MuJoCo position and velocity arrays was 0 on this machine.

| Video | Source | Result |
| --- | --- | --- |
| `expert.mp4` | Official VLABench expert | 1/1 smoke-test episode |
| `action_replay.mp4` | Saved actuator commands executed in a fresh MuJoCo instance | 1/1 replay of the same episode |

See the [local runbook](docs/stage1.md) for commands, asset scope, and action format. See [upstream revisions](docs/upstream.md) for sources, licenses, and component status.

## Remaining milestones

1. Verify additional official skills and convert a small sample to LeRobot Dataset.
2. Train a shared π0.5 checkpoint and evaluate per task on held-out initializations and language variants.
3. Verify the SFT checkpoint in the RL environment before any PPO update and measure the effect of updating policy weights.
