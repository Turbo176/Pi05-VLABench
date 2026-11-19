# Pi05-VLABench

A staged project for multitask π0.5 imitation learning, closed-loop generalization evaluation, and PPO post-training on VLABench.

## Verified locally

The official VLABench expert and Franka robot ran in MuJoCo under WSL2. The `select_fruit` seed 42 and `insert_flower` seed 42 captures reached the official success condition in 132 and 181 steps. Re-executing their saved actuator commands in fresh simulators reached the same result with zero difference in the recorded MuJoCo position and velocity arrays.

| Video | Source |
| --- | --- |
| `expert.mp4` | Official VLABench expert |
| `action_replay.mp4`, `action_replay_hd.mp4` | Saved actuator commands executed in fresh MuJoCo |

The two-task LeRobot pilot has six successful demonstrations and 937 action-aligned frames at 10 Hz. Each split contains a distinct episode from both tasks. The exports preserve four RGB views, the original instruction, nine measured joint positions, and nine absolute actuator targets. The official `take_out_cool_drink` expert adds a 477-step demonstration of opening and closing a refrigerator. The three-task training and validation splits have 790 and 832 frames, respectively, in both the v2.1 and v3.0 exports. The 300-frame test split covers fruit and flower. The expert screening records cover successful, unsuccessful, and timed-out runs.

- [Local simulator and replay commands](docs/stage1.md)
- [Expert screening and LeRobot export commands](docs/stage2.md)
- [SFT to PPO source audit](docs/sft-rl-compatibility.md)
- [Pinned upstream revisions and licenses](docs/upstream.md)

## Remaining milestones

1. Confirm the third task across more initializations and grow the dataset.
2. Train one π0.5 checkpoint across tasks and evaluate on held-out initializations and language variants.
3. Verify the SFT policy in the RL environment before the first PPO update and measure policy changes and task success after training.
