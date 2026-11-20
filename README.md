# Pi05-VLABench

Multitask imitation learning and closed-loop generalization on VLABench, with π0.5 fine-tuning and PPO post-training as the next milestones.

## Tasks and results

I ran the official VLABench expert and Franka robot in MuJoCo under WSL2. The training split contains 60 successful, replay-verified demonstrations at 10 Hz. The export preserves four RGB views, the original instruction, nine measured joint positions, and nine absolute joint targets.

| Task | Expert screening | Accepted training episodes | Training frames |
| --- | ---: | ---: | ---: |
| Fruit selection | 22 / 22 (100%) | 20 | 3,031 |
| Flower insertion | 22 / 78 (28.2%) | 20 | 3,482 |
| Fridge drink retrieval | 21 / 55 (38.2%) | 20 | 10,007 |

Expert screening counts distinct task and seed pairs across the pilot and six expansions. A seed counts as successful if a recorded retry succeeded. The screening stopped as tasks reached their collection targets, so the table describes local collection. See the [pilot and expansion logs](docs/stage2.md#expert-screening) for failures and the [final dataset audit](docs/stage2-expansion-06.json) for coverage and split checks.

The simulator harness executed a short joint-position hold in all three tasks to verify the step budget. It also replayed two successful expert traces from the same fruit scene. The pear and banana goals finished in 151 and 136 steps, respectively, and both final physics states matched the source traces. The [smoke report](docs/closed-loop.md) records which action source each run used.

| Milestone | Status |
| --- | --- |
| Expert collection, fresh action replay, LeRobot export | Verified locally |
| Model input mapping, action execution, official task scoring | Verified locally |
| Base π0.5 and multitask SFT closed-loop scores | Pending GPU validation |
| SFT checkpoint handoff and π0.5 PPO | Pending GPU validation |

## Reproduce the fruit demo

Prepare the pinned simulator checkout, official asset subset, and WSL environment as described in the [local setup guide](docs/stage1.md#setup-and-run). From the repository root in WSL, run the following in a fresh run directory.

```bash
bash scripts/run_wsl.sh probe --task select_fruit --seed 42
bash scripts/run_wsl.sh collect --task select_fruit --seed 42 --output runs/select_fruit-42
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5
```

`expert.mp4` shows the upstream expert. `action_replay.mp4` shows a fresh simulator executing its saved actuator commands.

## Project layout

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Expert capture, action replay, LeRobot export, and local evaluation |
| [`configs/`](configs/) | Held-out evaluation seeds and paired fruit prompts |
| [`docs/`](docs/) | Runbooks, measured results, and source audit |

Start with the [LeRobot export guide](docs/stage2.md), [policy input and normalization checks](docs/policy-io.md), or [closed-loop evaluation runbook](docs/closed-loop.md). The [SFT to PPO audit](docs/sft-rl-compatibility.md) follows checkpoint, camera, normalization, and controller interfaces at the pinned revisions. The first GPU milestone is one real training batch, checkpoint save and reload, and a short closed-loop rollout.

## Dependencies and license

Built on VLABench, MuJoCo, LeRobot, OpenPI, and RLinf. See [upstream revisions and attribution](docs/upstream.md). Project code is [MIT-licensed](LICENSE). Downloaded assets and model checkpoints retain their upstream terms.
