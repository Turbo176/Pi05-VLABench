<h1 align="center">Pi05-VLABench</h1>

<p align="center">
  <strong>π0.5 imitation learning and reinforcement learning in VLABench</strong><br>
  Expert demonstrations · LeRobot datasets · Multitask SFT · PPO · Closed-loop evaluation
</p>

<p align="center">
  <a href="#demos">🎬 Demos</a> ·
  <a href="#results">📊 Results</a> ·
  <a href="#training">📈 Training</a> ·
  <a href="#pipeline">🧩 Pipeline</a><br>
  <a href="#quickstart">🚀 Quickstart</a> ·
  <a href="#experiments">🧪 Experiments</a> ·
  <a href="#credits">📚 Credits</a>
</p>

A reproducible robot-learning project built around official VLABench tasks and controllers. The completed fruit-selection workflow connects verified expert data, a shared fruit/flower π0.5 SFT checkpoint, autonomous evaluation, and PPO updates to the same policy.

The current experiment uses **64 demonstrations, 9,904 training frames, and 90 SFT updates**. PPO updates the action-expert LoRA and action heads; a zero-update handoff checks that loading the SFT model into RL preserves its behavior. Flower insertion and fridge retrieval have expert demonstrations, while their learned-policy workflows remain unfinished.

<a id="demos"></a>

## 🎬 Autonomous fruit selection

Same held-out scene, seed 1501, with the instruction **“Put the pear into the plate_seen.”** Both clips show fresh autonomous rollouts.

| Shared multitask SFT | PPO from the same SFT checkpoint |
| :---: | :---: |
| [![SFT places the pear on the plate](media/policy/fruit-completion-v1-sft-1501.gif)](media/policy/fruit-completion-v1-sft-1501.mp4?raw=true) | [![PPO places the pear on the plate](media/policy/fruit-completion-v1-ppo-1501.gif)](media/policy/fruit-completion-v1-ppo-1501.mp4?raw=true) |
| Success · 160 controls | Success · 160 controls |

[Reference rollout on the same scene](media/policy/fruit-completion-v1-base-1501.mp4?raw=true) · [Video verification](docs/experiments/2026-11-22/fruit-completion-v1/local-video-verification.json)

MP4s preserve 10 Hz timing at 1024 × 1024; GIF previews play at 2× speed. Each recording matches the scored episode's initial policy images and complete action hash. These clips illustrate one scene; aggregate results follow below.

<details>
<summary>🦾 Official expert action replays: three different skills</summary>

These recordings execute saved official-expert commands in a fresh MuJoCo simulation.

| 🍎 Fruit selection | 🌷 Flower insertion | 🧃 Fridge drink retrieval |
| :---: | :---: | :---: |
| [![Expert fruit replay](media/demos/select-fruit-preview.gif)](media/demos/select-fruit-replay.mp4?raw=true) | [![Expert flower replay](media/demos/insert-flower-preview.gif)](media/demos/insert-flower-replay.mp4?raw=true) | [![Expert fridge replay](media/demos/fridge-drink-preview.gif)](media/demos/fridge-drink-replay.mp4?raw=true) |
| [Replay audit](docs/stage1-action-replay.json) | [Replay audit](docs/stage2-insert-flower-replay.json) | [Replay audit](docs/stage2-fridge-rgb-replay.json) |

The gallery belongs to the original local joint-space dataset: 60 successful episodes / 16,520 frames. The current SFT uses the separately audited pose-policy dataset above. [Dataset records](docs/stage2.md)

</details>

<a id="results"></a>

## 📊 Closed-loop results

Ten fixed held-out fruit scenes, seeds 1501–1510. Every policy uses the same initialization, policy cameras, instruction, official Franka controller, and 600-control budget. Control runs at 10 Hz, predicting ten actions and executing five per observation.

| Policy | Official successes | Success rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Published VLABench reference | 4/10 | 40% | 16.8–68.7% |
| Shared multitask SFT90 | 5/10 | 50% | 23.7–76.3% |
| Fruit PPO1 after SFT90 | 5/10 | 50% | 23.7–76.3% |

The reference is a published VLABench π0.5 model already fine-tuned upstream. SFT gained one success overall in this sample; the single PPO iteration left aggregate success unchanged. Ten scenes give wide uncertainty. Paired language tests also expose a PPO regression, documented with every prompt and outcome in the [full evaluation report](docs/experiments/2026-11-22/fruit-completion-v1/README.md).

PPO used **731 environment controls, 147 action chunks, and 19 optimizer steps**. Saved-checkpoint comparison verifies changes to **224 actor tensors**, including action-expert LoRA and action heads. [Handoff check](docs/experiments/2026-11-22/fruit-completion-v1/handoff.json) · [Actor update audit](docs/experiments/2026-11-22/fruit-completion-v1/saved-checkpoint-update-audit.json) · [All paired results](docs/experiments/2026-11-22/fruit-completion-v1/ordinary-comparison.json)

<a id="training"></a>

## 📈 Training signals

![Complete SFT loss history and actor gradients](docs/experiments/2026-11-23/training-and-motion/sft-training.png)

All 90 SFT updates are shown. Thin lines show recorded values; thick lines show a ten-update trailing mean. Loss varies across sampled minibatches, so closed-loop evaluation remains the performance measure.

[Training logs and interpretation](docs/experiments/2026-11-23/training-and-motion/README.md) · [Command and measured-motion analysis](docs/experiments/2026-11-23/training-and-motion/motion-audit.png) · [Reward design and literature](docs/fruit-reward-review.md)

<a id="pipeline"></a>

## 🧩 Data and policy pipeline

**Official expert → action replay → LeRobot export → shared SFT → closed-loop evaluation → PPO → paired evaluation**

- **Data validation.** Replay saved commands from the recorded initialization, check official success, and isolate complete episodes and initialization conditions across splits.
- **Policy interface.** Use forward, right, and wrist RGB plus the literal instruction. Measured end-effector pose and gripper state support the shared transforms and action decoder. Expert target geometry, reward, and success labels stay outside policy inputs.
- **Action alignment.** Preserve the upstream state-conditioning setting, normalization, pose decoding, ten-action horizon, and five-control execution rule across SFT and RL. The zero-update handoff compares predictions, log probabilities, and an entire closed-loop rollout.
- **Experiment records.** Retain checkpoint hashes, frozen configurations, source snapshots, raw outcomes, actor gradients, and parameter changes.

The pose-policy actions contain XYZ in meters, Euler angles in radians, and a gripper coordinate, decoded through the official controller. The published configuration uses `discrete_state_input=False`; state supports pose decoding rather than adding a new learned state input. [Policy I/O](docs/policy-io.md) · [Current experiment settings](docs/experiments/2026-11-22/fruit-completion-v1/README.md)

<a id="quickstart"></a>

## 🚀 Run an expert demonstration

The commands below were verified on Ubuntu 22.04 under WSL2. Prepare the pinned simulator, official asset subset, and independent Python environment using the [setup guide](docs/stage1.md#setup-and-run). Run from the repository root with a fresh output directory.

```bash
bash scripts/run_wsl.sh probe --task select_fruit --seed 42
bash scripts/run_wsl.sh collect --task select_fruit --seed 42 --output runs/select_fruit-42
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5
```

`expert.mp4` records the expert. `action_replay.mp4` records a fresh simulation executing saved actions. Model training and evaluation require the separate GPU setup in the experiment report.

<a id="experiments"></a>

## 🧪 Experiments and reproduction

| Record | Contents |
| --- | --- |
| [Fruit SFT → PPO](docs/experiments/2026-11-22/fruit-completion-v1/README.md) | Frozen checkpoints, executed commands, evaluation, costs, and raw evidence |
| [Training and motion analysis](docs/experiments/2026-11-23/training-and-motion/README.md) | Loss and gradient plots, time-aligned motion measurements, reproducible plotting command |
| [Reward review](docs/fruit-reward-review.md) | Actual reward, literature, and an untrained staged-reward proposal |
| [Expert data and LeRobot export](docs/stage2.md) | Collection, replay checks, dataset fields, and split audits |
| [Earlier experiments](docs/experiments/2026-11-22/pose-robustness-v2/README.md) | Previous interfaces, diagnostic runs, and their separate results |

`scripts/` contains collection, conversion, training, PPO, evaluation, and analysis tools; `configs/` records experiment settings. Curated videos live in `media/`. Weights, source datasets, assets, caches, and credentials remain outside Git.

<a id="credits"></a>

## 📚 Credits and license

Built on VLABench, MuJoCo, LeRobot, OpenPI, and RLinf. Project code uses the MIT license; assets and model checkpoints retain their upstream terms. [Pinned revisions and attribution](docs/upstream.md) · [MIT license](LICENSE)