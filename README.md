<h1 align="center">Pi05-VLABench</h1>

<p align="center">
  <strong>Multitask robot learning with π0.5 on VLABench</strong><br>
  Official expert demonstrations · LeRobot datasets · Closed-loop evaluation
</p>

<p align="center">
  <strong>🧭 Contents</strong><br>
  <a href="#demos">🎬 Demos</a> ·
  <a href="#data-pipeline">🧩 Data pipeline</a> ·
  <a href="#tasks-and-results">📊 Results</a><br>
  <a href="#reproduce-the-fruit-demo">🚀 Quickstart</a> ·
  <a href="#project-layout">📂 Project layout</a> ·
  <a href="#dependencies-and-license">📚 Credits</a>
</p>

<a name="demos"></a>

## 🎬 Demos

A Franka arm selects fruit, inserts a flower into a vase, and retrieves a drink from a fridge. Each clip shows a fresh MuJoCo simulation executing saved commands from the official VLABench expert.

| 🍎 Fruit selection | 🌷 Flower insertion | 🧃 Fridge drink retrieval |
| :---: | :---: | :---: |
| [![Franka picks an apple and places it on a plate](media/demos/select-fruit-preview.gif)](media/demos/select-fruit-replay.mp4?raw=true) | [![Franka picks a rose and inserts it into a vase](media/demos/insert-flower-preview.gif)](media/demos/insert-flower-replay.mp4?raw=true) | [![Franka opens the fridge, retrieves a drink, and closes the door](media/demos/fridge-drink-preview.gif)](media/demos/fridge-drink-replay.mp4?raw=true) |
| [▶ Full replay · 13.3 s](media/demos/select-fruit-replay.mp4?raw=true) | [▶ Full replay · 18.2 s](media/demos/insert-flower-replay.mp4?raw=true) | [▶ Full replay · 47.8 s](media/demos/fridge-drink-replay.mp4?raw=true) |
| [Replay report · seed 42](docs/stage1-action-replay.json) | [Replay report · seed 42](docs/stage2-insert-flower-replay.json) | [Replay report · seed 43](docs/stage2-fridge-rgb-replay.json) |

The full MP4s show all four cameras at the original 10 Hz timing. The GIF previews run at 2× for fruit and flower and 4× for the fridge.

<a name="data-pipeline"></a>

## 🧩 Data pipeline

**🦾 Official expert → 🔁 Action replay and success check → 📦 LeRobot export**

- **Record.** Save four RGB views, the original instruction, nine measured joint positions, and the absolute actuator targets applied at each 10 Hz control step. Arm joint targets use radians and finger joint targets use meters.
- **Replay.** Restore the recorded initialization, execute the saved commands, and check the simulator trace and official task success.
- **Export.** Keep successful, replay-verified episodes for training. Split by complete episode and initialization. For model input, the adapter uses the forward and wrist views, measured joint positions, and the instruction.

[Collection and LeRobot export](docs/stage2.md) · [Policy inputs and normalization](docs/policy-io.md)

<a name="tasks-and-results"></a>

## 📊 Results and training status

The local **LeRobot v2.1 training split has 60 episodes and 16,520 frames**, with 20 successful, replay-verified episodes per task.

| Task | Official task ID | Expert screening | Training episodes | Training frames |
| --- | --- | ---: | ---: | ---: |
| 🍎 Fruit selection | `select_fruit` | 22 / 22 (100%) | 20 | 3,031 |
| 🌷 Flower insertion | `insert_flower` | 22 / 78 (28.2%) | 20 | 3,482 |
| 🧃 Fridge drink retrieval | `take_out_cool_drink` | 21 / 55 (38.2%) | 20 | 10,007 |

Expert screening counts distinct task and seed pairs across the pilot and six expansions. A seed counts as successful if a recorded retry succeeded. Screening stopped as each task reached its collection target. [Screening logs and failures](docs/stage2.md#expert-screening) · [Final dataset audit](docs/stage2-expansion-06.json)

| Milestone | Status |
| --- | --- |
| Expert collection, fresh action replay, and LeRobot export | ✅ Verified locally |
| Policy input mapping, action execution, and official task scoring | ✅ Verified locally |
| Base π0.5 and multitask supervised fine-tuning (SFT) | ⏳ Pending GPU validation |
| SFT checkpoint handoff and PPO post-training | ⏳ Pending GPU validation |

The local evaluation harness verified the action budget with joint-position holds on all three tasks. With the same fruit scene and different goals, saved expert actions reached the pear goal in 151 steps and the banana goal in 136. Both final physics states matched their source traces. Model evaluation will use held-out seeds and fruit prompts that change the wording or the target. [Local smoke test](docs/closed-loop.md) · [Evaluation protocol](configs/evaluation-v1.json) · [SFT-to-PPO audit](docs/sft-rl-compatibility.md)

<a name="reproduce-the-fruit-demo"></a>

## 🚀 Quickstart

Reproduce the fruit demonstration on Ubuntu 22.04 under WSL2. Prepare the pinned simulator checkout, official asset subset, and Python environment using the [local setup guide](docs/stage1.md#setup-and-run). Then run these commands from the repository root with a fresh output directory.

```bash
bash scripts/run_wsl.sh probe --task select_fruit --seed 42
bash scripts/run_wsl.sh collect --task select_fruit --seed 42 --output runs/select_fruit-42
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5
```

`expert.mp4` records the expert run. `action_replay.mp4` shows a fresh simulator executing the saved actuator commands.

<a name="project-layout"></a>

## 📂 Project layout

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Expert capture, action replay, LeRobot export, and local evaluation |
| [`configs/`](configs/) | Held-out evaluation seeds and paired fruit prompts |
| [`docs/`](docs/) | Reproduction guides, experiment reports, and upstream version pins |
| [`media/demos/`](media/demos/) | Expert action-replay previews and full four-view videos |

<a name="dependencies-and-license"></a>

## 📚 Credits and license

Built on VLABench, MuJoCo, LeRobot, OpenPI, and RLinf. Project code uses the MIT license. Assets and model checkpoints retain their upstream terms. [Upstream revisions and attribution](docs/upstream.md) · [MIT license](LICENSE)
