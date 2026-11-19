# Stage 1 local runbook

Validated on Windows 10 build 19045 and Ubuntu 22.04.5 under WSL2. Windows reported 15.36 GiB RAM, a GTX 1650 Ti with 4 GiB VRAM, and 70.5 GiB initially free on D:. WSL exposed 7.4 GiB RAM and detected the same GPU. The run used Mesa EGL software rendering. No system packages were installed.

| Component | Location on the test machine |
| --- | --- |
| Project | `D:\Robotics\Pi05-VLABench` (`/mnt/d/Robotics/Pi05-VLABench`) |
| Checkouts | `.local/VLABench` and `.local/rrt-algorithms` |
| Official assets | `.local/VLABench/VLABench/assets` |
| Python environment | `/var/tmp/pi05-vlabench-1000/venv` on the WSL filesystem backed by `D:\WSL\Ubuntu-22.04\ext4.vhdx` |
| pip and rendering caches | `.local/pip-cache` and `.local/cache` |
| Episode, videos and JSON reports | `runs/select_fruit-42/` |

## Setup and run

From the project root, check out VLABench and RRT Algorithms into `.local/` at the exact revisions in [upstream.md](upstream.md). The RRT repository's package build omits `rrt_algorithms/rrt`, so `run_wsl.sh` loads the planner from the checkout. The setup script uses WSL's Linux filesystem for the environment because installing directly to the Windows mount was slow.

From Windows PowerShell, fetch the official assets used by this test. Windows Python 3.9 works for this tool. Requests use a direct connection by default. For networks where Drive direct access fails, `--proxy-fallback` tries the configured network route after a direct probe. Each invocation caps the planned ZIP ranges at 100 MiB.

```powershell
python scripts/vlabench_assets.py scenes --match '^scenes/default/empty\.xml$' --manifest .local/default-xml.json
python scripts/vlabench_assets.py scenes --xml-deps scenes/default/empty.xml --manifest .local/assets-select-fruit-default.json
python scripts/vlabench_assets.py objects --mesh-directories --match '^obj/meshes/(table|fruit/(apple/apple_10|banana/banana_8|orange/orange_2|pear/pear_12|kiwi/kiwi_1|lemon/lemon_10|mango/mango_1|peach/peach_6)|tablewares/plates/(plate3|plate_9))/' --match '^obj/assets/textures/wood0\.png$' --manifest .local/assets-select-fruit-objects.json
```

The asset subset keeps one official mesh variant per fruit category and two official plate variants, one on each side of VLABench's plate split. `select_fruit` can draw distractors from both seen and unseen fruit categories. The two official Drive bundles total 6,003,270,210 bytes. This test used HTTP range requests for roughly 100 MB of asset data. The task config contains `studyroom_0`, but the upstream scene constructor resolves this task to `scenes/default/empty.xml`. The trial extracted the named studyroom first and reused byte-identical assets. New setups can fetch the resolved default scene directly.

Run from WSL Ubuntu 22.04 with Python 3.10 and Mesa EGL already installed.

```bash
bash scripts/setup_wsl.sh
bash scripts/run_wsl.sh probe
bash scripts/run_wsl.sh collect --seed 42 --output runs/select_fruit-42
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5
```

To render the replay at 768 x 768 per camera, execute the same commands from the saved episode with `--hd`:

```bash
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5 --hd
```

The HD replay passed the same state checks and decoded as 133 frames at 1536 x 1536 and 10 fps. It writes `action_replay_hd.mp4` alongside the episode.

## Episode fields

Each HDF5 row pairs the RGB and robot state **before** action `t` with the 9 values passed to `env.step()` at that same tick. The 7 arm channels are absolute joint targets in radians. The last 2 are finger joint position targets in metres. Their coordinates are the individual joint coordinates. `state` records measured joint positions in the same order. `time` and `next_time` are MuJoCo seconds and differ by 0.1 s. The camera order is `right`, `left`, `forward`, `franka/Franka_wrist_cam`. The final image is a separate post-action frame.

Replay checks the initial MuJoCo state and body transforms, step timing, each pre-action robot state, each post-action position/velocity array, and terminal/success flags. It receives actions from HDF5 and renders fresh frames for `action_replay.mp4`.

Candidate policy inputs for dataset conversion are instruction, RGB, and measured robot state. Action is the supervised target. Episode configuration, simulator state, reward, success, and terminal flags are provenance or diagnostics. The raw archive stores all of these fields for audit and replay.

Both videos decoded as 133 frames at 512 x 512 and 10 fps. The captured HDF5 SHA-256 is `d1572aaa6b2b4e4a814e83749a41fb3c9a612e099075c0f965fb1ed242d51c79`. See [the installed Python versions](stage1-pip-freeze.txt), [the expert report](stage1-expert.json), and [the replay report](stage1-action-replay.json).
