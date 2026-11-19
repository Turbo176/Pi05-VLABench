# Upstream revisions

| Component | Revision | Local status |
| --- | --- | --- |
| [VLABench](https://github.com/OpenMOSS/VLABench) | `cf588fe60c0c7282174fe979f5913170cfe69017` | Expert, task, robot, and simulator used in the smoke test |
| [RRT Algorithms](https://github.com/motion-planning/rrt-algorithms) | `14e47963780affe3ff3bb9fc33a4b14e1a733e85` | Expert planning dependency used from the source checkout |
| [LeRobot VLABench guide](https://huggingface.co/docs/lerobot/vlabench) | Documentation reviewed 2026-11-19. Git HEAD observed at `8ec1a474da5f3f90a670d9343386995ec9f81264`. VLABench requirements pin LeRobot separately at `6674e368249472c91382eb54bb8501c94c7f0c56`. | Dataset conversion pending |
| [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` observed and README reviewed | SFT integration pending |
| [RLinf](https://github.com/RLinf/RLinf) | `a29c0a6c37b02881b8d4af0900fa97f35e44b34c` observed and README reviewed | PPO integration pending |

The VLABench checkout also records an OpenPI fork at `c0b0bd4bce430e1874fbea68623d98ff13934f3e`. The smoke test uses neither that fork nor the Physical Intelligence checkout. OpenPI/RLinf checkpoint compatibility requires source inspection and a runtime test.

Official asset archives are identified by the Drive IDs in VLABench's `scripts/download_assets.py`. The subset downloader records the source archive IDs, file sizes, and SHA-256 hashes in `.local/`. Keep the upstream MIT notices for [VLABench](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/LICENSE.txt) and [RRT Algorithms](https://github.com/motion-planning/rrt-algorithms/blob/14e47963780affe3ff3bb9fc33a4b14e1a733e85/LICENSE) in their respective checkouts. The Franka model has its own license in `VLABench/assets/robots/franka_emika_panda/LICENSE`. Upstream code and assets stay in the ignored `.local/` directory.
