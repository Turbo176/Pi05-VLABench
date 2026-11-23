# Upstream revisions

| Component | Revision | Local status |
| --- | --- | --- |
| [VLABench](https://github.com/OpenMOSS/VLABench) | `cf588fe60c0c7282174fe979f5913170cfe69017` | Expert, task, robot, and simulator used in the smoke test |
| [RRT Algorithms](https://github.com/motion-planning/rrt-algorithms) | `14e47963780affe3ff3bb9fc33a4b14e1a733e85` | Expert planning dependency used from the source checkout |
| [LeRobot VLABench guide](https://huggingface.co/docs/lerobot/vlabench) | Documentation reviewed 2026-11-19. Git HEAD observed at `8ec1a474da5f3f90a670d9343386995ec9f81264`. VLABench requirements pin LeRobot separately at `6674e368249472c91382eb54bb8501c94c7f0c56`. | Stage 2 conversion uses LeRobot v0.4.4 at `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` | Official `pi05_base` weights and OpenPI tokenization and normalization in the GPU experiment |
| [RLinf](https://github.com/RLinf/RLinf) | `a29c0a6c37b02881b8d4af0900fa97f35e44b34c` | Checkpoint conversion, `Pi0` for SFT and `Pi0RL` for PPO |

The VLABench checkout also records an OpenPI fork at `c0b0bd4bce430e1874fbea68623d98ff13934f3e`. The GPU experiment used the Physical Intelligence revision in the table and verified the SFT checkpoint handoff to RLinf by comparing complete action traces. The legacy LeRobot v2.1 format was checked with LeRobot revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. [Checkpoint audit](sft-rl-compatibility.md)

The pose-policy experiment uses the [VLABench OpenPI fork](https://github.com/Shiduo-zh/openpi/tree/788ef1f20e655de5bb40869fb7b7fcdd49732387) at `788ef1f20e655de5bb40869fb7b7fcdd49732387`. Its published VLABench π0.5 Primitive model is pinned to `f658a8bec801aab960fe0ceab370a123b31c1837`. [Experiment and checkpoint provenance](experiments/2026-11-21/failure-diagnosis/reference-aligned-fruit/shared-pose/README.md)

Official asset archives are identified by the Drive IDs in VLABench's `scripts/download_assets.py`. The subset downloader records the source archive IDs, file sizes, and SHA-256 hashes in `.local/`. Keep the upstream MIT notices for [VLABench](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/LICENSE.txt) and [RRT Algorithms](https://github.com/motion-planning/rrt-algorithms/blob/14e47963780affe3ff3bb9fc33a4b14e1a733e85/LICENSE) in their respective checkouts. The Franka model has its own license in `VLABench/assets/robots/franka_emika_panda/LICENSE`. Upstream code and assets stay in the ignored `.local/` directory.
