# Script guide

The supported learned-policy path uses seven-coordinate pose actions. Start with the entry points below; the other Python files supply their shared data, model, controller, and scoring functions.

| Step | Entry point | Instructions |
| --- | --- | --- |
| Local simulator and expert replay | `run_wsl.sh` | [Local setup](../docs/stage1.md) |
| Pose demonstrations and replay admission | `collect_pose_expert_batch.py`, `rerender_pose_dataset.py` | [Data pipeline](../docs/stage2.md) |
| LeRobot export | `convert_lerobot_wsl.sh` | [Data formats and alignment](../docs/stage2.md) |
| Shared fruit/flower SFT | `train_reference_aligned_fruit.py` | [GPU commands](../docs/single-gpu.md) |
| Zero-update SFT-to-RL check | `check_reference_rl_handoff.py` | [GPU commands](../docs/single-gpu.md) |
| Fruit PPO | `ppo_reference_aligned_fruit.py` | [GPU commands](../docs/single-gpu.md) |
| Fixed-scene model evaluation | `evaluate_pose_development.py` | [GPU commands](../docs/single-gpu.md) |
| Results and video recording | `summarize_fruit_completion.py`, `record_fruit_completion.py` | [Final experiment](../docs/experiments/2026-11-22/fruit-completion-v1/README.md) |
| CPU evidence verification | `review_fruit_experiment.py` | [Expected output](../docs/experiments/2026-11-23/project-overview/README.md) |
| Saved-log plots and motion audit | `analyze_training_and_motion.py` | [Analysis commands](../docs/experiments/2026-11-23/training-and-motion/README.md) |

`select_fruit.py` owns official environment setup and saved-action capture/replay. `reference_pose_lerobot.py` reads aligned data; `pi05_runtime.py` constructs the policy; `ppo_pi05.py` supplies likelihood and PPO update functions. These modules are shared dependencies, not separate competing training pipelines.

`audit_pose_training_contract.py` and `check_reference_fruit_label_replay.py` check action alignment. The saved-checkpoint audit and CPU evidence reviewer document actual actor updates. `prepare_reference_pose_cases.py` freezes evaluation resets. The local setup wrappers and `vlabench_assets.py` prepare the isolated simulator and official asset subset.

The three JSON configurations have different roles: `evaluation-fruit-completion-v1.json` fixes the final comparison, `ppo-fruit-completion-v1.json` records its PPO recipe, and `evaluation-v1.json` supplies defaults to the standalone reference evaluator. SFT settings and commands are recorded in the GPU guide and saved training summary.
