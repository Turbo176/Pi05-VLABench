# Script guide

The supported learned-policy path uses seven-coordinate pose actions. Start with the entry points below; the other Python files supply their shared data, model, controller, and scoring functions.

| Step | Entry point | Instructions |
| --- | --- | --- |
| Local simulator and expert replay | `run_wsl.sh` | [Local setup](../docs/stage1.md) |
| Pose demonstrations and replay admission | `collect_pose_expert_batch.py`, `rerender_pose_dataset.py` | [Data pipeline](../docs/stage2.md) |
| LeRobot export | `convert_lerobot_wsl.sh` | [Data formats and alignment](../docs/stage2.md) |
| Shared fruit/flower SFT | `train_reference_aligned_fruit.py` | [GPU commands](../docs/single-gpu.md) |
| Zero-update SFT-to-RL check | `check_reference_rl_handoff.py` | [GPU commands](../docs/single-gpu.md) |
| Fruit PPO | `train_fruit_ppo.py` (reward follow-up); `ppo_reference_aligned_fruit.py` (frozen experiment) | [GPU commands and status](../docs/single-gpu.md) |
| Fixed-scene model evaluation | `evaluate_pose_development.py` | [GPU commands](../docs/single-gpu.md) |
| Results and video recording | `record_fruit_completion.py`; study-local `analyze.py` | [Forty-scene study](../docs/experiments/2026-11-24/fruit-recovery-v2/README.md) |
| CPU evidence verification | Study-local `analyze.py --verify-only`; `review_fruit_experiment.py` for the archived pilot | [Current commands](../docs/experiments/2026-11-24/fruit-recovery-v2/commands.md) |
| Saved-log plots and motion audit | `analyze_training_and_motion.py` | [Analysis commands](../docs/experiments/2026-11-23/training-and-motion/README.md) |

`select_fruit.py` owns official environment setup and saved-action capture/replay. `reference_pose_lerobot.py` reads aligned data; `pi05_runtime.py` constructs the policy; `ppo_pi05.py` supplies likelihood and PPO update functions. These modules are shared dependencies, not separate competing training pipelines.

`audit_pose_training_contract.py` and `check_reference_fruit_label_replay.py` check action alignment. The saved-checkpoint audit and CPU evidence reviewer document actual actor updates. `prepare_reference_pose_cases.py` freezes evaluation resets. The local setup wrappers and `vlabench_assets.py` prepare the isolated simulator and official asset subset.

The three JSON configurations have different roles: `evaluation-fruit-completion-v1.json` fixes the final comparison, `ppo-fruit-completion-v1.json` records its PPO recipe, and `evaluation-v1.json` supplies defaults to the standalone reference evaluator. SFT settings and commands are recorded in the GPU guide and saved training summary.

`train_fruit_ppo.py` adds explicit reward versions, critic warmup and per-minibatch diagnostics. Its CPU tests live in `tests/test_fruit_ppo.py`. Both reward variants completed full-model GPU smoke training on November 24. The separate SFT180 expansion saved one PPO checkpoint and completed forty paired test scenes. The report distinguishes the saved checkpoint's cost from the later interrupted rollout. The legacy PPO helpers retain the historical experiment's source hashes.

For subsequent recordings, `record_fruit_completion.py` accepts `--ppo-weights` and `--label` to identify the selected checkpoint correctly. It still requires exact agreement with the scored rollout's observation and action hashes.
