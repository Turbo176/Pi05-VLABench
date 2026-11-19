# π0.5 SFT to PPO source audit

Repository revisions are recorded in [Upstream revisions](upstream.md). The local test exercised the pinned OpenPI LeRobot reader. This audit covers the model and rollout source. GPU validation is pending.

| Boundary | Finding at the pinned revisions |
| --- | --- |
| SFT model and dataset | RLinf's `FSDPVlaSftWorker` can build the official OpenPI dataloader with `framework="pytorch"`. The loader checks the model action dimension and prediction horizon against the OpenPI `TrainConfig`. The model factory constructs `Pi0` for SFT and `Pi0RL` for RL. `Pi0RL` inherits `Pi0`. |
| Runtime packages | RLinf's installer uses `rlinf-openpi==0.1.1` for the examined OpenPI environments and pins LeRobot to `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, the same reader tested locally. Verify the installed OpenPI fork and loader in the GPU environment. |
| Cameras | OpenPI and RLinf both default to `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb` in `preprocess_observation`. The VLABench dataset retains four original views. Choose the policy views and masks explicitly and verify the exact tensors after preprocessing in SFT and rollout. Processing a fourth camera requires an explicit change to the preprocessing call. |
| Environment and prompt | RLinf's environment registry has no VLABench entry. Its default OpenPI environment adapter maps `main_images`, optional wrist and extra views, `states`, and `task_descriptions`. VLABench needs its own registration and a mapping from the captured observations to those inputs. |
| State and action normalization | RLinf's official SFT loader and rollout transform builder both resolve an OpenPI data config. The rollout pipeline accepts an explicit `openpi_data.norm_stats_path`. Compute the statistics on the training split and use the same file for the SFT and RL checkpoints. VLABench uses measured joint positions and nine absolute actuator targets. Check any delta conversion and its inverse before using PPO. |
| Action chunk | The official SFT loader takes the prediction horizon from `TrainConfig.model.action_horizon`. `num_action_chunks` is the execution length. The local 50-step test covers dataset sampling at episode boundaries. Measure the actual horizon and execution length in the GPU smoke test. |
| Checkpoint handoff | RLinf has code to resolve consolidated `full_weights.pt` and safetensors. Some loader paths use `strict=False` and explicitly check the backbone keys. Review every missing and unexpected key and confirm that the RL load retains the SFT policy tensors. The expected RL-only value head can be initialized separately. |

## GPU smoke test

Start with the shared RLinf PyTorch route. Run one batch through forward and backward, save the weights and optimizer state, and resume. Compare matching inputs, seeded evaluation actions, and short closed-loop outcomes before and after loading the same policy into `Pi0RL`, with PPO updates disabled. Keep the camera masks, language, state, normalization, actuator conversion, chunk execution, reward, and termination fixed. For one PPO update, record policy gradients and parameter changes separately from the value head.

## Source locations

- RLinf [runtime installer](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/requirements/install.sh).
- OpenPI observation contract and default camera selection in [`models/model.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py).
- RLinf [official SFT loader](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/data/datasets/openpi/official_sft_data_loader.py) and [model factory](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/__init__.py).
- RLinf [camera preprocessing](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/modules/model.py), [environment registration](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/envs/__init__.py), and [rollout transforms](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/transforms/pipeline.py).
- RLinf [checkpoint loading](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/checkpoint.py) and [PPO policy forward path](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/tasks/rl.py).
