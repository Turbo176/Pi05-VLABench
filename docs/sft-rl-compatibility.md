# π0.5 SFT to PPO source audit

Repository revisions are recorded in [Upstream revisions](upstream.md). The local test exercised the pinned OpenPI LeRobot reader. This audit covers the model and rollout source. GPU validation is pending.

| Boundary | Finding at the pinned revisions |
| --- | --- |
| SFT model and dataset | RLinf's `FSDPVlaSftWorker` can build the official OpenPI dataloader with `framework="pytorch"`. The loader checks the model action dimension and prediction horizon against the OpenPI `TrainConfig`. The model factory constructs `Pi0` for SFT and `Pi0RL` for RL. `Pi0RL` inherits `Pi0`. |
| Runtime packages | RLinf's installer uses `rlinf-openpi==0.1.1` for the examined OpenPI environments and pins LeRobot to `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, the same reader tested locally. Verify the installed OpenPI fork and loader in the GPU environment. |
| Cameras | OpenPI and RLinf both default to `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb` in `preprocess_observation`. The VLABench dataset retains four original views. The [local input check](policy-io.md) matched the forward and wrist inputs with the unused wrist slot masked. Verify the exact tensors after SFT and rollout preprocessing on the GPU. Processing a fourth camera requires an explicit change to the preprocessing call. |
| Environment and prompt | RLinf's environment registry has no VLABench entry. Its default OpenPI environment adapter maps `main_images`, optional wrist and extra views, `states`, and `task_descriptions`. VLABench needs its own registration and a mapping from the captured observations to those inputs. |
| State and action normalization | RLinf's official SFT loader and rollout transform builder both resolve an OpenPI data config. The rollout pipeline accepts an explicit `openpi_data.norm_stats_path`. Compute the statistics on the training split and use the same file for the SFT and RL checkpoints. VLABench uses measured joint positions and nine absolute actuator targets. Check any delta conversion and its inverse before using PPO. |
| Action chunk | The official SFT loader takes the prediction horizon from `TrainConfig.model.action_horizon`. `num_action_chunks` is the execution length. The local 50-step test covers dataset sampling at episode boundaries. Measure the actual horizon and execution length in the GPU smoke test. |
| Checkpoint handoff | RLinf has code to resolve consolidated `full_weights.pt` and safetensors. Some loader paths use `strict=False` and explicitly check the backbone keys. Review every missing and unexpected key and confirm that the RL load retains the SFT policy tensors. The expected RL-only value head can be initialized separately. |

## GPU smoke test

Start with the shared RLinf PyTorch route. Run one batch through forward and backward, save the weights and optimizer state, and resume. Compare matching inputs, seeded evaluation actions, and short closed-loop outcomes before and after loading the same policy into `Pi0RL`, with PPO updates disabled. Keep the camera masks, language, state, normalization, actuator conversion, chunk execution, reward, and termination fixed. For one PPO update, record policy gradients and parameter changes separately from the value head.

## Closed-loop comparison

Use successful, replay-verified demonstrations for SFT. Keep every expert screening outcome in the collection report. Select the first PPO task using development rollouts after SFT. Prefer a short task with a clear reward and recurring policy failures on feasible starts. Reserve the final test starts and prompts before evaluating checkpoints, and exclude them from both SFT data and PPO rollouts.

Run the base π0.5, the multitask SFT checkpoint, and its PPO descendant from the same test starts, with the same observations, action execution, step budget, and official success condition. Verify the base model's action convention before scoring it. Also score the SFT checkpoint loaded into RLinf before the first update to check the handoff. Report successes/attempts and confidence intervals per task, including tasks outside the PPO training set. Keep episode-level outcomes to see which paired starts improved or regressed.

Include ordinary held-out starts and a predefined generalization group. For `select_fruit`, use unseen placements and prompts that identify different objects in the same starting scene, when the official task supports both targets. If SFT already solves every ordinary test episode, report that ceiling and measure both versions on the predefined generalization group. Report unchanged or worse results alongside gains. Use representative matched-start rollout videos to illustrate failure modes and corrections.

## Local evaluation preflight

The [evaluation config](../configs/evaluation-v1.json) reserves seeds 1001–1020 per task for ordinary starts and seeds 1101–1110 for paired fruit prompts. Each comparison uses a 600-control-step limit at 10 Hz. The training expansion runner rejects overlapping seed ranges. The preflight tested seed 1001 twice in each task and matched the initial simulator state, camera images, and instruction across resets. All three initialization hashes differ from the existing dataset splits.

The [frozen fruit prompts](../configs/evaluation-fruit-prompts-v1.json) record all 30 instructions across 10 layouts. For each layout, the three prompt variants produce identical initial simulator states and camera images. The 10 layouts have distinct initialization hashes, all absent from the training export.

The fruit trial at seed 1101 starts from the same simulator state and renders the same first images under the official pear instruction, a paraphrase, and an instruction targeting the banana in the scene. The official expert placed the pear in 151 steps and the banana in 136 steps in separate runs. Each final state satisfied the official containment condition only for its requested fruit. See the [preflight measurements](evaluation-preflight.json).

```bash
export PI05_SIM_SOURCE_ROOT=$(cat .local/active-sim-native)
for mode in official paraphrase alternate_target; do
  bash scripts/run_wsl.sh probe --task select_fruit --seed 1101 --fruit-prompt "$mode"
done
for mode in official alternate_target; do
  bash scripts/run_wsl.sh probe --task select_fruit --seed 1101 --fruit-prompt "$mode" --check-fruit-expert
done
```

## Source locations

- RLinf [runtime installer](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/requirements/install.sh).
- OpenPI observation contract and default camera selection in [`models/model.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py).
- RLinf [official SFT loader](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/data/datasets/openpi/official_sft_data_loader.py) and [model factory](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/__init__.py).
- RLinf [camera preprocessing](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/modules/model.py), [environment registration](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/envs/__init__.py), and [rollout transforms](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/transforms/pipeline.py).
- RLinf [checkpoint loading](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/checkpoint.py) and [PPO policy forward path](https://github.com/RLinf/RLinf/blob/a29c0a6c37b02881b8d4af0900fa97f35e44b34c/rlinf/models/embodiment/openpi/tasks/rl.py).
