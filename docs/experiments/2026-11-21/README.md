# π0.5 on VLABench · November 21, 2026

## Fine-tuning and handoff

The run used a single RTX 4080 SUPER with 32 GiB. The official `pi05_base` checkpoint was converted from JAX to the pinned RLinf PyTorch format. The multitask run updated 2,165,792 parameters in the action projections and timestep MLPs. It sampled 20 demonstrations per task, used 300 optimizer updates per task, and saved one shared checkpoint after 900 total updates. The update loop took 227.96 seconds and peak CUDA allocation was 9.49 GiB. [SFT summary](sft-summary.json) · [Per-update log](sft-updates.jsonl)

| Task | Mean loss, first 100 task updates | Mean loss, last 100 task updates | Development successes |
| --- | ---: | ---: | ---: |
| Fruit selection | 0.1727 | 0.1298 | 0 / 1 |
| Flower insertion | 0.1646 | 0.0924 | 0 / 1 |
| Fridge drink retrieval | 0.2259 | 0.1508 | 0 / 1 |

The three-episode GPU smoke test restored the optimizer and reproduced the next update. The final SFT checkpoint produced the same action hashes in `Pi0` and `Pi0RL` on all three development rollouts before PPO. All three used the 600-control-step budget. [Smoke test](sft-smoke-summary.json) · [Handoff check](handoff-verified.json) · [SFT development rollouts](sft-development.jsonl) · [PPO handoff rollouts](handoff-development.jsonl)

## PPO

The SFT policy achieved 0 / 4 official successes on fruit development starts 1201–1204. We added a distance-progress reward with weight 0.25 to the official terminal reward for the PPO run. The official evaluation metric remained task success. PPO used starts 1211–1214 and executed 2,400 control steps over 4 episodes, with 0 / 4 official successes during collection. The PPO rollout and optimizer loop took 285.83 seconds. [Development rollouts](fruit-reward-development.jsonl) · [PPO update](ppo-updates.jsonl)

The maximum discrepancy between sampled and recomputed log probabilities was 0. The maximum absolute change in an action-policy parameter from SFT was 7.23e-5. The saved PPO checkpoint records the SHA-256 of its SFT parent. The action network and a value head both received updates. The maximum unclipped actor gradient L2 was 102.28. Peak CUDA allocation was 6.72 GiB. The SFT and PPO training loops took 513.79 seconds in total. This excludes provisioning, downloads, checkpoint conversion, and evaluation.

## Evaluation protocol

The matched rollout suite uses identical starts, instructions, input normalization, noise seeds and 10 Hz actuator control across model stages. It runs up to 600 control steps, executes ten actions per forecast, and uses official VLABench success conditions. The baseline uses the official base weights with the project’s nine-coordinate VLABench adapter. [Protocol](../../../configs/evaluation-v1.json) · [Run commands and adapter details](../../single-gpu.md)

## First held-out seed

At seed 1001, every policy ran each of the three tasks for 600 control steps. All nine runs failed the official success check. [Paired summary and logs](first-seed/summary.json) · [Rollout videos](../../../media/policy/)

The first base-policy flower rollout had the same initial physics state and a different four-camera pixel hash. A standalone check rendered the scene before and after a fruit reset. All four views matched across the two processes and repeats. The replacement base rollout matched the SFT and PPO runs exactly at reset. We retained the original row and used the matched rerun for the comparison. [Original log](first-seed/ordinary-base-original.jsonl) · [Separate process checks](first-seed/rgb-cold.log) and [warm reset](first-seed/rgb-warm.log)

## Artifact digests

| Artifact | SHA-256 |
| --- | --- |
| Converted official base weights | `f6391204c480d6c54d31d63601e7963ef23301157df052a4b8183e7b019ce2f1` |
| Multitask SFT checkpoint | `2f35e799b0d5a177b755cc8b92b64e7e90f79e4df85b90c11e35d39b7b1c4401` |
| Fruit PPO checkpoint | `d34a10204d71184c7ff105f4ed177bb88a57b738779151388908d83ae8cae6a9` |
| Training normalization | `fc89b1b8f11586cd09f963ae9ceb8ef774fa44204905e43ddaa2614a3f3744e2` |
