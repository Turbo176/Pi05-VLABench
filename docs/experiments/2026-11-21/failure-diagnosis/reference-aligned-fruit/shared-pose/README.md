# Shared pose-policy experiment

Both variants fine-tuned the published VLABench π0.5 model on one LeRobot v2.1 training split. It contains 45 fruit, 10 flower, and 8 fridge episodes with 12,443 frames. Six separate validation episodes contain 1,239 frames. The policy uses forward, right, and wrist RGB, the instruction, and measured end-effector pose and gripper state. It predicts ten poses and executes five through the official adapter and inverse kinematics at 10 Hz. Development and held-out rollouts use the official success check and a 600-control cap. [Dataset verification](dataset-verification.json)

## Development results

| Policy | Fruit, seeds 1201–1204 | Flower, seeds 1211–1214 | Fridge, seeds 1221–1224 |
| --- | --- | --- | --- |
| Published VLABench π0.5 in RLinf | [2/4](reference-fruit-episodes.jsonl) | [1/4](reference-flower-episodes.jsonl) | [0/4](reference-fridge-episodes.jsonl) |
| [Shared SFT, 450 updates](sft-450/summary.json) | [1/4](sft-450/episodes.jsonl) | [0/4](sft-450-flower/episodes.jsonl) | [0/4](sft-450-fridge/episodes.jsonl) |
| [Shared SFT, 90 updates](sft-90/summary.json) | [1/4](sft-90/episodes.jsonl) | [1/4](sft-90-flower-episodes.jsonl) | Pending |
| PPO from 90-update SFT | [3/4](ppo/fruit-development-episodes.jsonl) | Pending | Pending |

With four cases, the 95% Wilson intervals for 0/4, 1/4, 2/4, and 3/4 are 0–49.0%, 4.6–69.9%, 15.0–85.0%, and 30.1–95.4%. The 450-update SFT solved seed 1204 in 137 controls. The 90-update SFT solved seed 1203 in 191 controls. [Video run record](sft-90-video-episode.jsonl) The remaining six fruit episodes moved the target without meeting the success condition. The 450-update version moved the target in three of four failed flower episodes. Its four fridge episodes each moved the target under 3 cm. The upstream JAX implementation of the published reference solved [3/4 fruit cases](reference-native/episodes.jsonl). The original OpenPI π0.5 base experiments use a different starting checkpoint and are reported in the [parent experiment](../README.md).

## Checkpoint selected for PPO

Both variants tied on fruit development success. We selected the 90-update variant with a frozen vision-language module. Its fixed-input actions matched exactly after saving and loading the weights. The RLinf actor then matched its normalized predictions and the shared NumPy action decode on three fixed input windows. On seed 1203, the reloaded actor solved the task in 191 controls with the same SHA-256 over actuator commands. The native RLinf float32 decode differed by at most 4.53e-7 in the fixed-input check. The shared NumPy decode used by both rollouts matched exactly. [Handoff audit](rlinf-handoff.json) · [Closed-loop record](rlinf-handoff-episodes.jsonl)

PPO uses separate training starts 1301 and 1302. The reward is the official binary success reward plus 0.5 times the within-chunk progress toward the fruit and plate, measured from simulator geometry. Reward geometry never enters the policy observation. The project invokes RLinf's π0.5 actor and records the log-probability for the portion of each action chunk that the robot executes. The run records per-episode controls, actor gradient norm, parameter change, and wall time.

## PPO update and evaluation

One PPO iteration collected 195 action chunks over 974 control steps. The exploratory policy finished one of two training starts. The iteration made 25 optimizer steps and took 249.8 seconds from the start of collection through checkpoint save, excluding model load. Peak allocated CUDA memory was 6.83 GiB. The largest pre-clipping actor gradient norm was 91.98, and the largest absolute actor parameter change from SFT was 4.26e-5. The actor updated the action expert's attention and MLP LoRA parameters, action projections, and time MLP. The critic took detached vision-language features. [Training rollouts](ppo/rollouts.jsonl) · [Optimizer and checkpoint audit](ppo/iteration-1.json)

The PPO checkpoint reloaded from the recorded SFT parent and the shared normalization. Fruit development success went from 1/4 to 3/4 under the same protocol. Seeds 1201, 1202, and 1204 changed from failure to success. Seed 1203 changed from success at 191 controls to failure at 600 controls. [Paired results and intervals](ppo/fruit-development-summary.json) · [Success video record](ppo/success-video-episode.jsonl) · [Failure video record](ppo/development-video-episode.jsonl)

## Reproduce

Use the pinned simulator, VLABench OpenPI fork, and RLinf runtime. Set `JAX_PLATFORMS=cpu` for paired evaluation noise. `REFERENCE` contains the converted published weights and its normalization. `DATASET` is the verified pose export. `FRUIT_CASES` contains the matched development reset signatures.

```bash
python scripts/train_reference_aligned_fruit.py \
    --base "$REFERENCE/model.safetensors" --fork "$OPENPI_FORK" \
    --reference-start "$REFERENCE" --lerobot "$DATASET" --paired-episodes "$FRUIT_CASES" \
    --multitask --vlm-lora-rank 0 --no-image-augmentation --updates 90 --lr 0.000001 --output "$SFT_RUN"
python scripts/check_reference_rl_handoff.py \
    --base "$REFERENCE/model.safetensors" --fork "$OPENPI_FORK" --run "$SFT_RUN" \
    --episode "$EXPERT_POSE_EPISODE" --paired-episodes "$SFT_RUN/episodes.jsonl" \
    --eval-seed 1203 --output "$HANDOFF_REPORT"
python scripts/ppo_reference_aligned_fruit.py \
    --base "$REFERENCE/model.safetensors" --fork "$OPENPI_FORK" --sft-run "$SFT_RUN" \
    --output "$PPO_RUN" --seeds 1301 1302 --iterations 1 --lr 0.000002 --shaping 0.5
```

## Held-out smoke test

Seed 1001 completed all three policy stages on all three tasks. The policies loaded the same starting weights, checkpoint lineage, normalization, and matched reset manifests. The bold entries reached the official goal before the 600-control cap.

| Task | Published reference | Shared SFT | PPO from shared SFT |
| --- | --- | --- | --- |
| Fruit selection | **1/1**, 174 controls | **1/1**, 123 controls | 0/1, 600 controls |
| Flower insertion | **1/1**, 152 controls | **1/1**, 396 controls | **1/1**, 152 controls |
| Fridge drink retrieval | 0/1, 600 controls | 0/1, 600 controls | 0/1, 600 controls |

On the fruit failure, PPO moved the target without completing the goal. All three fridge policies moved the target under 3 cm and never grasped the drink. On a separate WSL expert replay at seed 54, the door first passed the open threshold at step 94, the drink cleared the fridge at step 275, and the door closed for task success at step 484. All saved simulator states matched the new action execution exactly. [Expert phase replay](fridge-expert-phase-replay.json) These are single-start outcomes. [Paired records and summary](heldout-1001/summary.json)

## Held-out fruit results

All 20 preregistered fruit starts completed at each stage. The paired check matched the scene, language, initial policy images, sampler, normalization, and control budget.

| Policy | Successes | 95% Wilson interval |
| --- | ---: | ---: |
| Published reference | 12/20 (60%) | 38.7–78.1% |
| Shared SFT | 10/20 (50%) | 29.9–70.1% |
| PPO from shared SFT | 10/20 (50%) | 29.9–70.1% |

PPO solved two starts that SFT missed and failed two starts that SFT solved. Every failed rollout moved the fruit and exhausted the control budget before meeting the goal. [Raw records and paired summary](heldout-fruit/summary.json)

## Evaluation pause on November 22, 2026

The 00:44 China Standard Time snapshot contains 88 of 180 ordinary rollouts. All three policies completed fruit. The reference, SFT, and PPO policies completed 12, 11, and 5 of their 20 flower starts respectively. Fridge and language evaluations are pending in this snapshot. The partial flower counts cover different sets of starts.

The SFT flower process stopped at seed 1012. Four resets either raised a physics error or produced policy camera images that did not match the pinned manifest. The reference later reproduced that start. Resume with the original hashes and count only completed, matched rollouts. The PPO flower smoke test and formal evaluation at seed 1001 matched the initial physics, policy images, checkpoint, normalization, and first action hashes. Their later action hashes and outcomes differed. The smoke test succeeded in 152 controls and the formal rollout exhausted 600 controls.

## Held-out protocol

The reset manifests fix 20 ordinary starts per task and 10 fruit starts with three prompts each. The language cases include the official instruction, a paraphrase, and a different goal in the same scene. The three prompts share the same initial physics and policy-camera images. The episode initialization hashes are disjoint from the training and validation exports. [Protocol](../../../../../../configs/evaluation-v1.json) · [Reset manifests](cases/)

Run the suite with the verified reset manifests and the checkpoint from the PPO run above. Give the full suite and any subset different output directories.

```bash
bash scripts/run_reference_pose_suite.sh "$PYTHON" "$OPENPI_FORK" \
    "$REFERENCE/model.safetensors" "$SFT_RUN" "$PPO_RUN/ppo-1.pt" "$CASES" "$EVAL_OUT"
```

The runner resumes completed cases from their JSONL records after checking the checkpoint, normalization, reset, and control settings. The summary contains success counts, Wilson intervals, paired outcome changes, and failure categories. The language summary also counts changed first commands separately from changed full trajectories.
