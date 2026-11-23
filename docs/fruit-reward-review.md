# 🍐 Fruit PPO reward review

Reviewed on November 23, 2026. This note explains the completed experiment's reward and a possible next experiment. It does not change the frozen PPO checkpoint, registered configuration, or existing scores. No new PPO run has been started for this proposal.

## 🔎 What actually ran

The completed run used the reward in [`PPOClient.finish_chunk`](../scripts/ppo_reference_aligned_fruit.py) and [`fruit_potential`](../scripts/ppo_pi05.py):

```text
Phi = -min(gripper_to_target_3d_m, 0.7) - min(target_to_plate_xy_m, 0.5)
reward = official_success + 0.5 * (Phi_after_chunk - Phi_before_chunk)
```

Rewards are measured after each executed action chunk, normally five controls at 10 Hz. Official success ends the rollout, so its +1 is awarded once. The `0.5` is a coefficient on distance progress in meters; it is not a separate +0.5 lifting reward or a +0.5 placement reward. Simulator geometry is used only to calculate rewards and diagnostics, never as policy input.

The [actual two training rollouts](experiments/2026-11-22/fruit-completion-v1/ppo/rollouts.jsonl) were:

| Training seed | Official success | Controls | Undiscounted reward sum |
| --- | ---: | ---: | ---: |
| 1301, pear | Yes | 131 | 1.429031 |
| 1302, apple | No | 600 | -0.017333 |

Across 147 chunks, 57 had nonzero reward; the smallest chunk reward was -0.413554 and the largest was +0.990659. These are measured rewards from the completed experiment, not estimated new results. The model received 731 environment controls and 19 optimizer steps in one PPO iteration. That sample cannot isolate reward design as the cause of unchanged ordinary success or regressed language tests.

## ⚠️ Limitations to test

The current geometry terms do not explicitly distinguish touching from a stable grasp. Target-to-plate distance uses only XY, so moving an object above the plate can reduce that distance without placing it. The reach term rewards keeping the gripper close; moving the hand away after release can make it worse. These are objective-design concerns, not proof that any particular failed rollout exploited them. Official success is still the only evaluation criterion.

There is also a discount-consistency limitation: advantage calculation uses `gamma=0.99` per policy chunk, while the progress bonus uses `Phi_next - Phi_now`. Classical potential-based shaping uses `gamma * Phi_next - Phi_now`, with consistent terminal handling. The existing bonus is therefore a distance heuristic; this project does not claim that it preserves the original task's optimal policy. See Ng, Harada and Russell, [Policy invariance under reward transformations (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). Changing this formula alone would not establish that PPO will improve.

## 🧪 Proposed next experiment, not yet implemented

The suggested split of +0.5 for lifting and +0.5 for placement is a reasonable candidate to compare. A more completion-focused starting candidate is:

| Event | One-time reward | Detection requirement |
| --- | ---: | --- |
| Securely grasp and lift the instructed fruit | +0.2 | Verified grasp/contact and sustained lift above its reset height; do not count a push, bounce, or wrong fruit |
| Place the fruit and release | +0.8 | Preserve official task success; separately verify release and stable placement for this reward event |

The values are proposed hyperparameters, not validated settings. The un-discounted event reward is at most 1 per episode. Track an episode-local ledger so lifting, hovering, releasing and regrasping do not repeatedly collect the same milestone reward. Check the predicates against successful expert trajectories and failed trajectories before training. Keep official success reporting unchanged; if official success can occur before the release/stability check, explicitly define the extra validation controls and termination rule in the new protocol.

For context, robosuite's [official pick-and-place reward implementation](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/pick_place.py) also distinguishes reaching, grasping, lifting and hovering. It is a design reference, not code that was run in VLABench and not a proposed environment change.

Any later reward comparison should start from the exact same SFT90 parent and fix the training interaction budget, controller and evaluation settings. Register the new experiment before running it. Use training/development starts for debugging; the already reported 1501–1510 and 1601–1603 results have been inspected and must not become an untouched test set for a tuned reward. Preserve the present checkpoint and its negative results.
