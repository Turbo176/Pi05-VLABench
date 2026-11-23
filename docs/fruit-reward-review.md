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

## 📚 Literature and implementation references

| Reference | Relevant finding | Application here |
| --- | --- | --- |
| Ng, Harada and Russell, [Policy invariance under reward transformations, ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), Theorem 1 | Potential-based shaping uses the same discount as the objective: `gamma * Phi(next) - Phi(now)`. | A corrected shaping experiment needs consistent chunk discounting and terminal potentials. This theorem does not guarantee that finite-budget PPO improves. |
| Luo et al., [SERL, ICRA 2024](https://arxiv.org/html/2401.16013v4#S4.SS2), Section 4.2 | Reward specification needs a reliable success detector; classifier rewards can be exploited. | Keep the simulator's official success predicate and check intermediate events against trajectories. SERL uses off-policy RL and provides no PPO-specific guarantee. |
| [robosuite official PickPlace implementation](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/pick_place.py), `reward` and `staged_rewards` | The shaping distinguishes reaching, grasping, lifting, and hovering; lifting is gated on grasping. Its dense contribution uses the maximum stage reward. | Use grasp-conditioned lift detection as a design reference. Its weights and per-step reward are not the one-time milestone proposal below. The project continues to use VLABench. |

These sources support clear progress signals and careful reward semantics. None establishes that a 0.5/0.5 or 0.2/0.8 split is optimal for this model and task.

## 🧪 Proposed next experiment, not yet implemented

The suggested split of +0.5 for lifting and +0.5 for placement is a reasonable candidate to compare. A more completion-focused starting candidate is:

| Event | One-time reward | Detection requirement |
| --- | ---: | --- |
| Securely grasp and lift the instructed fruit | +0.2 | Verified grasp/contact and sustained lift above its reset height; do not count a push, bounce, or wrong fruit |
| Complete the official fruit task | +0.8 | The unchanged official success predicate becomes true; do not wait for an additional release or stability condition |

The values are proposed hyperparameters, not validated settings. The undiscounted event reward is at most 1 per episode; a success without the lift milestone receives 0.8. Track an episode-local ledger so lifting, hovering, releasing and regrasping do not repeatedly collect the same milestone reward. Detect lift persistence at each underlying control tick even though rewards are accumulated per policy chunk. Determine and freeze the height and persistence thresholds using training/development trajectories before PPO. Check wrong-object lifts, pushes, bounces, repeated grasps, and suspended objects above the plate as negative cases.

Release and stable placement remain separate diagnostics under this candidate. Official success can end an episode before a stricter stability test would pass; withholding the final reward until after termination would be a bug. An additional settling period would require its own explicitly different protocol. The proposal keeps the present success criterion, 600-control limit, and termination handling.

One-time milestones change the training objective and do not inherit the policy-invariance theorem. If potential-based shaping is tested separately, use the same `gamma=0.99` per chunk as advantage calculation, include remaining horizon in the finite-horizon state, and set potential to zero at every true terminal state, including the fixed task-budget endpoint. Continuing-task truncation would instead need its own bootstrap treatment. Do not combine both redesigns and describe their effect as a reward-only ablation of either one.

Any later reward comparison should start from the exact same SFT90 parent and fix the training interaction budget, learning rate, trainable actor parameters, sampler, controller, action execution, and evaluation settings. Change the reward alone. Register the new experiment and fresh held-out starts before running it; use matched training starts and multiple training seeds where the budget allows. The already reported 1501–1510 and 1601–1603 results have been inspected and cannot serve as an untouched test set for a tuned reward.

Log each reward component, official success, controls, actor/critic loss, entropy or the applicable sampler diagnostic, approximate KL, clipping fraction, actor gradients, and parameter changes. Compare official closed-loop success and interaction cost rather than shaped return alone. Preserve the present checkpoint and all paired results. A [separate motion analysis](experiments/2026-11-23/training-and-motion/README.md) documents command discontinuities; any smoothing experiment belongs in a different comparison.
