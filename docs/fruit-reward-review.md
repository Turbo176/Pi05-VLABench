# 🍐 Fruit PPO reward review

The latest [full-action fruit experiment](experiments/2026-11-24/fruit-full-action-v3/README.md) ran both the potential and milestone reward arms from the exact same SFT3000 parent. Development selection retained potential iteration two; final reference → SFT → PPO counts are **19/40 → 29/40 → 28/40** on forty reserved paired scenes. The complete comparison used 5,884 controls. Milestone iteration two exceeded the unchanged update guard and was rejected before saving; its cost remains included.

The implemented milestone pays +0.2 once for the instructed fruit grasped and held at least 4 cm above reset for three physical controls, then +0.8 once on official success. It was checked against the real official expert and tested for repeated events, wrong targets, pushes and chunk boundaries. [Reward event implementation](experiments/2026-11-24/fruit-full-action-v3/implementation/rl/source/milestones.py) · [Collector checks](experiments/2026-11-24/fruit-full-action-v3/implementation/rl/tests/test_milestone_collector.py)

The potential arm uses the objective's chunk discount and zero terminal potential. The milestone arm changes the objective; no policy-invariance claim is made for it. Neither reward guarantees PPO improvement. Both preserve the official success predicate and policy observation whitelist. All 33 CPU tests and saved actor-delta audits are retained in the report.

The review below records the earlier rewards and the literature reasoning that led to the follow-up. Its SFT90/SFT180 development and test results belong to those historical checkpoints.


Updated on November 24, 2026. This note explains the original experiment's reward and a separate follow-up. Both reward variants completed a full-model GPU smoke run, including critic warmup, actor updates, checkpoint saving and paired development evaluation. The legacy reward scored 10/16 and the corrected potential reward 5/16, against their SFT90 parent at 8/16. These are development results used for selection. The separate SFT180-to-PPO expansion completed a forty-scene test: reference 20/40, SFT180 17/40 and PPO1 16/40. It did not improve overall success. The frozen November 22 checkpoint, configuration and scores remain the historical baseline. [Follow-up results and costs](experiments/2026-11-24/fruit-recovery-v2/README.md)

## 🔎 Historical reward

The completed run used the reward in [`PPOClient.finish_chunk`](../scripts/ppo_reference_aligned_fruit.py) and [`fruit_potential`](../scripts/ppo_pi05.py):

```text
Phi = -min(gripper_to_target_3d_m, 0.7) - min(target_to_plate_xy_m, 0.5)
reward = official_success + 0.5 * (Phi_after_chunk - Phi_before_chunk)
```

Rewards are measured after each executed action chunk, normally five controls at 10 Hz. Official success ends the rollout, so its +1 is awarded once. The `0.5` is a coefficient on distance progress in meters; it is not a separate +0.5 lifting reward or a +0.5 placement reward. Simulator geometry is used only to calculate rewards and diagnostics, never as policy input.

The official `select_fruit` configuration uses `ContainCondition`; `FlatContainer.contain` checks the fruit origin against the plate region and a height threshold. It does not require opening the gripper or a settling period. A scored success can therefore end while the fruit is still held above the plate. All reported comparisons preserve that predicate; they are not measurements of released, stable placement. The pinned implementation is listed in [the upstream record](upstream.md).

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

## 🔧 Implemented preparation

[`train_fruit_ppo.py`](../scripts/train_fruit_ppo.py) is the next-run entry point. The two registered November 22 PPO scripts remain unchanged so their source hashes and historical results can still be verified.

| Item | Implemented behavior |
| --- | --- |
| `legacy-distance-v1` | Preserve `success + weight * (Phi_next - Phi_now)`, including the measured terminal potential |
| `discounted-terminal-potential-v2` | Use `success + weight * (gamma * Phi_next - Phi_now)`; the effective next potential is zero at success, environment termination or the finite task-budget endpoint |
| Critic warmup | Fit the value head to complete-episode discounted Monte Carlo returns; freeze actor parameters, check exact actor equality, and verify critic changes |
| PPO after warmup | Collect fresh on-policy episodes with the warmed critic before calculating GAE; warmup interactions count toward the budget |
| Replay checks | Compare all sampled/replayed likelihood coordinates before each update, including partially executed final chunks; reject nonfinite values |
| Update records | Actor loss, value MSE, sampled prefix likelihood-ratio diagnostics, clip fraction, gradients, actual actor changes and interaction counts |
| Scope | The current SFT90 parent has frozen vision/language features. The new entry rejects VLM LoRA, which needs a separately checked critic-gradient path |

The `approx_kl` field is a sampled likelihood-ratio statistic for the selected flow transition's executed action prefix. A success-shortened prefix depends on the sampled outcome. This statistic is a practical update guard, not an analytic full-chain KL or a KL between decoded robot-action distributions. Logs include executed-length counts. RLinf's `flow_sde` entropy placeholder is not plotted as measured policy entropy.

Seventeen CPU tests in [`test_fruit_ppo.py`](../tests/test_fruit_ppo.py) cover reward accounting, 100 randomized terminal-return identities, chunk discounting, clipping, warmup isolation, effective actor updates, post-update rollback, zero-reward actor skips, nonfinite ratios and partial replay coordinates. Replay failures are logged before raising, including coordinate errors that cancel in the aggregate. A post-step replay exception restores parameters and existing AdamW state before propagating the error. These are numerical and optimizer tests with small fixtures; they do not establish full-model training stability or a success-rate improvement. See the [follow-up commands](single-gpu.md#fruit-reward-follow-up).

The new optimizer checks the prefix-score diagnostic both before and after a minibatch step. An excessive post-step change restores actor/critic parameters and AdamW state. The full rollout batch is checked again before saving a checkpoint. An all-zero-reward batch can fit the critic but cannot produce a claimed actor update.

## 🖥️ Full-model GPU smoke validation

Both variants restarted from SFT90 with training starts 1301 and 1302 and the same sampler seed, learning rates, critic warmup and interaction limit. Each used 1,032 controls: 731 for critic warmup and 301 for a fresh on-policy collection. Both applied eight actor optimizer steps and saved a PPO checkpoint. All 216 action-expert LoRA tensors and eight action-head tensors changed in the corrected-reward run. The final sampled prefix-score diagnostic was 0.00360 for the corrected reward and 0.00488 for the legacy reward, below the configured 0.02 guard.

For the corrected reward, the critic's MSE on its warmup data decreased from 0.59479 to 0.03539, while the actor stayed exactly unchanged during warmup. This measures fitting of the sampled training returns, not generalization of the critic. The two successful fresh training rollouts were collected before the actor update; they are not evidence of a PPO success-rate improvement. Saved-policy development evaluation and the larger follow-up are separate steps.

## 🧪 Milestone design, subsequently implemented in the current report

The suggested split of +0.5 for lifting and +0.5 for placement is a reasonable candidate to compare. A more completion-focused starting candidate is:

| Event | One-time reward | Detection requirement |
| --- | ---: | --- |
| Securely grasp and lift the instructed fruit | +0.2 | Verified grasp/contact and sustained lift above its reset height; do not count a push, bounce, or wrong fruit |
| Complete the official fruit task | +0.8 | The unchanged official success predicate becomes true; do not wait for an additional release or stability condition |

The values were candidate hyperparameters and were subsequently used in the current comparison; that does not establish that they are optimal. The undiscounted event reward is at most 1 per episode; a success without the lift milestone receives 0.8. An episode-local ledger prevents repeated payments during lifting, hovering, releasing and regrasping. The current implementation checks persistence at every underlying control tick and freezes the lift threshold at 4 cm for three controls. Tests include wrong-object lifts, pushes, repeated grasps and chunk boundaries.

Release and stable placement remain separate diagnostics under this candidate. Official success can end an episode before a stricter stability test would pass; withholding the final reward until after termination would be a bug. An additional settling period would require its own explicitly different protocol. The proposal keeps the present success criterion, 600-control limit, and termination handling.

One-time milestones change the training objective and do not inherit the policy-invariance theorem. If potential-based shaping is tested separately, use the same `gamma=0.99` per chunk as advantage calculation, include remaining horizon in the finite-horizon state, and set potential to zero at every true terminal state, including the fixed task-budget endpoint. Continuing-task truncation would instead need its own bootstrap treatment. Do not combine both redesigns and describe their effect as a reward-only ablation of either one.

Any later reward comparison should start from the exact same SFT90 parent and fix the training interaction budget, learning rate, trainable actor parameters, sampler, controller, action execution, and evaluation settings. Change the reward alone. Register the new experiment and fresh held-out starts before running it; use matched training starts and multiple training seeds where the budget allows. The already reported 1501–1510 and 1601–1603 results have been inspected and cannot serve as an untouched test set for a tuned reward.

Log each reward component, official success, controls, actor/critic loss, entropy or the applicable sampler diagnostic, approximate KL, clipping fraction, actor gradients, and parameter changes. Compare official closed-loop success and interaction cost rather than shaped return alone. Preserve the present checkpoint and all paired results. A [separate motion analysis](experiments/2026-11-23/training-and-motion/README.md) documents command discontinuities; any smoothing experiment belongs in a different comparison.
