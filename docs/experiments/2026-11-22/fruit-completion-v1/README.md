# 🍎 Fruit-first completion experiment

The fruit experiment ran on November 22–23, 2026 (Asia/Shanghai). Expert data, LeRobot export, SFT, a zero-update RL handoff, one real PPO iteration, and every registered paired evaluation are complete. This is a small pipeline experiment: ordinary success did not improve after PPO, and language tests regressed.

Fruit-first completion followed the shared 90-update SFT checkpoint's fruit 3/4 and flower 0/4 development results. The earlier two-task gate failed; flower and fridge remain incomplete.

The parent remains the real shared model trained on 64 complete demonstrations (47 fruit, 17 flower), 9,904 frames. A separate five-episode, 744-frame validation split (four fruit, one flower) was used. No single-demonstration diagnostic checkpoint initializes this experiment.

## 🔗 Checkpoint and interface

| Item | Frozen value |
| --- | --- |
| Published reference | VLABench pi05 primitive; already fine-tuned upstream |
| Reference revision | `f658a8bec801aab960fe0ceab370a123b31c1837` |
| Shared SFT | `two-task-conservative-90`, 90 optimizer updates |
| SFT SHA-256 | `998181bb8cc8386766aa5bec3302b938234ab5da2a24e81cef199a8fd560eef3` |
| Normalization SHA-256 | `bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f` |
| Simulation | Official VLABench Franka and task/controller; MuJoCo; MSAA disabled |
| Control | 10 Hz, predict 10 actions, execute five, maximum 600 controls |
| Policy interface | Forward, right and wrist RGB; literal instruction; measured robot state for the shared transform/pose-delta decoder; published `discrete_state_input=False` is preserved |
| Action | XYZ in meters, original Euler convention in radians, gripper command; unchanged official pose-to-controller adapter |

The fresh zero-update RL handoff passed on November 22. Three real observation windows produced exactly equal normalized predictions and shared decoded actions. Rollout/recomputed log probabilities agreed exactly. A fresh development-1201 simulation reproduced the parent's successful 135-control execution and complete action hash. This is a loading check, not a PPO result.

## 🧪 Registered experiment

The configuration freezes one PPO iteration with training starts 1301 and 1302, at most 1,200 environment controls, actor learning rate `2e-6`, critic learning rate `1e-4`, and flow-SDE noise 0.25. The VLA's action-expert LoRA and action heads are trainable; the VLM stays frozen. Rewards combine official success with the declared geometry-progress potential. Reward geometry, expert target truth and success labels never enter the policy input.

The PPO descendant is fixed before test results are read. Ordinary tests use reserved starts 1501–1510. Language tests use three separate scenes, 1601–1603, each with the official instruction, a paraphrase and a different target. These are small pilot samples. Development starts 1201–1204 remain a separate diagnostic group. The remaining original reserved seeds are outside this bounded experiment.

[PPO configuration](../../../../configs/ppo-fruit-completion-v1.json) · [Evaluation configuration](../../../../configs/evaluation-fruit-completion-v1.json) · [Shared SFT data and prior failures](../pose-robustness-v2/README.md)

## ▶️ Executed commands

The remote project root is `/root/autodl-tmp/Pi05-VLABench`. Repository `scripts/` and `configs/` are uploaded into `code/scripts/` and `code/configs/`; environments, upstream checkouts, assets, datasets and weights remain under ignored `.local/` and `runs/` paths. The standalone project on Windows remains `D:\Robotics\Pi05-VLABench`.

From the prepared remote project root, this command performs registration, handoff, PPO, development evaluation and the paired test sweep:

```bash
bash code/scripts/run-fruit-completion-v1.sh
```

The follow-up verifies the raw comparison and records the preselected ordinary-1501 start at all three stages, requiring the recorded action sequence to equal the scored execution:

```bash
bash code/scripts/run-fruit-completion-artifacts-v1.sh
```

The experiment deliberately rejects reusing an already registered output directory. The currently executed scripts are [pipeline](../pose-robustness-v2/run-fruit-completion-v1.sh) and [verification and videos](../pose-robustness-v2/run-fruit-completion-artifacts-v1.sh). No installation, asset download, model download, GitHub push or instance shutdown is part of these commands.

The actual run deliberately stopped redundant reference recomputation after a reuse audit, preserving the initial driver exit status and every partial case. Existing reference records matched all ten frozen reset manifests. Four fresh repeats (1501, 1502, 1503 and 1505) had exactly the same complete action hashes, success flags, control counts and IK counts. Three real observation windows also had identical inference transforms and decoded actions. The old configuration enabled label-only Euler wrapping, zero-initialized VLM adapters and frozen action-head gradients; these do not change reference inference when no learned adapters are restored. Original provenance fields and all 20 original reference records are retained; the protocol's numerical first ten seeds determine the subset.

The SFT/PPO evaluation then resumed without changing models, test cases or execution budgets:

```bash
bash code/scripts/run-fruit-completion-resume-v1.sh
```

[Executed resume driver](../pose-robustness-v2/run-fruit-completion-resume-v1.sh). The initial driver's intentional interruption is not a training failure or an omitted policy outcome.

## 📊 Results

The zero-update handoff, single PPO iteration, development comparison, ten ordinary held-out starts and all three paired language scenes are complete. The frozen models and test starts were not changed after viewing these results.

PPO consumed 731 environment controls across the two training starts, with one successful rollout. It performed 19 optimizer steps using 147 action chunks. Sampling and recomputed log probabilities differed by at most `3.8147e-6`. The action-expert LoRA changed 216/216 tensors and 6,930,323 elements; action heads changed 8/8 tensors and 2,140,024 elements. The maximum actor gradient norm before clipping was 77.079. The maximum parameter change from SFT was `3.5019e-5`. This establishes an actual VLA update, not only value-head training.

The rollout/update loop took 232.5 seconds and peaked at 7.33 GB of allocated CUDA memory on the instance reporting an RTX 4080 SUPER. Initialization and evaluation are additional costs. The saved PPO checkpoint SHA-256 is `0d2fac4c03db18ceac98c2344aec358713251e3b37cbcfa154349392c4ca0ad3`; both it and the parent SFT have verified local backups. One PPO iteration is a bounded pipeline experiment, not a claim of convergence.

### 📍 Development comparison

All four development cases finished with the same initialization hashes, policy RGB hashes, prompts, normalization, seeded evaluation sampler and execution budget.

| Seed | Target | Published reference | Shared SFT90 | Fruit PPO1 |
| --- | --- | --- | --- | --- |
| 1201 | Banana | Success / 120 controls | Success / 135 | Success / 174 |
| 1202 | Pear | Success / 136 controls | Success / 149 | Success / 179 |
| 1203 | Orange | Failure / 600 controls | Failure / 600 | Failure / 600 |
| 1204 | Pear | Failure / 600 controls | Success / 125 | Success / 123 |
| Total | | 2/4 | 3/4 | 3/4 |

The PPO policy preserved the same successful starts. It took longer on 1201/1202 and two fewer controls on 1204. The failed orange case had 164, 139 and 280 nonconverged IK controls for reference, SFT and PPO respectively. These observations are failure evidence, not proof of a unique training cause. Development success did not improve after PPO.

### 🎬 Fixed-scene autonomous videos

The recorded held-out scene is seed 1501, asking for the pear. The reference failed at 600 controls; SFT and PPO both succeeded at 160. Every video reproduced the corresponding quantitative episode's input hashes, complete action hash, official outcome and IK count. The video renderer uses 512-pixel display cameras for a 1024-pixel mosaic, while policy images retain the scored resolution. The final SFT control opens the gripper and satisfies official success; an earlier preview frame may still show the pear held.

[Published reference · failure](../../../../media/policy/fruit-completion-v1-base-1501.mp4?raw=true) · [Multitask SFT · success](../../../../media/policy/fruit-completion-v1-sft-1501.mp4?raw=true) · [PPO · success](../../../../media/policy/fruit-completion-v1-ppo-1501.mp4?raw=true)

### 📊 Ten ordinary held-out starts

| Policy | Successes | Rate | Wilson 95% interval | Total controls |
| --- | ---: | ---: | ---: | ---: |
| Published reference | 4/10 | 40% | 16.8–68.7% | 4,198 |
| Shared SFT90 | 5/10 | 50% | 23.7–76.3% | 3,861 |
| Fruit PPO1 | 5/10 | 50% | 23.7–76.3% | 3,737 |

SFT changed three reference failures to success (1501, 1504, 1506), but lost two reference successes (1503, 1507). PPO changed two SFT failures to success (1507, 1509), but lost two SFT successes (1504, 1506). PPO therefore did not improve total success in this fixed sample. All failed ordinary rollouts moved the target but did not satisfy the official goal within budget. Nonconverged IK controls totaled 1,304 / 1,067 / 769 for reference / SFT / PPO; this diagnostic is not itself a task-success measure.

[Every paired outcome, failure and confidence interval](ordinary-comparison.json). Ten starts and overlapping intervals do not establish a statistically reliable improvement. These results are retained without another PPO update or checkpoint selection.

### 🗣️ Paired language tests

All three policies see the same physical initial state and policy RGB in each scene. Only the instruction changes. Official prompts name pear, orange and pear on seeds 1601, 1602 and 1603; alternate prompts name lemon, lemon and banana. Paraphrases use “Please place the [fruit] on the plate.”

| Prompt group | Published reference | Shared SFT90 | Fruit PPO1 |
| --- | ---: | ---: | ---: |
| Official instruction | 1/3 | 1/3 | 0/3 |
| Paraphrase | 0/3 | 1/3 | 0/3 |
| Different target | 1/3 | 1/3 | 1/3 |

Each 1/3 result has Wilson 95% interval 6.1–79.2%; each 0/3 result has interval 0.0–56.1%. The three groups reuse three scenes and must not be pooled as nine independent scenes.

The reference succeeded on 1601/official and 1603/alternate. SFT retained these and succeeded on 1603/paraphrase. PPO retained only 1603/alternate, losing both the SFT official and paraphrase successes. The initial predicted action changed under both changed prompts in every scene and stage, but no stage completed both the original and alternate goal in the same scene. Changed actions alone do not prove correct language use. These results show a language-following limitation and PPO regression; there was no additional update or test-seed selection.

![Official success rates with Wilson 95% intervals](figures/success-rates.png)

### 📦 Evidence and completion checks

[Complete quantitative comparison](comparison.json) contains all stage scores, confidence intervals, paired improvements/regressions, language instructions and outcomes, PPO rollout costs, and actor update evidence. [Saved-checkpoint audit](saved-checkpoint-update-audit.json) independently compared all 224 saved actor tensors against the exact SFT parent and verified all 232 optimizer states reached step 19. [Split audit](split-isolation.json) checks complete-episode and initial-state separation.

Every raw evaluation episode, including failures, is retained alongside source snapshots, frozen configurations, training updates and original scheduling/interruption logs. The [export manifest](evidence-export.json) lists file sizes and SHA-256 hashes. All 201 exported files passed the [local transfer verification](local-evidence-verification.json). Model weights, asset libraries, source datasets and credentials are excluded from this evidence export; SFT/PPO weights have separate ignored local backups.

[Zero-update handoff](handoff.json) · [PPO optimizer and gradient evidence](ppo/iteration-1.json) · [PPO training rollouts](ppo/rollouts.jsonl) · [SFT training summary](sft-parent/summary.json) · [Final local checks](final-local-check.json)

The final artifact job initially failed while plotting zero-success Wilson intervals: floating point rounding put the lower endpoint slightly above zero, producing a negative error-bar length. The plotting fix clamps only displayed error lengths to zero. Quantitative records and intervals remain unchanged. The [recovery check](artifacts-recovery.json) rehashed all nine scored input files, both figures, the three videos and their executed recorder sources, then compared every recorded rollout with its scored counterpart. The failed exit status and traceback remain archived.

After the plotting fix, these commands were actually executed in the prepared remote environment:

```bash
source .local/reference-env.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
root=runs/pose-robustness-v2/fruit-completion-v1
"$py" code/scripts/plot_fruit_completion.py --comparison "$root/comparison.json" --output "$root/figures"
"$py" code/scripts/finalize_fruit_completion_artifacts.py --root "$root"
"$py" code/scripts/export_fruit_completion_evidence.py --root "$root"
```

The finalizer and exporter reject incomplete inputs; their completed evidence outputs are preserved rather than overwritten on reruns. The instance was left running after experiment completion. Repository publication is separate from these experiment commands; weights, datasets and assets stay excluded from Git.
