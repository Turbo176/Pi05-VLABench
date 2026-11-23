# Failure diagnosis

The initial comparison ran 117 held-out episodes before pausing. The base policy completed 36 trials, SFT 40, and PPO 41. Each reached the step budget with zero official successes. The [partial logs](paused-evaluation/) contain different numbers of trials per stage. One flower start has a camera hash difference. The [first-seed summary](../first-seed/summary.json) contains the verified paired pilot.

We measured action prediction error on each task's separate validation episode. We selected 16 evenly spaced frames per episode and compared ten controls forecast from the recorded observation with the expert's next ten controls. Column values are mean absolute error across the seven arm joints in radians.

| Task | Base with the absolute-action adapter | Initial SFT |
| --- | ---: | ---: |
| Fruit selection | 1.012 | 0.500 |
| Flower insertion | 1.125 | 0.547 |
| Fridge drink retrieval | 1.080 | 0.443 |

[Base measurements](base.json) · [Initial SFT measurements](sft.json) · [Absolute-action LoRA](absolute-lora/validation-actions.json) · [Relative-arm LoRA](relative-lora/validation-actions.json)

The first SFT run updated four projection and timestep modules. The follow-up updates rank-16 adapters in the attention and feed-forward layers of the action expert as well as the original four modules. A second run uses relative arm targets and absolute finger targets. The absolute and relative variants use the same training split, seed, update count, and learning rate. Both decode to absolute actuator targets for the simulator. A codec round-trip check on one window per task gave a maximum absolute error of 2.5e-7 when decoding from either the raw or normalized measured state.

On fruit starts 1201–1204, each 2,400-update run achieved 0/4 official successes. The official expert solved all four starts in 121–181 steps on the training server. The starts matched both SFT runs in initial physics and instruction. Each SFT run also reached 0/3 on the three-task development probe. The absolute-action model moved the target fruit by about 15 cm on start 1201 without qualifying as a success. The relative-arm model moved the target by less than 3 cm in all four fruit starts.

[Official expert trials](fruit-development-expert.json) · [Absolute-action fruit trials](absolute-lora/fruit-development.jsonl) · [Relative-arm fruit trials](relative-lora/fruit-development.jsonl)

## Single training episode

The saved expert controls from fruit seed 42 also succeeded on the training server in 132 steps. The initial joint and body poses exactly matched the local recording. The maximum difference between the stored expert controls and the float32 controls in the LeRobot data was 1.19e-7.

Both 600-update models trained on episode 0 reached the 600-step limit on the same start. The absolute model also failed when the action predictions were conditioned on recorded expert frames and joint measurements at ten-step intervals. The model moved the apple by at most 1.3 cm in that test. Its seven-joint mean error was 0.06–0.17 rad across the fourteen prediction windows. The trained actor still had material action error on its training inputs. Live tests with 5 or 20 executed actions per prediction and with the same initial denoising noise per call also reached their diagnostic time limits.

[Server replay](fruit-training-diagnostic/fruit-42-server-replay.json) · [Absolute-arm rollout](fruit-training-diagnostic/training-seed-42.jsonl) · [Relative-arm rollout](fruit-training-diagnostic/arm-delta/training-seed-42.jsonl) · [Observation and action chunk tests](fruit-training-diagnostic/replanning-diagnostic.jsonl)

The relative targets are measured from the joint pose at the start of a prediction. For the first arm joint, the 1st-to-99th percentile span was 0.232 rad at step 1 and 2.859 rad at step 50. The per-step calibration option fits a separate scale for each prediction step and shares the absolute finger scale across steps. Encoding and decoding one training window per task returned the original controls to within 3.0e-7.

[Per-step calibration codec check](per-step-normalization-check.json)

## Stronger weighting for the executed controls

A further 2,600 updates from the relative-arm checkpoint gave the first ten controls four times the loss weight of the other forty. The fruit arm error fell from 0.135 to 0.108 rad on the recorded validation windows. Official success on the four fruit development starts was 0/4. In the single training-scene diagnostic, the version emphasizing the executed controls reached the apple and pushed it across the table without a successful grasp. Tests with per-step calibration, language-model adapters, and deterministic one-step flow sampling also reached 0/1 on that training start.

[Multitask checkpoint summary](near-term-lora/summary.json) · [Fruit runs](near-term-lora/fruit-development.jsonl) · [Single-scene tests](fruit-training-diagnostic/)

## Expert recovery data

On fruit training seed 42, the near-term model executed 12–55 controls before the official expert took over. The expert solved seven of eight such starts. Re-executing the saved policy prefix followed by the expert's controls succeeded for all seven accepted traces with an exact match of the measured robot pose at takeover. The 737 labeled controls begin at the expert's first action.

[Recovery collection and replay results](fruit-expert-recovery.json)

The diagnostic warm-started the single-scene near-term model for 1,200 additional updates and sampled the recovery segments for 65% of training draws. On the original training start it moved the apple by 10.2 cm and reached the 600-step limit. [Recovery-trained rollout](fruit-training-diagnostic/recovery-sft/training-seed-42.jsonl)


## Additional training and recovery runs

The absolute-action continuation added rank-eight query and value adapters in the last four language-model layers. At 2,000, 4,000, 6,000, and 8,000 additional updates the policy scored 0/4 on fruit starts 1201–1204. The single-scene run with dense action-expert query and value updates also reached the step budget on training seed 42 after its evaluation was retried with available GPU memory.

[Continuation trials](absolute-longer/) · [Dense adaptation trial](fruit-training-diagnostic/dense-expert-seed-42.jsonl)

On eight other original fruit training starts, the near-term policy executed 25 or 40 controls before the official expert took over. Seven expert segments succeeded and replayed after the saved prefix. Their 916 labeled controls start at expert takeover. The pinned LeRobot v2.1 converter verified all 916 frames and 21 boundary checks for 50-step action windows. The failed start was seed 55. Fruit development starts 1201–1204 were excluded from collection.

[Collection report](fruit-multiscene-recovery.json) · [Dataset verification](fruit-multiscene-recovery-verification.json)

We ran two 1,500-update multitask continuations from the same checkpoint. Each used the same seed, task schedule, training export, recovery sampling, and optimizer. One supervised the nine actuator coordinates. The other also supervised the 23 zero-padded coordinates, matching the action dimensions used by OpenPI's flow loss. Each scored 0/4 on fruit development. The separate end-effector position loss diagnostic scored 0/1 on fruit training seed 42.

[Actuator-only loss](sft-arm-delta-recoveries-20261121/) · [Full action loss](sft-arm-delta-recoveries-full-loss-20261121/) · [Position loss diagnostic](fruit-fk-42-retry-20261121/)


## Replay sensitivity to arm errors

We re-executed two original fruit demonstrations on the GPU server. Each perturbed replay added a fixed Gaussian offset per arm joint for each block of ten controls. Finger commands, starting scenes, and the original trajectory lengths were fixed. The unperturbed controls succeeded on both scenes. The perturbation results were 4/6 at 0.01 rad, 2/6 at 0.03 rad, and 0/6 each at 0.06 and 0.10 rad. The standard deviation describes the injected offsets. This small diagnostic measures sensitivity to these specific perturbations.

[Replay measurements](fruit-training-diagnostic/fruit-expert-replay-tolerance.jsonl) · [Diagnostic script](fruit-training-diagnostic/diagnose-replay-tolerance.py)


## New fruit training scenes

The official expert solved all 24 new training starts, seeds 2010–2033. The saved controls reproduced official success and matched the captured physics trace on each start. The held-out fruit development starts remained 1201–1204. The LeRobot v2.1 exporter checked 3,565 frames and 72 action windows. The expanded run samples from 44 full fruit demonstrations, 20 flower demonstrations, and 20 fridge demonstrations while keeping the task schedule balanced.

[Expansion verification](fruit-expansion-verification.json)


Original OpenPI Orbax weights and the converted RLinf weights produced a mean absolute difference of 0.00925 in normalized actuator coordinates for the first ten controls of one recorded fruit input. The check fixed the tokenizer output, camera inputs, robot state, and initial diffusion noise. The native OpenPI path ran in JAX on CPU and the RLinf path ran in PyTorch on GPU. [Base conversion check](base-openpi-native-parity.json)

On fruit development start 1201 the full-loss policy displaced the target by 16.1 cm, with a maximum lift of 2.7 cm. The SFT and no-update RLinf runs matched the initial physics, initial RGB, instruction, action hashes, target lift, and failed outcome. Changing the fruit name while keeping the images, state, and sampling noise fixed affected the first action chunk by different amounts across starts. [Lift and handoff logs](sft-arm-delta-recoveries-full-loss-20261121/) · [Language sensitivity logs](fruit-language-action-sensitivity.log)


The expanded joint-action run completed 2,400 updates, saved its weights, and achieved 0/4 in the fruit development suite. Its four evaluation runs reached the 600-control budget. [Expanded joint-action run](sft-fruit-expanded-20261121/)

## End-effector pilot

The [VLABench reference adapter](https://github.com/Shiduo-zh/openpi/blob/788ef1f20e655de5bb40869fb7b7fcdd49732387/examples/vlabench/eval.py) predicts an end-effector pose and a gripper command. We converted stored joint targets to end-effector poses and re-executed them with the benchmark's inverse kinematics. Four expert replays across all three tasks reached official success with zero inverse-kinematics convergence failures. The policy pilot uses the same 84 demonstrations and the original OpenPI base weights. It retains the 50-action model horizon, ten-action execution, and 600-step budget. It estimates relative end-effector poses from recorded joint states and retains the two-camera input.

[Expert replay results](fruit-training-diagnostic/end-effector-replay.jsonl) · [Replay script](fruit-training-diagnostic/diagnose-end-effector-replay.py) · [Pilot script](train-ee-pi05-pilot.py)


Each end-effector model trained for 1,200 updates on the 84 demonstrations. The 50-action and 10-action prediction horizons both scored 0/4 on fruit, 0/1 on flower, and 0/1 on fridge development starts. The 50-action policy lifted the pear 7.1 cm on fruit seed 1202 before reaching the 600-control budget. The 10-action run had inverse-kinematics convergence failures on all four fruit starts.

[50-action pilot results](sft-ee-pilot-20261121/development.jsonl) · [10-action pilot results](sft-ee-horizon10-20261121/development.jsonl)

The published VLABench reference model is `VLABench/pi05-primitive-10task` at `f658a8bec801aab960fe0ceab370a123b31c1837`. Its inference files occupy approximately 12.44 GB. Its upstream runner uses three cameras and an end-effector adapter. A policy-side check on the same fruit development starts would establish how that reference behaves under the project's scoring and control budget. The verified published checkpoint was downloaded and evaluated on AutoDL.

[Reference release](https://huggingface.co/VLABench/pi05-primitive-10task/tree/f658a8bec801aab960fe0ceab370a123b31c1837) · [Reference adapter](https://github.com/Shiduo-zh/openpi/blob/788ef1f20e655de5bb40869fb7b7fcdd49732387/examples/vlabench/eval.py)

## Published VLABench checkpoint

The published VLABench π0.5 reference solved 3/4 fruit development starts. The expanded-data SFT checkpoint solved 0/4 on the same starts. Physics, audit RGB, and instructions matched. The reference ran through the upstream three-camera gripper-pose adapter. [Reference rollout and adapter checks](vlabench-reference/README.md)

The [reference-aligned fruit experiments](reference-aligned-fruit/README.md) replay 44 saved expert trajectories, verify the converted pose targets in MuJoCo, and report paired results for the two 600-update SFT runs.
