# Closed-loop evaluation smoke test

`run_episode` sends the policy the forward and wrist images, nine measured joint positions, and the instruction. It executes the first `--execute-steps` absolute actuator commands from each response through the official VLABench `env.step`. At each control tick it checks official task success, simulator termination, and the remaining budget. The result records the reset hashes, actions executed per call, reward, and stop reason.

## Local measurements

| Task, seed, prompt | Action source | Steps | Stop reason |
| --- | --- | ---: | --- |
| Fruit, 1001, official | Measured joint position hold | 3 | Step budget |
| Flower, 1001, official | Measured joint position hold | 3 | Step budget |
| Fridge drink, 1001, official | Measured joint position hold | 3 | Step budget |
| Fruit, 1101, pear | Saved official expert actions | 151 | Official success |
| Fruit, 1101, banana | Saved official expert actions | 136 | Official success |

The hold runs executed chunks of 2 and 1 control steps. The seed 1101 fruit runs matched the frozen initial physics and camera hashes and reproduced the recorded final physics state with zero maximum absolute error. These runs verify the simulator interface and scoring. Model evaluation is pending. [Machine-readable results](closed-loop-smoke.json)

## Reproduce in the local simulator

Run from the repository root in the existing WSL environment. A new run directory keeps each result and its action source together.

```bash
export PI05_SIM_SOURCE_ROOT="$(cat .local/active-sim-native)"
out=runs/policy-eval-smoke-v1-reproduction

for task in select_fruit insert_flower take_out_cool_drink; do
  bash scripts/run_wsl.sh evaluate --task "$task" --seed 1001 \
    --execute-steps 2 --hold-steps 3 --output "$out/$task-hold.json"
done

for mode in official alternate_target; do
  bash scripts/run_wsl.sh probe --task select_fruit --seed 1101 \
    --fruit-prompt "$mode" --check-fruit-expert \
    --expert-trace "$out/goal-$mode.h5" --output "$out/goal-$mode.json"
  bash scripts/run_wsl.sh evaluate --task select_fruit --seed 1101 \
    --suite fruit_language --prompt-mode "$mode" --execute-steps 8 \
    --replay-fixture "$out/goal-$mode.h5" --output "$out/fruit-$mode-replay.json"
done
```

The `run_episode` callback takes the mapped live observation and returns an action chunk in the simulator's units. Test checkpoint preprocessing, normalization, and action semantics on the GPU before running the [evaluation protocol](../configs/evaluation-v1.json). Choose the benchmark execution length before comparing checkpoints.
