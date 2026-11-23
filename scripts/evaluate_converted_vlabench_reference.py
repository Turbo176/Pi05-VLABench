"""Evaluate the published VLABench reference after conversion to RLinf."""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np
from openpi.shared import normalize
import torch

from evaluate_vlabench_reference import MODEL_REVISION, OPENPI_REVISION, TASK, build_episode, load_upstream_adapter
from pi05_runtime import build_model
from select_fruit import open_episode
from train_reference_aligned_fruit import AlignedPolicy, DIM, EXECUTE, HORIZON, PADDED_DIM, paired_episodes


class TorchReference(AlignedPolicy):
    def __init__(self, model, stats, fork, seed):
        super().__init__(model, stats, fork, shortest_euler=False)
        self.key = jax.random.key(seed)

    def infer(self, raw):
        self.key, sample_key = jax.random.split(self.key)
        noise = np.array(jax.random.normal(sample_key, (1, HORIZON, PADDED_DIM)))
        with torch.inference_mode():
            observation, _, row = self.encode(raw)
            prediction = self.model.sample_actions(observation, noise=torch.from_numpy(noise).to('cuda'))
        return self.outputs({'actions': prediction[0].float().cpu().numpy(), 'state': row['state']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('checkpoint', 'fork', 'paired-episodes', 'output'):
        parser.add_argument('--' + path, required=True, type=Path)
    parser.add_argument('--seed', type=int, action='append', choices=range(1201, 1205), required=True)
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        parser.error('Set JAX_PLATFORMS=cpu to generate the paired noise on CPU')
    reference = paired_episodes(args.paired_episodes)
    stats = normalize.deserialize_json((args.checkpoint / 'norm_stats.json').read_text())
    model, _ = build_model(args.checkpoint / 'model.safetensors', 'cuda', EXECUTE, action_horizon=HORIZON,
                           action_env_dim=DIM, discrete_state_input=False)
    adapter = load_upstream_adapter(args.fork)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for seed in args.seed:
            policy = TorchReference(model, stats, args.fork, seed)
            start = time.monotonic()
            with open_episode(seed, TASK) as env:
                result, rollout = build_episode(env, adapter(policy, replan_steps=EXECUTE), seed, 600, False)
                for key in ('initial_state_sha256', 'initial_rgb_sha256', 'instruction'):
                    if result[key] != reference[seed][key]:
                        raise RuntimeError(f'Seed {seed} does not match the published reference for {key}')
                for _ in rollout:
                    pass
            result.update(source='VLABench published Pi0.5 reference converted to RLinf', model_revision=MODEL_REVISION,
                          openpi_revision=OPENPI_REVISION, replan_steps=EXECUTE, max_control_steps=600,
                          control_hz=10, wall_seconds=time.monotonic() - start)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
