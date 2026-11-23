"""Run the VLABench-aligned fruit SFT pilot in the pinned native OpenPI backend."""
import argparse
import dataclasses
import functools
import hashlib
import json
from pathlib import Path
import time

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.models import model as openpi_model
from openpi.shared import normalize
from openpi.shared.nnx_utils import PathRegex, module_jit, state_map
from openpi.training.config import get_config
from openpi.training.weight_loaders import CheckpointWeightLoader
import orbax.checkpoint as ocp

from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import (AlignedPolicy, FruitEpisode, HORIZON, PADDED_DIM, TRAIN_SEEDS,
                                           VALIDATION_SEEDS, fit_stats, evaluate, paired_episodes)


class NativePolicy(AlignedPolicy):
    backend = 'OpenPI JAX'

    def __init__(self, model, stats, fork, seed=0):
        super().__init__(model, stats, fork, shortest_euler=True)
        self.key = jax.random.key(seed)
        self.sample = module_jit(model.sample_actions)

    def reset_rng(self, seed):
        self.key = jax.random.key(seed)

    def encode(self, raw):
        row = self.inputs(raw.copy())
        labels = row.pop('actions', None)
        observation = openpi_model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], row))
        return observation, None if labels is None else jnp.asarray(labels, dtype=jnp.float32)[None], row

    def infer(self, raw, noise=None):
        self.key, key = jax.random.split(self.key)
        observation, _, row = self.encode(raw)
        prediction = np.asarray(self.sample(key, observation, noise=noise))[0]
        return self.outputs({'actions': prediction, 'state': row['state']})


def initialize(config, base, key, trainable, optimizer):
    def make_state(rng):
        return nnx.state(config.create(rng))

    template = jax.eval_shape(make_state, key)
    params = CheckpointWeightLoader(str(base.absolute())).load(template.to_pure_dict())
    flat = traverse_util.flatten_dict(params)
    eligible = traverse_util.flatten_dict(template.filter(trainable).to_pure_dict())
    params = traverse_util.unflatten_dict({path: (value if path in eligible else np.asarray(value, dtype=jnp.bfloat16))
                                          for path, value in flat.items() if not isinstance(value, jax.ShapeDtypeStruct)})
    del flat

    @functools.partial(jax.jit, donate_argnums=(1,))
    def make_model(rng, weights):
        model = config.create(rng)
        graph, state = nnx.split(model)
        state.replace_by_pure_dict(weights)
        state = state_map(state, nnx.Not(trainable), lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return state, optimizer.init(state.filter(trainable))

    state, optimizer_state = make_model(key, params)
    graph = nnx.graphdef(nnx.eval_shape(config.create, key))
    return graph, state, optimizer_state


def train(args):
    import openpi

    if args.fork.resolve() not in Path(openpi.__file__).resolve().parents:
        raise RuntimeError('Prepend the pinned VLABench OpenPI fork to PYTHONPATH')
    if not jax.devices('gpu'):
        raise RuntimeError('Expected CUDA JAX')
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (42,)
    if args.updates > 1 and args.lerobot is None:
        if args.label_audit is None:
            raise ValueError('Supply the converted label audit for a full run')
        audit = {row['seed']: row['success'] for row in map(json.loads, args.label_audit.read_text().splitlines())}
        if set(audit) != set(TRAIN_SEEDS):
            raise ValueError('Expected all fruit replay outcomes')
        seeds = tuple(s for s in TRAIN_SEEDS if audit[s] and s not in VALIDATION_SEEDS)
    if args.lerobot is None:
        episodes = [FruitEpisode(args.cache / f'{seed}.h5') for seed in seeds]
    else:
        episodes = load_pose_split(args.lerobot, 'train')
        validation = load_pose_split(args.lerobot, 'validation')
        all_resets = [ep.initial_state for ep in episodes + validation]
        all_resets += [r['initial_state_sha256'] for r in paired_episodes(args.paired_episodes).values()]
        if len(set(all_resets)) != len(all_resets):
            raise ValueError('The dataset and development resets overlap')
        seeds = [episode.seed for episode in episodes]
        if set(seeds) - set(TRAIN_SEEDS) or set(seeds).intersection(VALIDATION_SEEDS):
            raise ValueError('The LeRobot export must only contain training resets')
    stats = fit_stats(episodes, shortest_euler=True)
    stats_bytes = normalize.serialize_json(stats).encode()
    (args.output / 'norm_stats.json').write_bytes(stats_bytes)
    model_config = dataclasses.replace(get_config('pi05_ft_vlabench_primitive').model,
                                     action_expert_variant='gemma_300m_lora')
    trainable = nnx.All(nnx.Param, PathRegex('.*(lora|action_(in|out)_proj|time_mlp_(in|out)).*'))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr))
    graph, state, opt_state = initialize(model_config, args.base, jax.random.key(263), trainable, optimizer)
    print(f'Loaded native base weights on {jax.devices()[0]}', flush=True)
    policy = NativePolicy(nnx.merge(graph, state), stats, args.fork)
    raw = episodes[0].row(34)
    obs, target, _ = policy.encode(raw)
    if target.shape != (1, HORIZON, PADDED_DIM):
        raise ValueError('Unexpected training target shape')
    params_initial = jax.device_get(state.filter(trainable).to_pure_dict())
    del policy

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt, rng, obs, target):
        model = nnx.merge(graph, params)
        def loss_fn(active_model):
            return active_model.compute_loss(rng, obs, target, train=False).mean()
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable))(model)
        subset = params.filter(trainable)
        updates, opt = optimizer.update(grads, opt, subset)
        nnx.update(model, optax.apply_updates(subset, updates))
        return nnx.state(model), opt, loss, optax.global_norm(grads)

    rng = np.random.default_rng(263)
    batch_policy = AlignedPolicy(None, stats, args.fork, shortest_euler=True)
    start = time.monotonic()
    with (args.output / 'updates.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, args.updates + 1):
            sample = episodes[int(rng.integers(len(episodes)))]
            frame = int(rng.integers(len(sample.state)))
            batch = batch_policy.inputs(sample.row(frame))
            target = jnp.asarray(batch.pop('actions'), dtype=jnp.float32)[None]
            obs = openpi_model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], batch))
            state, opt_state, loss, grad = step(state, opt_state, jax.random.fold_in(jax.random.key(263), update), obs, target)
            row = dict(update=update, loss=float(loss), grad_l2=float(grad), frame=frame, seed=sample.seed)
            if not np.isfinite(row['loss']) or not np.isfinite(row['grad_l2']):
                raise ValueError('Non-finite update')
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if update in (1, args.updates) or update % 50 == 0:
                print(json.dumps(row), flush=True)
    after = jax.device_get(state.filter(trainable).to_pure_dict())
    max_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(params_initial), strict=True))
    if max_delta <= 0:
        raise RuntimeError('The native policy weights did not change')
    policy = NativePolicy(nnx.merge(graph, state), stats, args.fork)
    raw.pop('actions')
    noise = jax.random.normal(jax.random.key(13), (1, HORIZON, PADDED_DIM))
    expected = policy.infer(raw, noise)['actions']
    checkpoint = (args.output / 'training-state').absolute()
    sampler_state = rng.bit_generator.state
    item = {'actor': after, 'optimizer_arrays': jax.device_get(jax.tree.leaves(opt_state))}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint, item)
        recovered = checkpointer.restore(checkpoint, item=item)
    for saved, loaded in zip(jax.tree.leaves(item), jax.tree.leaves(recovered), strict=True):
        np.testing.assert_array_equal(loaded, saved)
    (args.output / 'sampler-state.json').write_text(json.dumps(sampler_state, indent=2) + '\n')
    subset = state.filter(trainable)
    subset.replace_by_pure_dict(recovered['actor'])
    model = nnx.merge(graph, state)
    nnx.update(model, subset)
    restored = NativePolicy(model, stats, args.fork)
    actual = restored.infer(raw, noise)['actions']
    np.testing.assert_array_equal(actual, expected)
    summary = dict(backend='OpenPI JAX', updates=args.updates, train_seeds=seeds, max_actor_delta=max_delta,
                   train_frames=sum(len(episode.state) for episode in episodes), lr=args.lr,
                   input_format='LeRobot v2.1' if args.lerobot else 'HDF5 replay cache',
                   norm_stats_sha256=hashlib.sha256(stats_bytes).hexdigest(),
                   reload_max_action_difference=float(np.max(np.abs(actual - expected))), wall_seconds=time.monotonic() - start)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if args.updates > 1:
        evaluate(restored, args.output, paired_episodes(args.paired_episodes), 600, args.video, 'native-sft')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('base', 'fork', 'output', 'paired-episodes'):
        parser.add_argument('--' + path, required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache', type=Path)
    source.add_argument('--lerobot', type=Path)
    parser.add_argument('--label-audit', type=Path)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.updates < 1 or args.lr <= 0:
        parser.error('Use a positive update count and learning rate')
    train(args)


if __name__ == '__main__':
    main()
