"""Fit π0.5 on replay-verified fruit episodes using the published VLABench I/O."""
import argparse
from contextlib import contextmanager, nullcontext
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import time

from dm_control.rl.control import PhysicsError
import h5py
import numpy as np
from openpi import transforms
from openpi.shared import normalize
from openpi.training.config import get_config
from rlinf.models.embodiment.openpi.modules.model import Observation
import torch

from convert_lerobot import POSE_TASKS
from evaluate_vlabench_reference import MODEL_REVISION, OPENPI_REVISION, TASK, build_episode, load_upstream_adapter
from reference_pose_lerobot import load_pose_split
from select_fruit import open_episode, set_fruit_prompt, video
from summarize_pi05 import initial_rgb_key
from pi05_runtime import (ACTOR_MODULES, actor_delta, build_model, file_sha256, is_action_expert_parameter,
                          is_actor_parameter, restore_trainable, trainable_snapshot)

IMAGE_KEYS = ('observation/image', 'observation/second_image', 'observation/wrist_image')
HORIZON, EXECUTE, DIM, PADDED_DIM = 10, 5, 7, 32
TRAIN_SEEDS = (42, 53, 54, 55, 56, 57, 58, *range(63, 76), *range(2010, 2034))
EXPANSION_TRAIN_SEEDS = tuple(range(2050, 2120))
VALIDATION_SEEDS = (55, 69, 2012, 2027)
DEVELOPMENT_SEEDS = (1201, 1202, 1203, 1204)
ADAPTER_RANK, VLM_RANK = 16, 8


class FruitEpisode:
    task = TASK

    def __init__(self, path):
        self.path = path
        with h5py.File(path, 'r') as source:
            self.task = str(source.attrs.get('task', TASK))
            self.seed = int(source.attrs['seed'])
            self.prompt = str(source.attrs['instruction'])
            self.initial_state = str(source.attrs['initial_state_sha256'])
            self.render_samples = int(source.attrs['render_samples']) if 'render_samples' in source.attrs else None
            self.policy_prefix_steps = len(source['policy_prefix']) if 'policy_prefix' in source else 0
            self.state = source['state'][:]
            self.action = source['actions'][:]
            if (not source['success'][-1] or len(source['rgb']) != len(self.state)
                    or self.state.shape != self.action.shape or self.state.shape[1] != DIM):
                raise ValueError(f'Invalid aligned trajectory {path}')
        # Same episode-end padding as the project's LeRobot reader.
        index = np.minimum(np.arange(len(self.state))[:, None] + np.arange(HORIZON), len(self.state) - 1)
        self.chunks = self.action[index]

    def row(self, frame):
        with h5py.File(self.path, 'r') as source:
            views = source['rgb'][frame]
        return {**dict(zip(IMAGE_KEYS, views, strict=True)), 'observation/state': self.state[frame],
                'prompt': self.prompt, 'actions': self.chunks[frame]}


def wrap_euler(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def fit_stats(episodes, *, shortest_euler):
    state = np.concatenate([item.state for item in episodes])
    action = np.concatenate([item.chunks - np.pad(item.state[:, None, :6], ((0, 0), (0, 0), (0, 1)))
                             for item in episodes]).reshape(-1, DIM)
    if shortest_euler:
        action[:, 3:6] = wrap_euler(action[:, 3:6])
    stats = {}
    for key, values in (('state', state), ('actions', action)):
        values = np.pad(values, ((0, 0), (0, PADDED_DIM - DIM)))
        low, high = np.quantile(values, [0.01, 0.99], axis=0)
        stats[key] = normalize.NormStats(mean=values.mean(axis=0), std=values.std(axis=0), q01=low, q99=high)
    return stats


class AlignedPolicy:
    backend = 'RLinf'

    def __init__(self, model, stats, fork, *, shortest_euler, discrete_state=False, reference_noise=False):
        cfg = get_config('pi05_ft_vlabench_primitive')
        model_config = dataclasses.replace(cfg.model, discrete_state_input=discrete_state)
        data = cfg.data.create(cfg.assets_dirs, model_config)
        if cfg.model.discrete_state_input or model_config.action_horizon != HORIZON or not data.use_quantile_norm:
            raise RuntimeError('The upstream model configuration changed')
        self.model, self.fork = model, fork
        self.data_inputs = transforms.compose(data.data_transforms.inputs)
        self.model_inputs = transforms.compose(data.model_transforms.inputs)
        self.shortest_euler = shortest_euler
        self.discrete_state = discrete_state
        self.inputs = transforms.compose([self.prepare_labels, transforms.Normalize(stats, use_quantiles=True),
                                          self.model_inputs])
        self.outputs = transforms.compose([transforms.Unnormalize(stats, use_quantiles=True),
                                           *data.data_transforms.outputs])
        self.generator = None
        self.reference_noise = reference_noise
        self.reference_key = None

    def reset_rng(self, seed):
        if self.reference_noise:
            import jax

            if jax.default_backend() != 'cpu':
                raise RuntimeError('Set JAX_PLATFORMS=cpu for paired reference noise')
            self.reference_key = jax.random.key(seed)
        else:
            self.generator = torch.Generator(device='cuda').manual_seed(seed)

    def prepare_labels(self, raw):
        row = self.data_inputs(raw)
        if self.shortest_euler and 'actions' in row:
            row['actions'][..., 3:6] = wrap_euler(row['actions'][..., 3:6])
        return row

    def encode(self, raw):
        required = {*IMAGE_KEYS, 'prompt', 'observation/state'}
        if raw.keys() not in (required, required | {'actions'}):
            raise ValueError('Unexpected policy input fields')
        row = self.inputs(raw.copy())
        def tensor(value):
            return torch.from_numpy(np.array(value, copy=True))[None].to('cuda')
        observation = Observation.from_dict(dict(
            image={k: tensor(v) for k, v in row['image'].items()},
            image_mask={k: tensor(v) for k, v in row['image_mask'].items()},
            state=tensor(row['state'].astype(np.float32)),
            tokenized_prompt=tensor(row['tokenized_prompt'].astype(np.int64)),
            tokenized_prompt_mask=tensor(row['tokenized_prompt_mask']),
        ))
        labels = tensor(row['actions'].astype(np.float32)) if 'actions' in row else None
        return observation, labels, row

    def reference_sample_noise(self):
        if self.reference_key is None:
            return None
        import jax

        self.reference_key, key = jax.random.split(self.reference_key)
        array = np.array(jax.random.normal(key, (1, HORIZON, PADDED_DIM)))
        return torch.from_numpy(array).to('cuda')

    def encode_training_batch(self, rows):
        encoded = [self.encode(raw) for raw in rows]
        if not encoded or any(labels is None for _, labels, _ in encoded):
            raise ValueError('Training batches need a label for every observation')
        observations = [item[0] for item in encoded]
        batch = Observation.from_dict(dict(
            image={key: torch.cat([obs.images[key] for obs in observations]) for key in observations[0].images},
            image_mask={key: torch.cat([obs.image_masks[key] for obs in observations])
                        for key in observations[0].image_masks},
            state=torch.cat([obs.state for obs in observations]),
            tokenized_prompt=torch.cat([obs.tokenized_prompt for obs in observations]),
            tokenized_prompt_mask=torch.cat([obs.tokenized_prompt_mask for obs in observations]),
        ))
        return batch, torch.cat([item[1] for item in encoded])

    def infer(self, raw):
        with torch.inference_mode():
            obs, _, row = self.encode(raw)
            prediction = self.model.sample_actions(obs, num_steps=10, rng=self.generator,
                                                   noise=self.reference_sample_noise())
            return self.outputs({'actions': prediction[0].float().cpu().numpy(), 'state': row['state']})


def check_inputs(policy, raw):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

    obs, labels, row = policy.encode(raw)
    changed = {**raw, 'observation/state': raw['observation/state'] + np.arange(DIM, dtype=np.float32)}
    other, _, _ = policy.encode(changed)
    assert torch.equal(obs.tokenized_prompt, other.tokenized_prompt) != policy.discrete_state
    assert not torch.equal(obs.state, other.state)
    assert all(bool(v.item()) for v in obs.image_masks.values()) and labels.shape == (1, HORIZON, PADDED_DIM)
    obs = preprocess_observation(obs, train=False)
    for key, image in row['image'].items():
        actual = obs.images[key][0].float().cpu().numpy()
        np.testing.assert_allclose(actual, image.astype(np.float32) / 127.5 - 1, rtol=0, atol=2e-7)
    recovered = policy.outputs({'state': row['state'], 'actions': row['actions'].copy()})['actions']
    difference = recovered - raw['actions']
    if policy.shortest_euler:
        difference[:, 3:6] = wrap_euler(difference[:, 3:6])
    round_trip = float(np.max(np.abs(difference)))
    if round_trip > 1e-5:
        raise ValueError(f'Normalized actions did not round trip: {round_trip:g}')
    noise = torch.randn(1, HORIZON, PADDED_DIM, device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(450))
    with torch.inference_mode():
        original, _, _ = policy.encode(raw)
        before = policy.model.sample_actions(original, noise=noise)
        after = policy.model.sample_actions(other, noise=noise)
    state_effect = float((before - after).abs().max())
    if bool(state_effect) != policy.discrete_state:
        raise RuntimeError('Observation state conditioning differs from the selected interface')
    return {'round_trip_max': round_trip, 'normalized_state_ablation_max': state_effect,
            'active_image_keys': list(obs.images), 'normalized_padding_value': float(labels[0, 0, -1])}


def validation_mae(policy, episodes):
    all_errors = []
    for episode in episodes:
        for frame in np.linspace(0, len(episode.state) - 1, 6, dtype=int):
            row = episode.row(frame)
            actions = row.pop('actions')
            policy.generator = torch.Generator(device='cuda').manual_seed(500 + int(frame))
            errors = policy.infer(row)['actions'][:EXECUTE] - actions[:EXECUTE]
            errors[:, 3:6] = wrap_euler(errors[:, 3:6])
            all_errors.append(errors)
    errors = np.abs(np.concatenate(all_errors))
    return dict(windows=len(all_errors), xyz_mae_m=float(errors[:, :3].mean()),
                euler_mae_rad=float(errors[:, 3:6].mean()), gripper_mae=float(errors[:, 6].mean()))


@contextmanager
def matched_rollout(seed, task, prompt_mode, agent, expected, limit, record, trace, *, render_samples=None):
    for attempt in range(1, 5):
        try:
            env = open_episode(seed, task, render_samples=render_samples)
        except PhysicsError as error:
            logging.warning('Reset %s/%d (try %d): %s', task, seed, attempt, error)
            continue
        with env:
            set_fruit_prompt(env, prompt_mode)
            row, replay = build_episode(env, agent, seed, limit, record,
                                        task=task, prompt_mode=prompt_mode, trace=trace, record_physics_failure=True)
            rgb_key = initial_rgb_key(row, expected)
            for key in ('initial_state_sha256', 'instruction', 'prompt_mode'):
                if row[key] != expected[key]:
                    raise ValueError(f'Evaluation seed {seed} has a different {key}')
            if row[rgb_key] != expected[rgb_key]:
                logging.warning('Reset %s/%d (try %d): %s %s did not match expected %s',
                                task, seed, attempt, rgb_key, row[rgb_key], expected[rgb_key])
                replay.close()
                continue
            row['reset_attempts'] = attempt
            yield row, replay
            return
    raise RuntimeError(f'Could not reproduce the pinned reset for {task} seed {seed} after four attempts')


def evaluate(policy, output, expected, limit, record, stage, *, seeds=DEVELOPMENT_SEEDS, trace=False,
             execute=EXECUTE, task=TASK, prompt_mode='official', resume=False, render_samples=None):
    seeds = tuple(seeds)
    if len(set(seeds)) != len(seeds) or not set(seeds) <= expected.keys():
        raise ValueError('Requested seeds must be unique and present in the paired reset records')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'episodes.jsonl'
    run_identity = dict(getattr(policy, 'provenance', {}), stage=stage, backend=policy.backend,
                        openpi_revision=OPENPI_REVISION, control_hz=10, replan_steps=execute,
                        max_control_steps=limit,
                        sampler='JAX seeded noise' if getattr(policy, 'reference_noise', False) else 'policy default')
    if render_samples is not None:
        if any(row.get('render_samples') != render_samples for row in expected.values()):
            raise ValueError('The reset manifests use a different render sample count')
        run_identity['render_samples'] = render_samples
    completed = set()
    if resume and path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                seed = row['seed']
                if seed not in seeds or seed in completed:
                    raise ValueError(f'Unexpected or duplicate previously recorded seed {seed}')
                reset = expected[seed]
                keys = ('initial_state_sha256', 'instruction', 'prompt_mode', initial_rgb_key(row, reset))
                if (row['task'] != task or any(row[k] != reset[k] for k in keys)
                        or any(row[k] != value for k, value in run_identity.items())):
                    raise ValueError(f'Seed {seed} was run with a different evaluation configuration')
                completed.add(seed)
    adapter = load_upstream_adapter(policy.fork)
    with path.open('a' if resume else 'x', encoding='utf-8') as stream:
        for seed in seeds:
            if seed in completed:
                continue
            start = time.monotonic()
            policy.reset_rng(seed)
            trace_path = output / f'{task}-{seed}-controls.jsonl'
            trace_context = trace_path.open('x', encoding='utf-8') if trace else nullcontext()
            with trace_context as control_trace, matched_rollout(
                    seed, task, prompt_mode, adapter(policy, replan_steps=execute), expected[seed],
                    limit, record, control_trace, render_samples=render_samples) as (row, replay):
                if record:
                    video(output / f'{task}-{seed}.mp4', replay, 10,
                          f'Pi0.5 {task} {stage.upper()} | seed {seed}', row['instruction'])
                else:
                    for _ in replay:
                        pass
            row.update(run_identity, wall_seconds=time.monotonic() - start)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(json.dumps(row), flush=True)


def paired_episodes(path):
    return {row['seed']: row for row in map(json.loads, path.read_text().splitlines())}


def load_aligned_policy(run, base, fork, stage='sft', ppo_weights=None):
    stats_bytes = (run / 'norm_stats.json').read_bytes()
    stats = normalize.deserialize_json(stats_bytes.decode())
    checkpoint = torch.load(run / 'sft.pt', map_location='cpu', weights_only=True)
    info = checkpoint['metadata']
    base_sha256 = file_sha256(base)
    if info.get('base_sha256', base_sha256) != base_sha256:
        raise ValueError('The supplied base weights differ from the SFT training parent')
    if (hashlib.sha256(stats_bytes).hexdigest() != info['norm_stats_sha256']
            or info['horizon'] != HORIZON or info['execute'] != EXECUTE):
        raise ValueError('Unexpected normalization or action horizon in the SFT checkpoint')
    torch.manual_seed(info['seed'])
    model, _ = build_model(base, 'cuda', EXECUTE, lora_rank=info['actor_lora_rank'],
                           vlm_lora_rank=info['vlm_lora_rank'], action_horizon=HORIZON,
                           action_env_dim=DIM, discrete_state_input=info.get('discrete_state', False),
                           train_action_heads=info.get('train_action_heads', True),
                           train_full_action_expert=info.get('train_full_action_expert', False))
    sft_sha256 = hashlib.sha256((run / 'sft.pt').read_bytes()).hexdigest()
    if stage == 'sft':
        restore_trainable(model, checkpoint['policy'])
    elif stage == 'ppo':
        ppo = torch.load(ppo_weights, map_location='cpu', weights_only=True)
        if (ppo['ppo']['sft_sha256'] != sft_sha256
                or ppo['ppo']['norm_stats_sha256'] != info['norm_stats_sha256']):
            raise ValueError('The PPO checkpoint does not descend from the specified SFT checkpoint')
        restore_trainable(model, {name: value for name, value in ppo['policy'].items() if is_actor_parameter(name)})
    policy = AlignedPolicy(model, stats, fork, shortest_euler=info.get('shortest_euler', False),
                           discrete_state=info.get('discrete_state', False),
                           reference_noise=bool(info.get('reference_model_revision')))
    policy.training_render_samples = info.get('render_samples')
    policy.provenance = dict(sft_sha256=sft_sha256,
                             norm_stats_sha256=info['norm_stats_sha256'],
                             reference_model_revision=info.get('reference_model_revision'),
                             base_sha256=base_sha256)
    if info.get('reference_input_profile'):
        policy.provenance['reference_input_profile'] = info['reference_input_profile']
    if info.get('reference_action_profile'):
        policy.provenance['reference_action_profile'] = info['reference_action_profile']
    if info.get('training_parameter_profile'):
        policy.provenance['training_parameter_profile'] = info['training_parameter_profile']
    if info.get('diagnostic_parameter_profile'):
        policy.provenance['diagnostic_parameter_profile'] = info['diagnostic_parameter_profile']
    if stage != 'base' and info.get('training_scope'):
        policy.provenance.update(training_scope=info['training_scope'], diagnostic_case=info['diagnostic_case'])
    if stage == 'base':
        # The checkpoint supplies configuration; its trained parameters are not restored.
        policy.provenance['configuration_source_sha256'] = policy.provenance.pop('sft_sha256')
    if stage == 'ppo':
        policy.provenance['ppo_sha256'] = hashlib.sha256(ppo_weights.read_bytes()).hexdigest()
    return policy


def evaluate_checkpoint(args):
    policy = load_aligned_policy(args.load_run, args.base, args.fork, args.eval_stage, args.ppo_weights)
    expected = paired_episodes(args.paired_episodes)
    if any(row.get('task', TASK) != args.eval_task or row.get('prompt_mode', 'official') != args.prompt_mode
           for row in expected.values()):
        raise ValueError('The paired reset records must all belong to the requested task')
    seeds = args.eval_seed or sorted(expected)
    render_samples = args.render_samples if args.render_samples is not None else policy.training_render_samples
    evaluate(policy, args.output, expected, args.max_steps, args.video, args.eval_stage,
             seeds=seeds, trace=args.trace_controls, execute=args.eval_execute, task=args.eval_task,
             prompt_mode=args.prompt_mode, resume=args.resume_evaluation, render_samples=render_samples)


def flow_loss(model, observation, labels, action_loss, *, image_augmentation=True, gripper_loss_weight=1.0):
    per_control = model.compute_loss(observation, labels, train=image_augmentation)
    if action_loss == 'executed-poses':
        per_control = per_control[:, :EXECUTE, :DIM]
    if gripper_loss_weight == 1.0:
        return per_control.mean()
    weights = per_control.new_ones(per_control.shape[-1])
    weights[6] = gripper_loss_weight
    return ((per_control * weights).sum(dim=-1) / weights.sum()).mean()


def check_training_augmentation(policy, raw):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

    with torch.no_grad():
        observation, _, _ = policy.encode(raw)
        plain = preprocess_observation(observation, train=False)
        augmented = preprocess_observation(observation, train=True, rng=torch.Generator().manual_seed(107))
        deltas = {key: float((augmented.images[key] - image).abs().max())
                  for key, image in plain.images.items()}
        if not all(value > 0 for value in deltas.values()):
            raise RuntimeError('Training image augmentation did not change every active camera')
        for key in ('state', 'tokenized_prompt', 'tokenized_prompt_mask'):
            if not torch.equal(getattr(plain, key), getattr(augmented, key)):
                raise RuntimeError(f'Image augmentation changed {key}')
    return deltas


def vlm_adapter_parameter(name):
    return name.startswith('llm.') and '.0.q_lora_' in name


def save_training_checkpoint(output, model, optimizer, info, rng):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'sft.pt'
    torch.save(dict(policy=trainable_snapshot(model), optimizer=optimizer.state_dict(), metadata=dict(info),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    sampler_rng=rng.bit_generator.state), path)
    return path


def training_pools(episodes, *, multitask, recovery_share):
    """Keep task balancing outside the within-task expert/correction choice."""
    present_tasks = {ep.task for ep in episodes}
    tasks = (tuple(task for task in POSE_TASKS if task in present_tasks) if multitask
             else tuple(sorted(present_tasks)))
    pools = {task: dict(expert_demo=[ep for ep in episodes if ep.task == task and not ep.policy_prefix_steps],
                        expert_correction=[ep for ep in episodes if ep.task == task and ep.policy_prefix_steps])
             for task in tasks}
    if any(not group['expert_demo'] for group in pools.values()):
        raise ValueError('Every task requires complete expert demonstrations')
    if recovery_share and not any(group['expert_correction'] for group in pools.values()):
        raise ValueError('Recovery sampling requires verified expert corrections')
    return pools


def train(args):
    full_expert = bool(args.full_action_expert_diagnostic or args.full_action_expert)
    validation_seeds = () if args.smoke else VALIDATION_SEEDS
    rejected = []
    if args.lerobot is not None:
        train_set = load_pose_split(args.lerobot, 'train')
        for extra_dataset in args.extra_lerobot:
            train_set.extend(load_pose_split(extra_dataset, 'train'))
        validation = [] if args.smoke else load_pose_split(args.lerobot, 'validation')
        if args.training_tasks:
            train_set = [episode for episode in train_set if episode.task in args.training_tasks]
            validation = [episode for episode in validation if episode.task in args.training_tasks]
        if args.diagnostic_case:
            diagnostic_task, diagnostic_seed = args.diagnostic_case.rsplit(':', 1)
            train_set = [episode for episode in train_set
                         if (episode.task, episode.seed) == (diagnostic_task, int(diagnostic_seed))]
            if len(train_set) != 1:
                raise ValueError('The diagnostic must select exactly one existing training episode')
            validation = []
        expected_tasks = (set(args.training_tasks or POSE_TASKS) if args.multitask else
                          {args.diagnostic_case.rsplit(':', 1)[0] if args.diagnostic_case else TASK})
        if {episode.task for episode in train_set} != expected_tasks:
            raise ValueError(f'Expected training data for {sorted(expected_tasks)}')
        if validation and {episode.task for episode in validation} != expected_tasks:
            raise ValueError('Validation and training must cover the same tasks')
        train_ids = {(episode.task, episode.seed) for episode in train_set}
        validation_ids = {(episode.task, episode.seed) for episode in validation}
        if len(train_ids) != len(train_set) or train_ids.intersection(validation_ids):
            raise ValueError('The training and validation task/seed pairs overlap')
        train_seeds = tuple(episode.seed for episode in train_set if episode.task == TASK)
        validation_seeds = tuple(episode.seed for episode in validation if episode.task == TASK)
        allowed = set(TRAIN_SEEDS + EXPANSION_TRAIN_SEEDS) - set(VALIDATION_SEEDS)
        if set(train_seeds) - allowed:
            raise ValueError('Expected curated fruit training seeds')
        if args.multitask and any(episode.seed >= 1000 for episode in train_set + validation if episode.task != TASK):
            raise ValueError('Keep the held-out task seeds outside the training dataset')
    elif args.smoke:
        train_seeds = (42,)
    else:
        audit_rows = [json.loads(line) for line in args.label_audit.read_text().splitlines()]
        audited = {row['seed']: row['success'] for row in audit_rows}
        if len(audit_rows) != len(TRAIN_SEEDS) or set(audited) != set(TRAIN_SEEDS):
            raise ValueError('Expected exactly one label replay outcome per fruit episode')
        rejected = [seed for seed in TRAIN_SEEDS if not audited[seed] and seed not in validation_seeds]
        train_seeds = tuple(seed for seed in TRAIN_SEEDS if seed not in validation_seeds and audited[seed])
    if args.lerobot is None:
        train_set = [FruitEpisode(args.cache / f'{seed}.h5') for seed in train_seeds]
        validation = [FruitEpisode(args.cache / f'{seed}.h5') for seed in validation_seeds]
    if args.render_samples is not None and any(getattr(ep, 'render_samples', None) != args.render_samples
                                              for ep in train_set + validation):
        raise ValueError('The training and validation data must declare the requested render sample count')
    expected = paired_episodes(args.paired_episodes)
    hashes = [ep.initial_state for ep in train_set + validation] + [expected[s]['initial_state_sha256'] for s in DEVELOPMENT_SEEDS]
    if len(hashes) != len(set(hashes)):
        raise ValueError('The training, validation and development resets overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    initializer = None
    init_metadata = {}
    if args.reference_start:
        stats_text = (args.reference_start / 'norm_stats.json').read_text(encoding='utf-8')
        stats = normalize.deserialize_json(stats_text)
        if args.action_norm_diagnostic or args.refit_action_quantiles:
            stats['actions'] = fit_stats(train_set, shortest_euler=args.shortest_euler)['actions']
            stats_text = normalize.serialize_json(stats)
    elif args.init_run is None:
        stats = fit_stats(train_set, shortest_euler=args.shortest_euler)
        stats_text = normalize.serialize_json(stats)
    else:
        initializer_path = args.init_run / 'sft.pt'
        initializer = torch.load(initializer_path, map_location='cpu', weights_only=True)
        stats_bytes = (args.init_run / 'norm_stats.json').read_bytes()
        init_metadata = initializer['metadata']
        if (init_metadata['norm_stats_sha256'] != hashlib.sha256(stats_bytes).hexdigest()
                or init_metadata['horizon'] != HORIZON or init_metadata['execute'] != EXECUTE
                or init_metadata['actor_lora_rank'] != args.actor_lora_rank
                or init_metadata['vlm_lora_rank'] != args.vlm_lora_rank
                or init_metadata.get('shortest_euler', False) != args.shortest_euler
                or init_metadata.get('train_action_heads', True) != args.train_action_heads
                or init_metadata.get('discrete_state', False) != args.discrete_state
                or init_metadata.get('train_full_action_expert', False) != full_expert):
            raise ValueError('The initialization checkpoint has a different action or observation interface')
        stats_text = stats_bytes.decode()
        stats = normalize.deserialize_json(stats_text)
    (args.output / 'norm_stats.json').write_text(stats_text, encoding='utf-8')
    model, _ = build_model(args.base, 'cuda', EXECUTE, lora_rank=args.actor_lora_rank,
                           vlm_lora_rank=args.vlm_lora_rank,
                           action_horizon=HORIZON, action_env_dim=DIM, discrete_state_input=args.discrete_state,
                           train_action_heads=args.train_action_heads,
                           train_full_action_expert=full_expert)
    frozen_heads = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                    if name.startswith(ACTOR_MODULES)} if not args.train_action_heads else None
    policy = AlignedPolicy(model, stats, args.fork, shortest_euler=args.shortest_euler,
                           discrete_state=args.discrete_state,
                           reference_noise=bool(args.reference_start or init_metadata.get('reference_model_revision')))
    sample_episode = train_set[0]
    raw = sample_episode.row(min(34, len(sample_episode.state) - 1))
    info = dict(seed=args.seed, train_seeds=train_seeds, validation_seeds=validation_seeds,
                base_sha256=file_sha256(args.base),
                rejected_by_label_replay=rejected,
                train_frames=sum(len(ep.state) for ep in train_set), validation_frames=sum(len(ep.state) for ep in validation),
                horizon=HORIZON, execute=EXECUTE, actor_lora_rank=args.actor_lora_rank, vlm_lora_rank=args.vlm_lora_rank,
                shortest_euler=args.shortest_euler, discrete_state=args.discrete_state,
                input_format='LeRobot v2.1' if args.lerobot else 'HDF5 replay cache',
                openpi_revision=OPENPI_REVISION, lr=args.lr, action_loss=args.action_loss,
                gripper_loss_weight=args.gripper_loss_weight,
                gradient_accumulation=args.gradient_accumulation, image_augmentation=args.image_augmentation,
                micro_batch_size=args.micro_batch_size,
                instruction_balanced=args.instruction_balanced,
                train_action_heads=args.train_action_heads,
                train_full_action_expert=full_expert,
                render_samples=args.render_samples,
                training_script_sha256=file_sha256(__file__),
                runtime_script_sha256=file_sha256(Path(__file__).with_name('pi05_runtime.py')),
                norm_stats_sha256=hashlib.sha256(stats_text.encode()).hexdigest(),
                device=torch.cuda.get_device_name(), interface=check_inputs(policy, raw))
    if args.lerobot:
        info['dataset_verification_sha256'] = file_sha256(args.lerobot / 'verification.json')
        info['extra_dataset_verification_sha256'] = [file_sha256(root / 'verification.json')
                                                       for root in args.extra_lerobot]
    if args.diagnostic_case:
        info['training_scope'] = 'single_training_episode_imitation_not_generalization'
        info['diagnostic_case'] = args.diagnostic_case
        if args.state_input_diagnostic:
            info['reference_input_profile'] = 'measured_state_tokens_diagnostic_not_upstream_default'
        if args.action_norm_diagnostic or args.refit_action_quantiles:
            info['reference_action_profile'] = 'training_episode_chunk_quantiles_diagnostic_not_reference_norm'
        if full_expert:
            info['diagnostic_parameter_profile'] = 'full_action_expert_vlm_frozen'
        info['diagnostic_training_mae_before'] = validation_mae(policy, train_set)
    if args.smoke and args.image_augmentation:
        info['training_augmentation_image_max_delta'] = check_training_augmentation(policy, raw)
    if args.reference_start:
        info['reference_model_revision'] = MODEL_REVISION
    if initializer is not None:
        restore_trainable(model, initializer['policy'])
        info['initial_checkpoint_sha256'] = hashlib.sha256(initializer_path.read_bytes()).hexdigest()
        info['initial_checkpoint_optimizer_updates'] = (init_metadata.get('initial_checkpoint_optimizer_updates', 0)
                                                       + init_metadata['updates'])
        if init_metadata.get('base_sha256') != info['base_sha256']:
            raise ValueError('The initialization checkpoint used different base weights')
        if init_metadata.get('reference_model_revision'):
            info['reference_model_revision'] = init_metadata['reference_model_revision']
        if init_metadata.get('training_scope'):
            raise ValueError('Single-episode diagnostic weights cannot initialize a multitask result')
        del initializer
    if args.multitask:
        info['training_tasks'] = [task for task in POSE_TASKS if task in {ep.task for ep in train_set}]
        info['train_task_seeds'] = [(ep.task, ep.seed) for ep in train_set]
        info['validation_task_seeds'] = [(ep.task, ep.seed) for ep in validation]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.full_action_expert:
        if len(train_set) < 2:
            raise ValueError('Training-split full-expert mode needs multiple complete verified episodes')
        info['training_parameter_profile'] = 'full_action_expert_vlm_frozen'
    if args.refit_action_quantiles:
        info['reference_action_profile'] = 'train_split_shortest_euler_chunk_quantiles'
        info['action_quantile_fit_task_seeds'] = [(ep.task, ep.seed) for ep in train_set]
        info['action_quantile_fit_frames'] = sum(len(ep.state) for ep in train_set)
        info['state_normalization_source'] = 'unchanged_upstream_reference'
    info['trainable_parameter_count'] = sum(p.numel() for p in trainable)
    if full_expert:
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        if any(not (name.startswith(ACTOR_MODULES) or is_action_expert_parameter(name)) for name in names):
            raise RuntimeError('Full action diagnostic unexpectedly enabled a VLM parameter')
        expert_count = sum(p.numel() for name, p in model.named_parameters()
                           if p.requires_grad and is_action_expert_parameter(name))
        if not expert_count:
            raise RuntimeError('No action expert parameters were enabled')
        info['action_expert_parameter_count'] = expert_count
        (args.output / 'trainable-parameters.json').write_text(json.dumps(names, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info), flush=True)
    if args.fp32_master_optimizer:
        from fp32_master_adamw import FP32MasterAdamW
        optimizer = FP32MasterAdamW(trainable, lr=args.lr)
        info['optimizer_precision_profile'] = FP32MasterAdamW.FORMAT
        info['fp32_master_numel'] = sum(optimizer.optimizer_parameters[i].numel()
                                      for i in optimizer.master_indices)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    info['fp32_master_optimizer'] = args.fp32_master_optimizer
    initial_weights = trainable_snapshot(model) if args.compare_base else None
    info['optimizer_state_saved'] = True
    info['intermediate_snapshots_omit_optimizer'] = args.inference_only_snapshots
    first_update = 1
    if args.resume_run is not None:
        checkpoint = torch.load(args.resume_run / 'sft.pt', map_location='cpu', weights_only=True)
        source = checkpoint['metadata']
        for key in ('norm_stats_sha256', 'train_seeds', 'validation_seeds', 'seed', 'lr', 'shortest_euler',
                    'actor_lora_rank', 'vlm_lora_rank'):
            if source.get(key) != info[key]:
                raise ValueError(f'Resumed training differs on {key}')
        if source.get('action_loss', 'all') != args.action_loss:
            raise ValueError('Resuming requires the same flow loss objective')
        if (source.get('gripper_loss_weight', 1.0) != args.gripper_loss_weight
                or source.get('diagnostic_case') != args.diagnostic_case):
            raise ValueError('Resuming requires the same gripper weighting and diagnostic scope')
        if source.get('discrete_state', False) != args.discrete_state:
            raise ValueError('Resuming requires matching state conditioning')
        if (source.get('gradient_accumulation', 1) != args.gradient_accumulation
                or source.get('micro_batch_size', 1) != args.micro_batch_size
                or source.get('image_augmentation', False) != args.image_augmentation):
            raise ValueError('Resuming requires the same minibatch and image augmentation settings')
        if (source.get('train_action_heads', True) != args.train_action_heads
                or source.get('train_full_action_expert', False) != full_expert):
            raise ValueError('Resuming requires the same trainable parameter groups')
        if source.get('instruction_balanced', False) != args.instruction_balanced:
            raise ValueError('Resuming requires the same instruction sampler')
        for key in ('base_sha256', 'render_samples', 'dataset_verification_sha256',
                    'train_task_seeds', 'validation_task_seeds', 'reference_model_revision'):
            if source.get(key) != info.get(key):
                raise ValueError(f'Resumed training differs on {key}')
        if (source.get('extra_dataset_verification_sha256', [])
                != info.get('extra_dataset_verification_sha256', [])
                or source.get('recovery_share', 0.0) != args.recovery_share):
            raise ValueError('Resuming requires the same extra datasets and recovery sampling')
        if source.get('fp32_master_optimizer', False) != args.fp32_master_optimizer:
            raise ValueError('Resume with the original optimizer precision profile')
        first_update = source['updates'] + 1
        if first_update > args.updates or args.compare_base:
            raise ValueError('Choose a later total update count for resumed training')
        info['resumed_from_update'] = source['updates']
        info['prior_training_seconds'] = source.get('prior_training_seconds', 0) + source['training_seconds']
        restore_trainable(model, checkpoint['policy'])
        if source.get('optimizer_state_saved', True) is False:
            raise ValueError('An inference-only snapshot cannot resume training')
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        rng.bit_generator.state = checkpoint['sampler_rng']
        if validation:
            info['validation_prior_sft'] = validation_mae(policy, validation)
        del checkpoint
    elif validation:
        key = 'validation_initial_sft' if args.init_run else 'validation_base'
        info[key] = validation_mae(policy, validation)
    pools = training_pools(train_set, multitask=args.multitask, recovery_share=args.recovery_share)
    sampling_tasks = tuple(pools)
    if args.recovery_share:
        info['correction_task_seeds'] = [(ep.task, ep.seed) for ep in train_set if ep.policy_prefix_steps]
        info['recovery_share'] = args.recovery_share
        info['recovery_share_scope'] = 'within tasks that have verified corrections; other tasks use expert demos'
    info['training_source_counts'] = {task: {source: len(items) for source, items in group.items()}
                                      for task, group in pools.items()}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with (args.output / 'updates.jsonl').open('x', encoding='utf-8') as stream:
        for step in range(first_update, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            sampled = []
            mean_loss = 0.0
            for micro_step in range(args.gradient_accumulation):
                rows = []
                for slot in range(args.micro_batch_size):
                    sample_index = ((step - 1) * args.gradient_accumulation + micro_step) * args.micro_batch_size + slot
                    task = sampling_tasks[sample_index % len(sampling_tasks)]
                    group = pools[task]
                    source = ('expert_correction' if args.recovery_share and group['expert_correction']
                              and rng.random() < args.recovery_share else 'expert_demo')
                    candidates = (group[source] if args.recovery_share else
                                  group['expert_demo'] + group['expert_correction'])
                    if args.instruction_balanced:
                        prompts = sorted({episode.prompt for episode in candidates})
                        prompt = prompts[int(rng.integers(len(prompts)))]
                        candidates = [episode for episode in candidates if episode.prompt == prompt]
                    ep = candidates[int(rng.integers(len(candidates)))]
                    frame = int(rng.integers(len(ep.state)))
                    rows.append(ep.row(frame))
                    sampled.append(dict(task=ep.task, expert_seed=ep.seed, frame=frame,
                                        source='expert_correction' if ep.policy_prefix_steps else 'expert_demo'))
                obs, labels = policy.encode_training_batch(rows)
                loss = flow_loss(model, obs, labels, args.action_loss, image_augmentation=args.image_augmentation,
                                 gripper_loss_weight=args.gripper_loss_weight)
                (loss / args.gradient_accumulation).backward()
                mean_loss += float(loss.detach()) / args.gradient_accumulation
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            before = trainable_snapshot(model) if step == first_update else None
            if step == first_update:
                if full_expert:
                    info['first_action_expert_grad_l2_after_clip'] = float(torch.stack([
                        p.grad.detach().float().square().sum() for name, p in model.named_parameters()
                        if is_action_expert_parameter(name) and p.grad is not None
                    ]).sum().sqrt())
                info['first_vlm_grad_l2_after_clip'] = float(torch.stack([
                    p.grad.detach().float().square().sum() for name, p in model.named_parameters()
                    if vlm_adapter_parameter(name) and p.grad is not None
                ]).sum().sqrt()) if args.vlm_lora_rank else 0.0
            optimizer.step()
            row = dict(update=step, expert_seed=ep.seed, frame=frame, loss=mean_loss, grad_l2=float(grad))
            if args.gradient_accumulation * args.micro_batch_size > 1:
                row['samples'] = sampled
            if args.multitask:
                row['task'] = ep.task
            if args.recovery_share:
                row['source'] = 'expert_correction' if ep.policy_prefix_steps else 'expert_demo'
            if step == first_update:
                info['first_actor_max_delta'] = actor_delta(model, before)
                if full_expert:
                    info['first_action_expert_max_delta'] = max(
                        float((p.detach().cpu() - before[name]).abs().max())
                        for name, p in model.named_parameters() if is_action_expert_parameter(name))
                    if (info['first_action_expert_grad_l2_after_clip'] <= 0
                            or info['first_action_expert_max_delta'] <= 0):
                        raise RuntimeError('The full action expert did not receive an effective update')
                    row['action_expert_max_delta'] = info['first_action_expert_max_delta']
                    row['action_expert_grad_l2_after_clip'] = info['first_action_expert_grad_l2_after_clip']
                info['first_vlm_max_delta'] = max((float((p.detach().cpu() - before[name]).abs().max())
                                                 for name, p in model.named_parameters()
                                                 if vlm_adapter_parameter(name)), default=0.0)
                if info['first_actor_max_delta'] <= 0 or grad <= 0:
                    raise RuntimeError('The actor did not update')
                if args.vlm_lora_rank and (info['first_vlm_grad_l2_after_clip'] <= 0
                                          or info['first_vlm_max_delta'] <= 0):
                    raise RuntimeError('The selected VLM adapters did not update')
                row['actor_max_delta'] = info['first_actor_max_delta']
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if step in (first_update, args.updates) or step % 50 == 0:
                print(json.dumps(row), flush=True)
            if args.save_every and step % args.save_every == 0 and step < args.updates:
                snapshot_dir = args.output / f'checkpoint-{step:04d}'
                snapshot_info = dict(info, updates=step, sampled_windows=step * args.gradient_accumulation * args.micro_batch_size,
                                     training_seconds=time.monotonic() - start,
                                     peak_cuda_bytes=torch.cuda.max_memory_allocated())
                (snapshot_dir).mkdir(exist_ok=False)
                (snapshot_dir / 'norm_stats.json').write_text(stats_text, encoding='utf-8')
                if args.inference_only_snapshots:
                    snapshot_info['optimizer_state_saved'] = False
                    torch.save(dict(policy=trainable_snapshot(model), metadata=snapshot_info,
                                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                    sampler_rng=rng.bit_generator.state), snapshot_dir / 'sft.pt')
                else:
                    save_training_checkpoint(snapshot_dir, model, optimizer, snapshot_info, rng)
                (snapshot_dir / 'summary.json').write_text(json.dumps(snapshot_info, indent=2) + '\n', encoding='utf-8')
    info.update(updates=args.updates, training_seconds=time.monotonic() - start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                sampled_windows=args.updates * args.gradient_accumulation * args.micro_batch_size)
    if frozen_heads is not None:
        parameters = dict(model.named_parameters())
        if any(not torch.equal(value, parameters[name].detach().cpu()) for name, value in frozen_heads.items()):
            raise RuntimeError('A frozen action projection or time MLP changed')
        info['frozen_action_heads_unchanged'] = True
    raw.pop('actions')
    policy.generator = torch.Generator(device='cuda').manual_seed(13)
    before = policy.infer(raw)['actions']
    checkpoint = save_training_checkpoint(args.output, model, optimizer, info, rng)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for param in trainable:
            param.zero_()
    restore_trainable(model, restored['policy'])
    optimizer.load_state_dict(restored['optimizer'])
    if not optimizer.state:
        raise RuntimeError('Checkpoint did not restore optimizer state')
    policy.generator = torch.Generator(device='cuda').manual_seed(13)
    after = policy.infer(raw)['actions']
    np.testing.assert_array_equal(after, before)
    info['reload_max_action_difference'] = float(np.max(np.abs(after - before)))
    if validation:
        info['validation_sft'] = validation_mae(policy, validation)
    if args.diagnostic_case:
        info['diagnostic_training_mae_after'] = validation_mae(policy, train_set)
    (args.output / 'summary.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info), flush=True)
    if args.diagnostic_rollout:
        episode, = train_set
        with open_episode(episode.seed, episode.task, render_samples=args.render_samples) as env:
            reset, replay = build_episode(env, None, episode.seed, 1, False, task=episode.task)
            replay.close()
        if (reset['initial_state_sha256'] != episode.initial_state
                or reset['instruction'] != episode.prompt):
            raise ValueError('The diagnostic rollout reset differs from the selected training episode')
        (args.output / 'diagnostic-reset.jsonl').write_text(json.dumps(reset) + '\n', encoding='utf-8')
        policy.provenance = dict(
            sft_sha256=file_sha256(checkpoint), base_sha256=info['base_sha256'],
            norm_stats_sha256=info['norm_stats_sha256'],
            reference_model_revision=info.get('reference_model_revision'),
            training_scope=info['training_scope'], diagnostic_case=info['diagnostic_case'])
        evaluate(policy, args.output / 'training-reset', {episode.seed: reset}, args.max_steps,
                 args.video, 'sft', seeds=[episode.seed], trace=True, task=episode.task,
                 render_samples=args.render_samples)
    if not args.smoke and not args.skip_evaluation:
        evaluate(policy, args.output, expected, args.max_steps, args.video, 'sft',
                 seeds=args.train_eval_seed or DEVELOPMENT_SEEDS, render_samples=args.render_samples)
        if initial_weights is not None:
            restore_trainable(model, initial_weights)
            evaluate(policy, args.output / 'base', expected, args.max_steps, False, 'base',
                     seeds=args.train_eval_seed or DEVELOPMENT_SEEDS, render_samples=args.render_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'paired-episodes', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--cache', type=Path)
    data.add_argument('--lerobot', type=Path)
    parser.add_argument('--full-action-expert', action='store_true',
                        help='Train the full action expert on a verified multi-episode LeRobot training split')
    parser.add_argument('--refit-action-quantiles', action='store_true',
                        help='Fit shortest-Euler action quantiles only on the selected training split; preserve state statistics')
    parser.add_argument('--diagnostic-case', metavar='TASK:SEED',
                        help='Fit one existing training episode; this is not a generalization experiment')
    parser.add_argument('--diagnostic-rollout', action='store_true',
                        help='Run the fitted policy on its one declared training reset after saving')
    parser.add_argument('--gripper-loss-weight', type=float, default=1.0,
                        help='Relative weight of the gripper coordinate in the flow matching loss')
    parser.add_argument('--discrete-state', action='store_true', help='Tokenize measured robot state with the prompt')
    parser.add_argument('--state-input-diagnostic', action='store_true',
                        help='Explicitly label a single-episode reference-input adaptation; disallowed for multitask training')
    parser.add_argument('--action-norm-diagnostic', action='store_true',
                        help='Refit only action quantiles on the declared training episode; diagnostic-only output profile')
    parser.add_argument('--full-action-expert-diagnostic', action='store_true',
                        help='Unfreeze the action expert only for a declared single-episode diagnostic')
    parser.add_argument('--fp32-master-optimizer', action='store_true',
                        help='Accumulate full-action-expert updates in FP32 without changing inference dtypes')
    parser.add_argument('--load-run', type=Path)
    parser.add_argument('--resume-run', type=Path, help='Restore saved policy, optimizer and random streams')
    parser.add_argument('--reference-start', type=Path, help='Start with the published VLABench weights and stats')
    parser.add_argument('--extra-lerobot', type=Path, action='append', default=[],
                        help='Append another verified training split without copying it')
    parser.add_argument('--init-run', type=Path, help='Start from trained parameters and keep their normalization')
    parser.add_argument('--recovery-share', type=float, default=0.0,
                        help='Within each task, sample corrections at this share when available')
    parser.add_argument('--eval-stage', choices=('base', 'sft', 'ppo'), default='sft')
    parser.add_argument('--ppo-weights', type=Path)
    parser.add_argument('--eval-execute', type=int, choices=range(1, HORIZON + 1), default=EXECUTE)
    parser.add_argument('--train-eval-seed', action='append', type=int, choices=DEVELOPMENT_SEEDS)
    parser.add_argument('--actor-lora-rank', type=int, default=ADAPTER_RANK)
    parser.add_argument('--vlm-lora-rank', type=int, default=VLM_RANK)
    parser.add_argument('--freeze-action-heads', dest='train_action_heads', action='store_false', default=True,
                        help='Update LoRA adapters while keeping pretrained action projections and time MLPs fixed')
    parser.add_argument('--action-loss', choices=('all', 'executed-poses'), default='all',
                        help='Optionally fit only the seven coordinates in the first five controls')
    parser.add_argument('--eval-seed', action='append', type=int)
    parser.add_argument('--eval-task', choices=POSE_TASKS, default=TASK)
    parser.add_argument('--prompt-mode', choices=('official', 'paraphrase', 'alternate_target'), default='official')
    parser.add_argument('--multitask', action='store_true', help='Cycle across selected pose tasks; all three by default')
    parser.add_argument('--instruction-balanced', action='store_true',
                        help='Sample actual instructions equally within each task and source pool')
    parser.add_argument('--training-tasks', choices=POSE_TASKS, nargs='+',
                        help='Train only these tasks with --multitask; default retains all three tasks')
    parser.add_argument('--resume-evaluation', action='store_true', help='Resume a matched pose evaluation')
    parser.add_argument('--trace-controls', action='store_true', help='Write per-control diagnostics for --load-run')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--compare-base', action='store_true')
    parser.add_argument('--label-audit', type=Path)
    parser.add_argument('--shortest-euler', action='store_true',
                        help='Learn equivalent Euler deltas in [-pi, pi)')
    parser.add_argument('--seed', type=int, default=263)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--save-every', type=int, default=0, help='Keep intermediate checkpoints at this update interval')
    parser.add_argument('--inference-only-snapshots', action='store_true',
                        help='Intermediate snapshots omit optimizer state; final checkpoint remains resumable')
    parser.add_argument('--skip-evaluation', action='store_true', help='Save the trained checkpoint for separate task evaluations')
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8))
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--micro-batch-size', type=int, default=1,
                        help='Encode this many complete windows in each real forward/backward pass')
    parser.add_argument('--no-image-augmentation', dest='image_augmentation', action='store_false', default=True)
    args = parser.parse_args()
    import openpi

    if args.fork.resolve() not in Path(openpi.__file__).resolve().parents:
        parser.error('Prepend the pinned VLABench OpenPI fork to PYTHONPATH')
    if not 1 <= args.max_steps <= 600:
        parser.error('Expected an evaluation budget in [1, 600] controls')
    if (args.eval_stage == 'ppo') != bool(args.ppo_weights) or (args.ppo_weights and not args.load_run):
        parser.error('Supply both --load-run and --ppo-weights for PPO evaluation')
    if args.resume_evaluation and (args.load_run is None or args.trace_controls or args.video):
        parser.error('--resume-evaluation requires --load-run without trace or video recording')
    if args.load_run is not None:
        evaluate_checkpoint(args)
        return
    if (args.eval_seed or args.trace_controls or args.eval_execute != EXECUTE or args.eval_task != TASK
            or args.prompt_mode != 'official'):
        parser.error('Diagnostic evaluation arguments require --load-run')
    if (args.cache is None and args.lerobot is None) or args.updates < 1 or args.lr <= 0:
        parser.error('Training requires a cache, positive update count, and positive learning rate')
    if args.actor_lora_rank < 0 or args.vlm_lora_rank < 0:
        parser.error('LoRA ranks must be nonnegative')
    if not np.isfinite(args.gripper_loss_weight) or args.gripper_loss_weight <= 0:
        parser.error('The gripper loss weight must be finite and positive')
    if args.diagnostic_case:
        try:
            diagnostic_task, diagnostic_seed = args.diagnostic_case.rsplit(':', 1)
            int(diagnostic_seed)
        except ValueError:
            parser.error('Use --diagnostic-case TASK:SEED')
        if (diagnostic_task not in POSE_TASKS or args.lerobot is None or args.multitask
                or args.smoke or args.extra_lerobot or args.recovery_share or not args.skip_evaluation):
            parser.error('A diagnostic requires one LeRobot training episode and --skip-evaluation')
    elif args.diagnostic_rollout:
        parser.error('--diagnostic-rollout requires --diagnostic-case')
    if not args.train_action_heads and not (args.actor_lora_rank or args.vlm_lora_rank):
        parser.error('Frozen action heads require at least one LoRA adapter')
    if args.gradient_accumulation < 1 or args.micro_batch_size < 1 or args.save_every < 0:
        parser.error('Gradient accumulation and batch size must be positive; save interval must be nonnegative')
    if args.fp32_master_optimizer and not (args.full_action_expert_diagnostic or args.full_action_expert):
        parser.error('FP32 master updates currently require the explicit full-action-expert diagnostic')
    if args.full_action_expert and (not args.reference_start or args.diagnostic_case
            or args.full_action_expert_diagnostic or args.actor_lora_rank or args.vlm_lora_rank
            or args.lerobot is None or not args.multitask):
        parser.error('Full-expert training-split mode requires published base weights, verified LeRobot episodes and no adapters or diagnostic case')
    if args.refit_action_quantiles and not (args.full_action_expert and args.reference_start and args.shortest_euler):
        parser.error('Train-split quantiles require explicit full-expert mode and shortest Euler labels')
    if args.state_input_diagnostic and not (args.diagnostic_case and args.discrete_state and args.reference_start):
        parser.error('The state-input diagnostic requires one diagnostic episode and explicit state tokenization')
    if args.action_norm_diagnostic and not (args.diagnostic_case and args.reference_start):
        parser.error('The action-normalization diagnostic requires one diagnostic episode and published parent weights')
    if args.full_action_expert_diagnostic and not (args.diagnostic_case and args.reference_start
                                                 and not args.actor_lora_rank and not args.vlm_lora_rank):
        parser.error('The full-action-expert diagnostic requires one episode and no LoRA adapters')
    if args.reference_start and ((args.discrete_state and not args.state_input_diagnostic) or args.init_run
            or args.base.resolve() != (args.reference_start / 'model.safetensors').resolve()):
        parser.error('Use the published weights and their unchanged state conditioning and normalization')
    if args.extra_lerobot and args.lerobot is None:
        parser.error('--extra-lerobot requires a LeRobot export')
    if args.init_run and (args.resume_run or args.compare_base):
        parser.error('--init-run starts a new training phase with its own optimizer')
    if not 0 <= args.recovery_share < 1 or (args.recovery_share and args.lerobot is None):
        parser.error('Recovery sampling requires LeRobot data and a share in [0, 1)')
    if args.multitask and args.lerobot is None:
        parser.error('Multitask training needs a LeRobot export')
    if args.training_tasks and (not args.multitask or len(args.training_tasks) < 1
                                or len(args.training_tasks) != len(set(args.training_tasks))):
        parser.error('--training-tasks requires --multitask and at least one distinct task')
    if args.smoke:
        args.updates = len(args.training_tasks or POSE_TASKS) if args.multitask else 1
    elif args.lerobot is None and args.label_audit is None:
        parser.error('--label-audit is required for the full pilot')
    train(args)


if __name__ == '__main__':
    main()
