"""Train fruit PPO with versioned rewards, critic warmup and update diagnostics.

The registered November 22 entry points remain unchanged for reproducibility.
Simulator and VLA imports are deferred so the reward and optimizer can be tested
on CPU without loading model weights or downloading assets.
"""

import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch


GAMMA, GAE_LAMBDA, PPO_CLIP = 0.99, 0.95, 0.2
LEGACY_REWARD = 'legacy-distance-v1'
CORRECTED_REWARD = 'discounted-terminal-potential-v2'
REWARD_VERSIONS = (LEGACY_REWARD, CORRECTED_REWARD)


def fruit_reward(before, after, solved, terminal, weight, version, gamma=GAMMA):
    """Use reward-only geometry; the finite task budget is a terminal state."""
    if (version not in REWARD_VERSIONS or not 0 <= weight <= 1 or not 0 < gamma <= 1
            or not all(math.isfinite(x) for x in (before, after))):
        raise ValueError('Invalid reward version, coefficient, discount or potential')
    terminal = bool(terminal or solved)
    effective_after = 0.0 if version == CORRECTED_REWARD and terminal else after
    discount = gamma if version == CORRECTED_REWARD else 1.0
    progress = weight * (discount * effective_after - before)
    return dict(reward=float(solved) + progress, official_reward=float(solved),
                progress_reward=progress, potential_before=before, potential_after=after,
                effective_potential_after=effective_after, terminal=terminal,
                reward_version=version)


def finish_returns(episode, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    """Compute per-chunk GAE and independent Monte Carlo critic-warmup targets."""
    if not episode or not episode[-1]['terminal'] or any(t['terminal'] for t in episode[:-1]):
        raise ValueError('Expected exactly one complete finite-horizon episode')
    next_value = advantage = mc_return = 0.0
    for transition in reversed(episode):
        mc_return = transition['reward'] + gamma * mc_return
        delta = transition['reward'] + gamma * next_value - transition['value']
        advantage = delta + gamma * gae_lambda * advantage
        transition.update(advantage=advantage, **{'return': advantage + transition['value']},
                          mc_return=mc_return)
        next_value = transition['value']


def parameter_groups(model):
    actor, critic = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (critic if name.startswith('value_head.') else actor).append(parameter)
    if not actor or not critic:
        raise ValueError('Expected trainable VLA actor parameters and a separate value head')
    return actor, critic


@contextmanager
def freeze_actor(actor, enabled):
    try:
        if enabled:
            for parameter in actor:
                parameter.grad = None
                parameter.requires_grad_(False)
        yield
    finally:
        if enabled:
            for parameter in actor:
                parameter.requires_grad_(True)


def ppo_terms(logp, old_logp, value, target, advantage):
    """Conditional selected-flow-step likelihood, summed over executed coordinates."""
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantage)
    return dict(actor_loss=-surrogate, value_loss=(value - target).square(),
                approx_kl=(ratio - 1) - log_ratio,
                clip_fraction=(torch.abs(ratio - 1) > PPO_CLIP).float(),
                ratio=ratio, log_ratio=log_ratio)


def optimize(model, optimizer, transitions, replay, rng, *, minibatch=8, epochs=1,
             critic_only=False, target_kl=0.02, record=None):
    """Update only the critic during warmup; retain actual actor-update evidence."""
    if not transitions or minibatch < 1 or epochs < 1 or target_kl <= 0:
        raise ValueError('Invalid optimizer data, batch size, epochs or KL budget')
    if not critic_only and not any(t['reward'] != 0 for t in transitions):
        def no_reward_record(entry):
            entry.update(phase='critic_only_no_reward', actor_update_skipped_reason='all_chunk_rewards_zero')
            if record is not None:
                record(entry)

        return optimize(model, optimizer, transitions, replay, rng, minibatch=minibatch, epochs=epochs,
                        critic_only=True, target_kl=target_kl, record=no_reward_record)
    actor, critic = parameter_groups(model)
    advantages = np.asarray([t['advantage'] for t in transitions], dtype=np.float64)
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)
    history = []
    with freeze_actor(actor, critic_only):
        for epoch in range(epochs):
            indices = rng.permutation(len(transitions))
            for offset in range(0, len(indices), minibatch):
                batch = indices[offset:offset + minibatch]
                optimizer.zero_grad(set_to_none=True)
                totals = dict(actor_loss=0., value_loss=0., approx_kl=0., clip_fraction=0.,
                              ratio=0., conditional_logprob_per_coordinate=0.)
                log_ratio_min, log_ratio_max = float('inf'), -float('inf')
                for index in batch:
                    transition = transitions[index]
                    logp, value = replay(transition)
                    target = transition['mc_return'] if critic_only else transition['return']
                    terms = ppo_terms(logp, transition['old_logp'], value, target, float(advantages[index]))
                    if not all(bool(torch.isfinite(x).all()) for x in terms.values()):
                        raise FloatingPointError('Nonfinite PPO loss or ratio; no optimizer step applied')
                    loss = 0.5 * terms['value_loss']
                    if not critic_only:
                        loss = loss + terms['actor_loss']
                    (loss / len(batch)).backward()
                    for key in totals:
                        item = (logp / (transition['executed'] * transition['action_dim'])
                                if key == 'conditional_logprob_per_coordinate' else terms[key])
                        totals[key] += float(item.detach()) / len(batch)
                    log_ratio_min = min(log_ratio_min, float(terms['log_ratio'].detach()))
                    log_ratio_max = max(log_ratio_max, float(terms['log_ratio'].detach()))
                pre_stopped = not critic_only and totals['approx_kl'] > target_kl
                skipped = pre_stopped
                if critic_only and any(p.grad is not None for p in actor):
                    raise RuntimeError('Critic warmup produced an actor gradient')
                actor_grad = (0.0 if critic_only else float(torch.nn.utils.clip_grad_norm_(
                    actor, 1.0, error_if_nonfinite=True)))
                critic_grad = float(torch.nn.utils.clip_grad_norm_(critic, 1.0, error_if_nonfinite=True))
                before = [] if critic_only or skipped else [p.detach().clone() for p in actor]
                critic_before = [] if critic_only or skipped else [p.detach().clone() for p in critic]
                optimizer_before = None if critic_only or skipped else copy.deepcopy(optimizer.state_dict())
                post_kl, reverted, post_error = None, False, None
                if not skipped:
                    optimizer.step()
                    if not critic_only:
                        try:
                            post_kl = prefix_kl([transitions[i] for i in batch], replay)
                        except Exception as error:
                            post_kl, post_error = float('inf'), error
                        reverted = not math.isfinite(post_kl) or post_kl > target_kl
                        if reverted:
                            with torch.no_grad():
                                for parameter, old in zip(actor + critic, before + critic_before):
                                    parameter.copy_(old)
                            optimizer.load_state_dict(optimizer_before)
                            skipped = True
                actor_delta = max((float((p.detach() - old).abs().max())
                                   for p, old in zip(actor, before)), default=0.0)
                entry = dict(epoch=epoch + 1, minibatch=len(history) + 1, transitions=len(batch),
                             phase='critic_warmup' if critic_only else 'ppo', **totals,
                             log_ratio_min=log_ratio_min, log_ratio_max=log_ratio_max,
                             actor_grad_l2=actor_grad, critic_grad_l2=critic_grad,
                             actor_max_abs_delta_this_step=actor_delta,
                             optimizer_step_applied=not skipped, stopped_for_kl=skipped,
                             prestep_stopped_for_kl=pre_stopped, poststep_reverted_for_kl=reverted,
                             postupdate_approx_kl=post_kl if post_kl is None or math.isfinite(post_kl) else None,
                             postupdate_diagnostic_error=str(post_error) if post_error is not None else None,
                             target_kl=target_kl,
                             executed_lengths={str(length): sum(transitions[i]['executed'] == length for i in batch)
                                               for length in sorted({transitions[i]['executed'] for i in batch})})
                history.append(entry)
                if record is not None:
                    record(entry)
                optimizer.zero_grad(set_to_none=True)
                if post_error is not None:
                    raise RuntimeError('Post-step likelihood check failed; parameters and optimizer '
                                       'were rolled back') from post_error
                if skipped:
                    return history
    return history


def prefix_kl(transitions, replay):
    """Sampled executed-prefix score statistic, not an analytic action-space KL."""
    values = []
    with torch.no_grad():
        for transition in transitions:
            logp, _ = replay(transition)
            log_ratio = float(logp) - transition['old_logp']
            try:
                values.append(math.expm1(log_ratio) - log_ratio)
            except OverflowError:
                return float('inf')
    return float(np.mean(values))


def check_replay(transitions, scores_for, record=None):
    errors, coordinate_errors = [], []
    with torch.no_grad():
        for transition in transitions:
            scores = scores_for(transition)
            old = transition['old_logprobs'].to(scores.device)
            if (old.shape != scores.shape or not bool(torch.isfinite(old).all())
                    or not bool(torch.isfinite(scores).all())):
                raise ValueError('Invalid sampled/replayed likelihood coordinates')
            errors.append(abs(float(scores.sum()) - transition['old_logp']))
            coordinate_errors.append(float((scores - old).abs().max()))
    error, coordinate_error = max(errors), max(coordinate_errors)
    passed = math.isfinite(error) and error <= 0.05 and coordinate_error <= 0.005
    report = dict(chunks=len(transitions), passed=passed, aggregate_max_abs_error=error,
                  coordinate_max_abs_error=coordinate_error,
                  worst_coordinate_chunk=int(np.argmax(coordinate_errors)))
    if record is not None:
        record(report)
    if not passed:
        raise RuntimeError(f'Replay mismatch: aggregate={error:g}, coordinate={coordinate_error:g}')
    return report


def value_metrics(transitions):
    target = np.asarray([t['mc_return'] for t in transitions], dtype=np.float64)
    prediction = np.asarray([t['value'] for t in transitions], dtype=np.float64)
    variance = float(target.var())
    return dict(mc_value_mse=float(np.mean((prediction - target) ** 2)),
                mc_explained_variance=(1 - float((prediction - target).var()) / variance
                                       if variance > 1e-12 else None),
                mean_value=float(prediction.mean()), mean_mc_return=float(target.mean()))


def collect_rollout(policy, adapter, seed, generator, controls, shaping, reward_version, render_samples,
                    *, control_trace=None, chunk_record=None):
    from evaluate_vlabench_reference import TASK, build_episode
    from ppo_pi05 import fruit_potential
    from ppo_reference_aligned_fruit import PPOClient
    from select_fruit import open_episode
    from train_reference_aligned_fruit import DIM, EXECUTE

    class RewardClient(PPOClient):
        def infer(self, row):
            decoded = super().infer(row)
            if chunk_record is not None:
                transition, _ = self.pending
                chunk_record(dict(chunk=len(self.transitions),
                                  measured_input_state=np.asarray(row['observation/state']).tolist(),
                                  raw_policy_actions=transition['chain'][0, -1, :, :DIM].float().tolist(),
                                  decoded_pose_actions=np.asarray(decoded['actions']).tolist()))
            return decoded

        def finish_chunk(self, executed, solved, *, terminal=False):
            if self.pending is None or not 1 <= executed <= EXECUTE:
                raise RuntimeError('Expected a sampled chunk and its actual executed length')
            transition, logprobs = self.pending
            potential = fruit_potential(self.env) if self.shaping else 0.0
            transition.update(fruit_reward(self.potential, potential, solved, terminal,
                                           self.shaping, reward_version))
            transition.update(executed=executed, action_dim=DIM,
                              old_logp=float(logprobs[:executed].sum()),
                              old_logprobs=logprobs[:executed].clone(),
                              denoise_index=int(transition['denoise_inds'].reshape(-1)[0]))
            self.transitions.append(transition)
            self.potential, self.pending = potential, None

    random_state = random.getstate()
    try:
        env = open_episode(seed, TASK, render_samples=render_samples)
    finally:
        random.setstate(random_state)
    with env:
        client = RewardClient(policy, env, generator, shaping)
        result, rollout = build_episode(env, adapter(client, replan_steps=EXECUTE), seed, controls, False,
                                        trace=control_trace)
        for _ in rollout:
            pass
        client.finish_chunk(result['steps'] - len(client.transitions) * EXECUTE,
                            result['success'], terminal=True)
    finish_returns(client.transitions)
    return client.transitions, result


def replay_scores(model, encoding, transition):
    observation, _ = encoding.encode(transition['row'])
    device = observation.state.device
    inputs = dict(chains=transition['chain'].to(device),
                  denoise_inds=transition['denoise_inds'].to(device),
                  tokenized_prompt=observation.tokenized_prompt,
                  tokenized_prompt_mask=observation.tokenized_prompt_mask, obs_state=observation.state)
    inputs.update({f'obs_image__{key}': value for key, value in observation.images.items()})
    inputs.update({f'obs_image_mask__{key}': value for key, value in observation.image_masks.items()})
    output = model.default_forward(inputs)
    scores, value = output['logprobs'][0, :transition['executed']], output['values'].reshape(())
    if (scores.shape != (transition['executed'], transition['action_dim'])
            or not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(value))):
        raise ValueError('Invalid replay score coordinates or critic value')
    return scores, value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'sft-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--reward-version', choices=REWARD_VERSIONS, required=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--critic-warmup-epochs', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--shaping', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=372)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--target-kl', type=float, default=0.02)
    parser.add_argument('--trace-controls', action='store_true',
                        help='Save raw/decoded chunks and official IK/measured-pose diagnostics')
    args = parser.parse_args()
    if (not 1 <= args.max_steps <= 600 or args.iterations < 1 or args.critic_warmup_epochs < 0
            or args.minibatch < 1 or not 0 <= args.shaping <= 1 or not 0 < args.lr <= 1
            or not 0 < args.critic_lr <= 1 or not 0 < args.target_kl <= 1
            or len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds))
            or any(not 1301 <= seed <= 1399 for seed in args.seeds)):
        parser.error('Use distinct training starts 1301–1399 and valid optimizer/rollout budgets')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires CUDA; CPU tests do not require a GPU')
    from openpi.shared import normalize
    from check_reference_rl_handoff import RLPolicy
    from evaluate_vlabench_reference import load_upstream_adapter
    from pi05_runtime import build_model, file_sha256, restore_trainable, trainable_snapshot
    from ppo_reference_aligned_fruit import ActorEncoding, actor_change_evidence
    from train_reference_aligned_fruit import DIM, EXECUTE, HORIZON

    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    existing_parent = next(path for path in args.output.parents if path.exists())
    if shutil.disk_usage(existing_parent).free < 2 * 1024 ** 3:
        raise RuntimeError('Less than 2 GiB free; review storage before starting training')
    checkpoint_path = args.sft_run / 'sft.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    stats_bytes = (args.sft_run / 'norm_stats.json').read_bytes()
    stats_hash = hashlib.sha256(stats_bytes).hexdigest()
    base_hash = file_sha256(args.base)
    if (metadata.get('base_sha256') != base_hash or metadata['norm_stats_sha256'] != stats_hash
            or metadata['horizon'] != HORIZON or metadata['execute'] != EXECUTE):
        raise ValueError('SFT parent weights, normalization or policy interface differ')
    if metadata['vlm_lora_rank'] != 0:
        raise ValueError('This version requires frozen VLM features to isolate critic gradients')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    model, _ = build_model(args.base, 'cuda', EXECUTE, rl=True, action_env_dim=DIM,
                           action_horizon=HORIZON, lora_rank=metadata['actor_lora_rank'],
                           vlm_lora_rank=metadata['vlm_lora_rank'],
                           discrete_state_input=metadata['discrete_state'],
                           train_action_heads=metadata.get('train_action_heads', True))
    restore_trainable(model, checkpoint['policy'])
    original = trainable_snapshot(model, actor_only=True)
    actor, critic = parameter_groups(model)
    if {id(p) for p in actor} != {id(p) for name, p in model.named_parameters() if name in original}:
        raise ValueError('Unexpected trainable non-critic parameters')
    policy = RLPolicy(model, normalize.deserialize_json(stats_bytes.decode()), args.fork,
                      shortest_euler=metadata['shortest_euler'], discrete_state=metadata['discrete_state'])
    encoding = ActorEncoding(policy)
    adapter = load_upstream_adapter(args.fork)
    optimizer = torch.optim.AdamW([dict(params=actor, lr=args.lr),
                                   dict(params=critic, lr=args.critic_lr)], weight_decay=0)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                    sft_sha256=file_sha256(checkpoint_path), norm_stats_sha256=stats_hash,
                    base_sha256=base_hash, training_script_sha256=file_sha256(__file__),
                    shared_ppo_script_sha256=file_sha256(Path(__file__).with_name('ppo_pi05.py')),
                    legacy_entry_sha256=file_sha256(Path(__file__).with_name('ppo_reference_aligned_fruit.py')),
                    gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip=PPO_CLIP, horizon=HORIZON,
                    executed=EXECUTE, action_dim=DIM, finite_horizon=True,
                    likelihood='selected flow transition; sum over executed steps and action dimensions',
                    kl_diagnostic='sampled executed-prefix score estimate; not full-chain or decoded-action KL',
                    maximum_controls=(args.iterations + int(args.critic_warmup_epochs > 0))
                                     * len(args.seeds) * args.max_steps)
    (args.output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    total_controls = total_steps = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def append(name, row):
        with (args.output / name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def replay(transition):
        scores, value = replay_scores(model, encoding, transition)
        return scores.sum(), value

    def collect(phase, iteration):
        nonlocal total_controls
        samples, successes = [], 0
        for seed in args.seeds:
            with ExitStack() as stack:
                control_trace, chunk_record = None, None
                if args.trace_controls:
                    stem = f'traces/{phase}-{iteration}-{seed}'
                    (args.output / 'traces').mkdir(exist_ok=True)
                    control_trace = stack.enter_context((args.output / f'{stem}-controls.jsonl').open('x'))
                    chunk_record = lambda row: append(f'{stem}-actions.jsonl', row)
                episode, result = collect_rollout(policy, adapter, seed, generator, args.max_steps,
                                                  args.shaping, args.reward_version, metadata.get('render_samples'),
                                                  control_trace=control_trace, chunk_record=chunk_record)
            total_controls += result['steps']
            successes += int(result['success'])
            append('rollouts.jsonl', dict(phase=phase, iteration=iteration, **result,
                   reward_version=args.reward_version, reward_sum=sum(t['reward'] for t in episode),
                   official_reward_sum=sum(t['official_reward'] for t in episode),
                   progress_reward_sum=sum(t['progress_reward'] for t in episode)))
            for chunk, transition in enumerate(episode):
                append('transitions.jsonl', dict(phase=phase, iteration=iteration, seed=seed, chunk=chunk,
                       **{key: value for key, value in transition.items()
                          if key not in ('row', 'chain', 'denoise_inds', 'old_logprobs')}))
            samples.extend(episode)
            print(json.dumps(dict(phase=phase, iteration=iteration, seed=seed,
                                  success=result['success'], controls=result['steps'])), flush=True)
        return samples, successes

    def check_logprobs(samples):
        report = check_replay(samples, lambda t: replay_scores(model, encoding, t)[0],
                              lambda row: append('replay-checks.jsonl', dict(total_controls=total_controls, **row)))
        return report['aggregate_max_abs_error']

    def update(samples, phase, iteration, epochs=1):
        nonlocal total_steps

        def record(entry):
            nonlocal total_steps
            total_steps += int(entry['optimizer_step_applied'])
            append('updates.jsonl', dict(iteration=iteration, total_optimizer_steps=total_steps,
                                       total_controls=total_controls, **entry))

        return optimize(model, optimizer, samples, replay, rng, minibatch=args.minibatch, epochs=epochs,
                        critic_only=phase == 'critic_warmup', target_kl=args.target_kl, record=record)

    if args.critic_warmup_epochs:
        samples, _ = collect('critic_warmup', 0)
        error = check_logprobs(samples)
        before = value_metrics(samples)
        critic_before = [p.detach().clone() for p in critic]
        history = update(samples, 'critic_warmup', 0, args.critic_warmup_epochs)
        current = trainable_snapshot(model, actor_only=True)
        if any(not torch.equal(current[name], value) for name, value in original.items()):
            raise RuntimeError('Critic warmup changed VLA actor parameters')
        critic_delta = max(float((p.detach() - old).abs().max()) for p, old in zip(critic, critic_before))
        if critic_delta <= 0:
            raise RuntimeError('Critic warmup did not change the value head')
        with torch.no_grad():
            for transition in samples:
                transition['value'] = float(replay(transition)[1])
        warmup = dict(before=before, after=value_metrics(samples), actor_unchanged=True,
                      actor_max_abs_delta=0.0, critic_max_abs_delta=critic_delta,
                      optimizer_steps=len(history), total_controls=total_controls,
                      preupdate_logprob_abs_error=error,
                      target='discounted Monte Carlo return; no random-critic bootstrapping')
        (args.output / 'warmup.json').write_text(json.dumps(warmup, indent=2) + '\n', encoding='utf-8')
        del samples, current, critic_before

    for iteration in range(1, args.iterations + 1):
        samples, successes = collect('ppo', iteration)
        error = check_logprobs(samples)
        before = value_metrics(samples)
        history = update(samples, 'ppo', iteration)
        final_kl = prefix_kl(samples, replay)
        weights = trainable_snapshot(model)
        changes = actor_change_evidence(weights, original)
        changed = any(row['changed_tensors'] for row in changes.values())
        effective_steps = sum(row['actor_max_abs_delta_this_step'] > 0 for row in history)
        report = dict(**manifest, optimizer_iterations=iteration, successes=successes, trials=len(args.seeds),
                      total_controls=total_controls, total_optimizer_steps=total_steps,
                      actor_optimizer_steps=effective_steps, replay_logprob_abs_error=error,
                      postupdate_full_batch_prefix_approx_kl=final_kl if math.isfinite(final_kl) else None,
                      fresh_rollout_value_metrics=before, actor_parameter_changes=changes,
                      stopped_for_kl=any(row['stopped_for_kl'] for row in history),
                      wall_seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.output / f'iteration-{iteration}.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if not changed or not effective_steps:
            raise RuntimeError('No effective actor update; diagnostics saved, no PPO checkpoint claimed')
        if not math.isfinite(final_kl) or final_kl > args.target_kl:
            raise RuntimeError('Final full-batch prefix score exceeds the KL guard; no checkpoint saved')
        torch.save(dict(policy=weights, optimizer=optimizer.state_dict(), metadata=metadata, ppo=report),
                   args.output / f'ppo-{iteration}.pt')
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
