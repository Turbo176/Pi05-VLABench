"""Project transforms and shared single-GPU RLinf Pi0 / Pi0RL loading."""

import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from openpi.transforms import Normalize, Unnormalize
from rlinf.models.embodiment.openpi.modules.lora import FeedForward as LoRAFeedForward, LoRAConfig
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from safetensors.torch import load_file

from vlabench_policy_io import (ACTION_HORIZON, ACTION_SPACES, ACTUATOR_DIM, MODEL_ACTION_DIM,
                                absolute_arm_targets, actuator_targets, relative_arm_targets)

ACTOR_MODULES = ('action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
DENOISE_STEPS = 10


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PolicyIO:
    def __init__(self, stats_file, device, token_length, action_space='absolute'):
        if action_space not in ACTION_SPACES:
            raise ValueError(f'Unknown action space {action_space}')
        self.device = device
        self.action_space = action_space
        self.tokenizer = PaligemmaTokenizer(token_length)
        self.stats_sha256 = hashlib.sha256(Path(stats_file).read_bytes()).hexdigest()
        stats = normalize.deserialize_json(Path(stats_file).read_text(encoding='utf-8'))
        for key in ('state', 'actions'):
            allowed_shapes = [(ACTUATOR_DIM,)]
            if key == 'actions' and action_space == 'arm_delta':
                allowed_shapes.append((ACTION_HORIZON, ACTUATOR_DIM))
            low, high = stats[key].q01, stats[key].q99
            if (low is None or high is None or low.shape not in allowed_shapes or high.shape != low.shape
                    or not np.isfinite(low).all() or not np.isfinite(high).all() or not np.all(high > low)):
                raise ValueError(f'Invalid nine-coordinate quantiles for {key}')
        self.normalize = Normalize(stats, use_quantiles=True)
        self.unnormalize = Unnormalize(stats, use_quantiles=True)
        self.unnormalize_actions = Unnormalize({'actions': stats['actions']}, use_quantiles=True)

    def encode(self, row):
        fields = {'state': row['state']}
        if 'actions' in row:
            actions = row['actions']
            fields['actions'] = relative_arm_targets(actions, row['state']) if self.action_space == 'arm_delta' else actions
        scaled = self.normalize(fields)
        tokens, mask = self.tokenizer.tokenize(row['prompt'], scaled['state'])
        obs = Observation.from_dict({
            'image': {key: self.tensor(rgb) for key, rgb in row['image'].items()},
            'image_mask': {key: self.tensor(np.asarray(value)) for key, value in row['image_mask'].items()},
            'state': self.tensor(scaled['state'].astype(np.float32)),
            'tokenized_prompt': self.tensor(tokens.astype(np.int64)),
            'tokenized_prompt_mask': self.tensor(mask),
        })
        labels = None
        if 'actions' in scaled:
            actions = scaled['actions'].astype(np.float32)
            if actions.shape != (ACTION_HORIZON, ACTUATOR_DIM):
                raise ValueError('Expected a complete 50-step action window')
            labels = self.tensor(np.pad(actions, ((0, 0), (0, MODEL_ACTION_DIM - ACTUATOR_DIM))))
        return obs, labels

    def tensor(self, array):
        return torch.from_numpy(np.array(array, copy=True))[None].to(self.device)

    def decode(self, output, state=None, *, normalized_state=False):
        data = {'actions': actuator_targets(output.detach().float().cpu().numpy()[0])}
        if self.action_space == 'arm_delta':
            if state is None:
                raise ValueError('Arm-delta actions require the measured pose at the start of the chunk')
            if normalized_state:
                data['state'] = np.asarray(state, dtype=np.float32)[:ACTUATOR_DIM]
        decoded = self.unnormalize(data) if 'state' in data else self.unnormalize_actions(data)
        actions = decoded['actions']
        if self.action_space == 'arm_delta':
            actions = absolute_arm_targets(actions, decoded['state'] if normalized_state else state)
        return actuator_targets(actions)


def is_action_expert_parameter(name):
    return bool(re.match(r'^llm\.(?:layers\.\d+\.(?:attn\.[qkvo]_proj|mlps|pre_attention_norms|pre_ffw_norms)\.1\.|final_norms\.1\.)', name))


def is_actor_parameter(name):
    return (name.startswith(ACTOR_MODULES) or is_action_expert_parameter(name)
            or (name.startswith('llm.') and '_lora_' in name))


class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.base = linear
        device = linear.weight.device
        self.q_lora_a = torch.nn.Parameter(torch.empty(rank, linear.in_features, device=device))
        self.q_lora_b = torch.nn.Parameter(torch.zeros(linear.out_features, rank, device=device))
        torch.nn.init.normal_(self.q_lora_a, std=0.01)

    def forward(self, inputs):
        first = torch.nn.functional.linear(inputs, self.q_lora_a.to(inputs.dtype))
        return self.base(inputs) + torch.nn.functional.linear(first, self.q_lora_b.to(inputs.dtype))


def install_action_expert_lora(model, rank):
    """Add zero-initialized rank-r updates to the action expert only."""
    if rank < 0:
        raise ValueError('LoRA rank must be nonnegative')
    if not rank:
        return
    for layer in model.llm.layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            projections = getattr(layer.attn, name)
            if projections[1] is not None:
                projections[1] = LoRALinear(projections[1], rank)
        base_mlp = layer.mlps[1]
        lora_mlp = LoRAFeedForward(base_mlp.features, base_mlp.hidden_dim, LoRAConfig(rank, alpha=rank))
        lora_mlp.w_gating = base_mlp.w_gating
        lora_mlp.w_linear = base_mlp.w_linear
        layer.mlps[1] = lora_mlp.to(base_mlp.w_linear.device)


def install_vlm_lora(model, rank):
    """Adapt query and value projections in the last four language-model layers."""
    if rank < 0:
        raise ValueError('VLM LoRA rank must be nonnegative')
    if not rank:
        return
    for layer in model.llm.layers[-4:]:
        for name in ('q_proj', 'v_proj'):
            projections = getattr(layer.attn, name)
            projections[0] = LoRALinear(projections[0], rank)


def build_model(base, device, execute_steps, rl=False, lora_rank=0, vlm_lora_rank=0, *,
                action_horizon=ACTION_HORIZON, action_env_dim=ACTUATOR_DIM, discrete_state_input=None,
                train_action_heads=True, train_full_action_expert=False):
    if not 1 <= execute_steps <= action_horizon:
        raise ValueError('Execution length must fit the prediction horizon')
    set_torch_compile(False)
    config = Pi0Config(pi05=True, action_horizon=action_horizon, dtype='bfloat16',
                       discrete_state_input=discrete_state_input)
    options = dict(num_steps=DENOISE_STEPS, action_env_dim=action_env_dim, action_chunk=execute_steps)
    if rl:
        from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig
        model = Pi0RL(config, **options, rl_cfg=Pi0RLConfig(add_value_head=True,
                     noise_method='flow_sde', noise_level=0.25, ignore_last=True,
                     train_expert_only=not vlm_lora_rank, detach_critic_input=True, value_after_vlm=True))
    else:
        model = Pi0(config, **options)
    if base is not None:
        weights = load_file(str(base), device='cpu')
        report = model.load_state_dict(weights, strict=False)
        expected = {name for name in model.state_dict() if rl and name.startswith('value_head.')}
        if set(report.missing_keys) != expected or report.unexpected_keys:
            raise RuntimeError(f'Unexpected checkpoint keys: {report}')
        del weights
    model.to_bfloat16_for_selected_params()
    model.to(device)
    install_action_expert_lora(model, lora_rank)
    install_vlm_lora(model, vlm_lora_rank)
    model.requires_grad_(False)
    for name, value in model.named_parameters():
        actor = is_actor_parameter(name) and (train_action_heads or not name.startswith(ACTOR_MODULES))
        if is_action_expert_parameter(name) and '_lora_' not in name and not train_full_action_expert:
            actor = False
        if actor or (rl and name.startswith('value_head.')):
            value.requires_grad_(True)
    return model.eval(), config


def trainable_snapshot(model, *, actor_only=False):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and (not actor_only or is_actor_parameter(n))}


def restore_trainable(model, weights):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    actor = {name for name in expected if is_actor_parameter(name)}
    if set(weights) not in (expected, actor) or not actor:
        raise ValueError('Checkpoint trainable keys differ from the policy')
    with torch.no_grad():
        for name, value in weights.items():
            if parameters[name].shape != value.shape or parameters[name].dtype != value.dtype:
                raise ValueError(f'Checkpoint tensor differs in shape or dtype: {name}')
            parameters[name].copy_(value)


def actor_delta(model, old):
    new = trainable_snapshot(model, actor_only=True)
    return max(float((new[key] - old[key]).abs().max()) for key in old)
