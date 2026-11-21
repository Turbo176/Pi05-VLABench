"""Project transforms and shared single-GPU RLinf Pi0 / Pi0RL loading."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from openpi.transforms import Normalize, Unnormalize
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from safetensors.torch import load_file

from vlabench_policy_io import ACTION_HORIZON, ACTUATOR_DIM, MODEL_ACTION_DIM, actuator_targets

ACTOR_MODULES = ('action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
DENOISE_STEPS = 10


class PolicyIO:
    def __init__(self, stats_file, device, token_length):
        self.device = device
        self.tokenizer = PaligemmaTokenizer(token_length)
        self.stats_sha256 = hashlib.sha256(Path(stats_file).read_bytes()).hexdigest()
        stats = normalize.deserialize_json(Path(stats_file).read_text(encoding='utf-8'))
        for key in ('state', 'actions'):
            if stats[key].q01.shape != (ACTUATOR_DIM,) or not np.all(stats[key].q99 > stats[key].q01):
                raise ValueError(f'Invalid nine-coordinate quantiles for {key}')
        self.normalize = Normalize(stats, use_quantiles=True)
        self.unnormalize = Unnormalize({'actions': stats['actions']}, use_quantiles=True)

    def encode(self, row):
        fields = {'state': row['state']}
        if 'actions' in row:
            fields['actions'] = row['actions']
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

    def decode(self, output):
        actions = actuator_targets(output.detach().float().cpu().numpy()[0])
        return actuator_targets(self.unnormalize({'actions': actions})['actions'])


def build_model(base, device, execute_steps, rl=False):
    set_torch_compile(False)
    config = Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, dtype='bfloat16')
    options = dict(num_steps=DENOISE_STEPS, action_env_dim=ACTUATOR_DIM, action_chunk=execute_steps)
    if rl:
        from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig
        model = Pi0RL(config, **options, rl_cfg=Pi0RLConfig(add_value_head=True,
                     noise_method='flow_sde', noise_level=0.25, ignore_last=True,
                     train_expert_only=True, detach_critic_input=True, value_after_vlm=True))
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
    model.requires_grad_(False)
    for name, value in model.named_parameters():
        if name.startswith(ACTOR_MODULES) or (rl and name.startswith('value_head.')):
            value.requires_grad_(True)
    return model.eval(), config


def trainable_snapshot(model, *, actor_only=False):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and (not actor_only or n.startswith(ACTOR_MODULES))}


def restore_trainable(model, weights):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    actor = {name for name in expected if name.startswith(ACTOR_MODULES)}
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
