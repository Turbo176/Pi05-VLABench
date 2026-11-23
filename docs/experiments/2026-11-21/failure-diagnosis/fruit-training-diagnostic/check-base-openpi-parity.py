import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:6]))
import jax
import jax.numpy as jnp
import numpy as np
import torch

from evaluate_pi05 import load_stage
from openpi.models import model as openpi_model
from openpi.models.pi0_config import Pi0Config
from openpi.shared.nnx_utils import module_jit
from pi05_data import LeRobotTrajectories

start = time.monotonic()
weights = Path('.local/checkpoints/pi05-base-rlinf/model.safetensors')
stats = Path('.local/compact-data/arm-delta-stats/norm_stats.json')
data = LeRobotTrajectories(Path('.local/compact-data/train'), Path('incoming/train-verification.json'))
row = data.example(0, 34)
model, policy, _ = load_stage(SimpleNamespace(base=weights, stats=stats, checkpoint=None,
                                              execute_steps=10, action_space='arm_delta', stage='base'))
with torch.inference_mode():
    obs, _ = policy.encode(row)
    noise = np.random.default_rng(42).normal(size=(1, 50, 32)).astype(np.float32)
    converted = model.sample_actions(obs, num_steps=10, noise=torch.from_numpy(noise).to('cuda'))
    converted = converted.detach().cpu().float().numpy()
    original_data = dict(image={k:v.detach().cpu().numpy() for k,v in obs.images.items()},
                         image_mask={k:v.detach().cpu().numpy() for k,v in obs.image_masks.items()},
                         state=obs.state.detach().cpu().numpy(),
                         tokenized_prompt=obs.tokenized_prompt.cpu().numpy().astype(np.int32),
                         tokenized_prompt_mask=obs.tokenized_prompt_mask.cpu().numpy())
del model, obs, policy
torch.cuda.empty_cache()
print('Collected converted policy output', flush=True)
original_observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, original_data))
native_config = Pi0Config(pi05=True, action_horizon=50, max_token_len=200)
params = openpi_model.restore_params(Path('.local/checkpoints/pi05_base/params'), dtype=jnp.bfloat16)
native = native_config.load(params)
print('Restored original Orbax weights', flush=True)
sample = module_jit(native.sample_actions)
original = np.asarray(sample(jax.random.key(1), original_observation, noise=jnp.asarray(noise)), dtype=np.float32)
error = abs(original - converted)
report = {'device_native': str(jax.devices()[0]), 'first_10_actuator_mae_normalized': float(error[:, :10, :9].mean()),
          'all_actuator_mae_normalized': float(error[:, :, :9].mean()), 'all_32_dims_mae_normalized': float(error.mean()),
          'max_abs_normalized_difference': float(error.max()), 'wall_seconds': time.monotonic()-start}
Path('logs/base-openpi-native-parity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
