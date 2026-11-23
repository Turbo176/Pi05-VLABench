import gc
import json
from pathlib import Path
import jax
import numpy as np
import torch
from VLABench.utils.utils import quaternion_to_euler
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training.config import get_config
from pi05_runtime import build_model
from select_fruit import camera_names, open_episode
from train_reference_aligned_fruit import AlignedPolicy, DIM, EXECUTE, HORIZON, PADDED_DIM

fork = next(Path('.local/upstream').glob('VLABench-openpi-*')).resolve()
orbax_root = next(Path('.local/checkpoints/vlabench-primitive-10task').glob('f658*')).resolve()
converted = Path('.local/checkpoints/vlabench-primitive-rlinf')
with open_episode(1201, 'select_fruit') as env:
    cameras = camera_names(env)
    images = [env.render(camera_id=cameras.index(key), **env.render_options) for key in
              ('forward', 'right', 'franka/Franka_wrist_cam')]
    ee = env.robot.get_ee_state(env.physics)
    state = np.r_[ee[:3] - env.get_robot_frame_position(), quaternion_to_euler(ee[3:7]), ee[-1]]
    raw = {'observation/image': images[0], 'observation/second_image': images[1],
           'observation/wrist_image': images[2], 'observation/state': state,
           'prompt': env.task.get_instruction()}
_, key = jax.random.split(jax.random.key(1201))
noise = np.array(jax.random.normal(key, (1, HORIZON, PADDED_DIM)))
reference = policy_config.create_trained_policy(get_config('pi05_ft_vlabench_primitive'), orbax_root)
expected = reference.infer(raw, noise=noise)['actions']
print('Computed native reference action', flush=True)
del reference
gc.collect()
model, _ = build_model(converted / 'model.safetensors', 'cuda', EXECUTE, action_horizon=HORIZON,
                       action_env_dim=DIM, discrete_state_input=False)
stats = normalize.deserialize_json((converted / 'norm_stats.json').read_text())
policy = AlignedPolicy(model, stats, fork, shortest_euler=False)
with torch.inference_mode():
    obs, _, row = policy.encode(raw)
    prediction = model.sample_actions(obs, noise=torch.from_numpy(noise).to('cuda'))
    actual = policy.outputs({'actions': prediction[0].float().cpu().numpy(), 'state': row['state']})['actions']
errors = actual[:EXECUTE] - expected[:EXECUTE]
errors[:,3:6] = (errors[:,3:6] + np.pi)%(2*np.pi)-np.pi
report = {'seed':1201,'frame':0,'jax_device': str(jax.devices()[0]),'torch_device': torch.cuda.get_device_name(),
          'shared_noise_shape': list(noise.shape),'xyz_mae_m':float(np.abs(errors[:,:3]).mean()),
          'xyz_max_m':float(np.abs(errors[:,:3]).max()), 'euler_mae_rad':float(np.abs(errors[:,3:6]).mean()),
          'euler_max_rad':float(np.abs(errors[:,3:6]).max()), 'gripper_max':float(np.abs(errors[:,6]).max())}
Path('logs/reference-aligned/backend-first-window.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
