import hashlib, json, pathlib
import mujoco
import numpy as np
from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from select_fruit import digest_arrays, open_episode
from summarize_pi05 import initial_rgb_key
from train_reference_aligned_fruit import paired_episodes
from audit_pose_reproducibility import integration_signature

root=pathlib.Path.cwd()
out=root/'runs/diagnostic-20261122-flower-fixed-first-chunk'
out.mkdir(exist_ok=False)
expected=paired_episodes(root/'.local/reference-aligned/formal-cases/insert_flower-official.jsonl')[1001]
actions=np.load(root/'runs/diagnostic-20261122-flower-ppo-repeat/repeat-1-predictions.npz')['actions'][0]
adapter=load_upstream_adapter(root/'.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387')
class Fixed:
    def infer(self, observation):
        return {'actions':actions.copy()}
def integration(env):
    mask=mujoco.mjtState.mjSTATE_INTEGRATION
    data=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,mask))
    mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,data,mask)
    return data
model_fields=('body_mass','body_inertia','dof_damping','dof_frictionloss','geom_pos','geom_quat','geom_size',
              'geom_type','geom_friction','tendon_stiffness','tendon_damping','light_pos','cam_pos','cam_quat')
records=[]
for repeat in range(6):
    for attempt in range(4):
        try: env=open_episode(1001,'insert_flower')
        except Exception as error:
            print(type(error).__name__,str(error),flush=True)
            continue
        with env:
            row, replay=build_episode(env,adapter(Fixed(),replan_steps=5),1001,5,False,task='insert_flower')
            key=initial_rgb_key(row,expected)
            if any(row[k]!=expected[k] for k in ('initial_state_sha256',key)):
                replay.close()
                continue
            before=integration(env)
            model={key:digest_arrays(np.asarray(getattr(env.physics.model,key))) for key in model_fields}
            for _ in replay: pass
            after=integration(env)
            frames=np.stack([env.render(camera_id=i,**env.render_options) for i in range(4)])
            repeated=np.stack([env.render(camera_id=i,**env.render_options) for i in range(4)])
            np.savez_compressed(out/f'repeat-{repeat+1}.npz',before=before,after=after,frames=frames,repeated=repeated)
            result=dict(repeat=repeat+1,attempt=attempt+1,initial=row['initial_state_sha256'],
                model=model,before=digest_arrays(before),after=digest_arrays(after),
                camera_hashes=[digest_arrays(view) for view in frames],
                repeated_camera_hashes=[digest_arrays(view) for view in repeated],
                action_sha256=row['action_sha256'])
            records.append(result)
            print(json.dumps(result),flush=True)
            break
    else: raise RuntimeError('Reset did not match')
(out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
base=np.load(out/'repeat-1.npz')
comparisons=[]
for i in range(2,7):
    other=np.load(out/f'repeat-{i}.npz')
    comparisons.append(dict(repeat=i,
        before_equal=bool(np.array_equal(base['before'],other['before'])),
        after_equal=bool(np.array_equal(base['after'],other['after'])),
        max_integration_difference=float(np.max(np.abs(base['after']-other['after']))),
        per_camera_changed_channels=[int(np.count_nonzero(left!=right)) for left,right in zip(base['frames'],other['frames'])],
        per_camera_max_byte_difference=[int(np.max(np.abs(left.astype(int)-right.astype(int)))) for left,right in zip(base['frames'],other['frames'])],
        camera_same_state_repeat_changes=[int(np.count_nonzero(left!=right)) for left,right in zip(other['frames'],other['repeated'])],
        model_equal=records[0]['model']==records[i-1]['model'],
        commands_equal=records[0]['action_sha256']==records[i-1]['action_sha256']))
(out/'summary.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(comparisons),flush=True)

