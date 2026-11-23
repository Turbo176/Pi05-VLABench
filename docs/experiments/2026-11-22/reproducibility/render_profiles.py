import json, pathlib
import numpy as np
from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from select_fruit import open_episode

root=pathlib.Path.cwd()
out=root/'runs/diagnostic-20261122-render-profiles'
out.mkdir(exist_ok=False)
actions=np.load(root/'runs/diagnostic-20261122-flower-ppo-repeat/repeat-1-predictions.npz')['actions'][0]
adapter=load_upstream_adapter(root/'.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387')
class Fixed:
    def infer(self, observation): return {'actions':actions.copy()}
profiles=[('default',None,None),('no_shadows',{'shadow':False},None),('no_reflections',{'reflection':False},None),
          ('no_shadow_reflection',{'shadow':False,'reflection':False},None),('no_msaa',None,0)]
results=[]
for label,flags,samples in profiles:
    with open_episode(1001,'insert_flower') as env:
        original_samples=int(env.physics.model.vis.quality.offsamples)
        if samples is not None: env.physics.model.vis.quality.offsamples=samples
        row,replay=build_episode(env,adapter(Fixed(),replan_steps=5),1001,5,False,task='insert_flower')
        for _ in replay: pass
        baseline=np.stack([env.render(camera_id=i,**env.render_options,render_flag_overrides=flags) for i in range(4)])
        differences=[]
        for repeat in range(16):
            frame=np.stack([env.render(camera_id=i,**env.render_options,render_flag_overrides=flags) for i in range(4)])
            differences.append([int(np.count_nonzero(a!=b)) for a,b in zip(baseline,frame)])
        item=dict(profile=label,original_samples=original_samples,flags=flags,samples=samples,
                  changed_channels=differences,unstable_render_count=sum(any(diff) for diff in differences))
        results.append(item)
        print(json.dumps(item),flush=True)
        np.savez_compressed(out/f'{label}.npz',baseline=baseline,last=frame)
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

