"""Exercise actual PPOClient and chunk reward/return accounting without GPU weights."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from train_fruit_ppo import collect_rollout, MILESTONE_REWARD, fruit_reward

class Target:
    def __init__(self, env): self.env = env
    def get_xpos(self, physics): return np.array([0., 0., .79 + .07 * (self.env.n >= self.env.lift_at)])
    def is_grasped(self, physics, robot): return self.env.n >= self.env.lift_at
class Env:
    def __init__(self, *, lift_at=4, controls=12, solved=True):
        from types import SimpleNamespace
        self.n=0; self.lift_at=lift_at; self.controls=controls; self.solved=solved
        self.physics=object(); self.robot=object()
        self.task=SimpleNamespace(target_entity='apple', entities={'apple': Target(self)})
    def step(self, action):
        from types import SimpleNamespace
        self.n += 1
        return SimpleNamespace(last=lambda: self.n == self.controls)
    def __enter__(self): return self
    def __exit__(self, *exc): pass
class Model:
    def _predict_train(self, obs, **kwargs):
        return None, dict(forward_inputs=dict(chains=torch.zeros(1, 2, 10, 32), denoise_inds=torch.zeros(1,dtype=torch.long)),
                          prev_values=torch.zeros(1), prev_logprobs=torch.zeros(1,5,7), model_actions=torch.zeros(1,10,32))
class Policy:
    model=Model()
    def __init__(self): self.rows=[]
    def encode(self,row):
        assert set(row)=={'observation/state','prompt','observation/image','observation/second_image','observation/wrist_image'}
        self.rows.append(row)
        return None,None,dict(state=np.zeros(32))
    def outputs(self,row): return dict(actions=np.zeros((10,7)))

def run_case(env):
    import select_fruit, evaluate_vlabench_reference
    policy=Policy(); original=env.step
    def build(env,client,seed,controls,record,trace=None):
        result=dict(steps=0,success=False)
        def rollout():
            for i in range(env.controls):
                if i%5==0:
                    client.infer({'observation/state':np.zeros(7),'prompt':'Put the apple into the plate_seen',
                                  **{k:np.zeros((2,2,3),dtype=np.uint8) for k in ('observation/image','observation/second_image','observation/wrist_image')}})
                env.step(np.zeros(9))
                result.update(steps=i+1,success=bool(env.solved and i+1==env.controls))
                yield None
        return result, rollout()
    with patch.object(select_fruit,'open_episode',return_value=env), patch.object(select_fruit,'success',side_effect=lambda e:bool(e.solved and e.n==e.controls)), patch.object(evaluate_vlabench_reference,'build_episode',side_effect=build):
        transitions,result=collect_rollout(policy,lambda client,**kw:client,1301,None,env.controls,0,MILESTONE_REWARD,0)
    assert env.step==original
    assert sum(t['executed'] for t in transitions)==env.controls
    assert len(policy.rows)==len(transitions)
    assert transitions[-1]['terminal'] and not any(t['terminal'] for t in transitions[:-1])
    assert all(np.isfinite(t['advantage']) and np.isfinite(t['mc_return']) for t in transitions)
    return transitions,result

def test_lift_event_crossing_chunk_and_partial_terminal():
    ts,result=run_case(Env(lift_at=4,controls=12))
    assert [t['executed'] for t in ts]==[5,5,2]
    assert [t['reward'] for t in ts]==[0,.2,.8]
    assert [e['control'] for t in ts for e in t['events']]==[6,12]
    assert sum(t['official_reward'] for t in ts)==.8
    assert sum(t['progress_reward'] for t in ts)==.2
    assert ts[0]['mc_return']==pytest.approx(.99*.2+.99**2*.8)
    assert all(t['old_logp']==0 and tuple(t['old_logprobs'].shape)==(t['executed'],7) for t in ts)

def test_event_on_exact_five_control_boundary_counted_once():
    ts,_=run_case(Env(lift_at=3,controls=10))
    assert [t['reward'] for t in ts]==[.2,.8]
    assert [t['executed'] for t in ts]==[5,5]

def test_failed_task_keeps_lift_but_does_not_invent_success():
    ts,result=run_case(Env(lift_at=4,controls=12,solved=False))
    assert not result['success'] and sum(t['reward'] for t in ts)==.2
    assert sum(t['official_reward'] for t in ts)==0

def test_no_lift_no_reward_and_exact_budget_terminal():
    ts,result=run_case(Env(lift_at=99,controls=10,solved=False))
    assert [t['reward'] for t in ts]==[0,0]
    assert ts[-1]['terminal']

def test_milestone_cannot_fall_through_to_distance_shaping():
    with pytest.raises(ValueError): fruit_reward(0,1,False,False,0,MILESTONE_REWARD)
    with pytest.raises(ValueError):
        collect_rollout(None,None,1301,None,10,.5,MILESTONE_REWARD,0)
