"""Select PPO on registered development episodes; never query the reserved test here."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc,hashlib,json,math,os
from pathlib import Path
from threading import Lock
import jax
import torch
import train_reference_aligned_fruit as trainer
from train_reference_aligned_fruit import load_aligned_policy,evaluate,paired_episodes
from evaluate_pose_development import LockedPolicy
from pi05_runtime import file_sha256,is_actor_parameter,restore_trainable
r=Path('runs/fruit-full-ppo-reward-compare-20261124')
parent=Path('runs/fruit-full-train47-20261124/storage/sft')
out=r/'evaluation';out.mkdir(exist_ok=False)
for arm in ['potential','milestone']:
    assert (r/f'{arm}-train.exit-code').exists(),'Both planned training arms must have ended before selection'
expected=paired_episodes(Path('runs/fruit-recovery-20261124/development-sft180/episodes.jsonl'))
assert len(expected)==16
screen=[1201,1202,1203,1204]
base=Path('.local/checkpoints/vlabench-primitive-rlinf/model.safetensors')
policy=load_aligned_policy(parent,base,Path(os.environ['PI05_REFERENCE_FORK']))
assert policy.provenance['sft_sha256']=='954f168626a30a068aae7e5c0ef0cc37a361e4ac068136d26e1ce02e46964bee'
jax.devices('cpu')
original_provenance=dict(policy.provenance)

def load_candidate(path):
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    p=saved['ppo']
    assert p['sft_sha256']==original_provenance['sft_sha256']
    assert p['norm_stats_sha256']==original_provenance['norm_stats_sha256']
    assert p['actor_optimizer_steps']>0 and p['train_full_action_expert']
    assert p['postupdate_full_batch_prefix_approx_kl'] is not None and p['postupdate_full_batch_prefix_approx_kl']<=.02
    actor={k:v for k,v in saved['policy'].items() if is_actor_parameter(k)}
    restore_trainable(policy.model,actor)
    sha=file_sha256(path)
    policy.provenance=dict(original_provenance,ppo_sha256=sha)
    del saved,actor
    gc.collect()
    return sha

def evaluate_cases(destination,seeds):
    destination.mkdir(parents=True,exist_ok=False)
    reset_lock,inference_lock=Lock(),Lock()
    original_open=trainer.open_episode
    def locked_open(*args,**kwargs):
        with reset_lock:return original_open(*args,**kwargs)
    def case(seed):
        agent=LockedPolicy(policy,inference_lock)
        target=destination/f'case-{seed}'
        evaluate(agent,target,expected,600,False,'ppo',seeds=[seed],render_samples=0)
        row,=[json.loads(line) for line in (target/'episodes.jsonl').read_text().splitlines()]
        return row
    trainer.open_episode=locked_open
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=[future.result() for future in as_completed([pool.submit(case,seed) for seed in seeds])]
    finally:trainer.open_episode=original_open
    rows.sort(key=lambda row:row['seed'])
    assert {row['seed'] for row in rows}==set(seeds)
    (destination/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return rows

report=dict(complete=False,reserved_test_used=False,parent_sft_sha256=original_provenance['sft_sha256'],candidates=[])
for arm in ['potential','milestone']:
    for iteration in [1,2]:
        path=r/'storage'/arm/f'ppo-{iteration}.pt'
        if not path.exists():continue
        sha=load_candidate(path)
        rows=evaluate_cases(out/f'{arm}-{iteration}-screen',screen)
        candidate=dict(arm=arm,iteration=iteration,checkpoint=str(path),sha256=sha,screen_successes=sum(row['success'] for row in rows),screen_trials=4,screen_controls=sum(row['steps'] for row in rows))
        report['candidates'].append(candidate)
        (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(candidate),flush=True)
assert report['candidates'],'No PPOcheckpoint with effective actor updates'
best=min(report['candidates'],key=lambda row:(-row['screen_successes'],row['screen_controls'],row['iteration'],row['arm']))
assert load_candidate(Path(best['checkpoint']))==best['sha256']
remaining=sorted(set(expected)-set(screen));assert len(remaining)==12
rows=evaluate_cases(out/'selected-remaining-development',remaining)
first=out/f"{best['arm']}-{best['iteration']}-screen/episodes.jsonl"
rows.extend(json.loads(line) for line in first.read_text().splitlines());rows.sort(key=lambda row:row['seed'])
assert len(rows)==16 and {row['seed'] for row in rows}==set(expected)
assert all(row['ppo_sha256']==best['sha256'] for row in rows)
(out/'selected-development.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
(out/'development-reuse.json').write_text(json.dumps(dict(screen_source=str(first),screen_source_sha256=file_sha256(first),reused_seeds=screen,newly_executed_seeds=remaining,policy_sha256=best['sha256'],same_initializations_sampler_and_control_budget=True),indent=2)+'\n')
report.update(complete=True,selected_candidate=best,selected_development_successes=sum(row['success'] for row in rows),selected_development_trials=16,selected_development_controls=sum(row['steps'] for row in rows),sft_development_successes=9)
report['ppo_better_than_sft_development']=report['selected_development_successes']>9
(out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
reserved=Path('runs/fruit-long-sft-20261124/reserved-test/select_fruit-official.jsonl')
assert file_sha256(reserved)=='8d62b9aaee8fd018143cd82efbfeb8c301472adf4ab6e0c730a89cebf5e1da78'
lock=dict(reserved_test_manifest=str(reserved),reserved_test_manifest_sha256=file_sha256(reserved),test_policy_calls_so_far=0,selection_used_only_development=True,upstream_reference=dict(base_sha256=original_provenance['base_sha256'],normalization_sha256='bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f',label='upstream VLABench-finetuned reference, not raw foundation model'),sft=dict(checkpoint=str(parent),sha256=original_provenance['sft_sha256'],normalization_sha256=original_provenance['norm_stats_sha256']),ppo=dict(checkpoint=best['checkpoint'],sha256=best['sha256'],parent_sft_sha256=original_provenance['sft_sha256'],normalization_sha256=original_provenance['norm_stats_sha256']))
(out/'final-test-model-lock.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(report),flush=True)
