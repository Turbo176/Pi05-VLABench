# Source used for the executed, offline parameter-group comparison.
from pathlib import Path
import json
import torch
from train_reference_aligned_fruit import load_aligned_policy, validation_mae
from pi05_runtime import trainable_snapshot, restore_trainable, ACTOR_MODULES
from reference_pose_lerobot import load_pose_split

root = Path('runs/pose-robustness-v2')
policy = load_aligned_policy(
    root / 'sft-300', Path('.local/checkpoints/vlabench-primitive-rlinf/model.safetensors'),
    Path('.local/upstream/VLABench-openpi-788ef1f20e655de5bb40869fb7b7fcdd49732387'), stage='base')
base = trainable_snapshot(policy.model)
trained = torch.load(root / 'sft-300/sft.pt', weights_only=True, map_location='cpu')['policy']
episodes = load_pose_split(Path('.local/reference-aligned/lerobot-tritask-msaa0-v2'), 'validation')
report = {}
filters = {
    'reference': lambda name: False,
    'heads_only': lambda name: name.startswith(ACTOR_MODULES),
    'lora_only': lambda name: name.startswith('llm.'),
    'vlm_only': lambda name: '.0.q_lora_' in name,
    'action_lora_only': lambda name: name.startswith('llm.') and '.0.q_lora_' not in name,
}
with torch.inference_mode():
    for name, select in filters.items():
        restore_trainable(policy.model, {key: trained[key] if select(key) else value for key, value in base.items()})
        report[name] = {
            task: validation_mae(policy, [episode for episode in episodes if episode.task == task])
            for task in sorted({episode.task for episode in episodes})}
        print(json.dumps(dict(variant=name, metrics=report[name])), flush=True)
(root / 'parameter-ablation.json').write_text(json.dumps(
    dict(kind='offline_parameter_ablation_not_new_policy_training', results=report), indent=2) + '\n')
