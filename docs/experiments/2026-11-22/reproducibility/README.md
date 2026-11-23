# Renderer repeatability audit

An earlier flower PPO policy produced different later action chunks despite the same reset, weights, normalization and sampling seed. The [original repeat audit](flower-ppo-original/summary.json) records one success and one failure. A [fixed-action probe](fixed-first-chunk/summary.json) found matching MuJoCo integration states but occasional one-level RGB differences.

Each graphics profile then rendered four cameras sixteen times after the same fixed controls:

| Profile | Render sets differing from the first set |
| --- | ---: |
| Default, four MSAA samples | 16/16 |
| Shadows disabled | 9/16 |
| Reflections disabled | 10/16 |
| Shadows and reflections disabled | 14/16 |
| MSAA samples set to zero | 0/16 |

[Graphics measurements](render-profiles/summary.json) retain all profiles. This is a renderer check, not a task-success comparison. With MSAA disabled, the same earlier PPO checkpoint repeated its complete 600-control flower rollout twice with matching inputs, predictions and commands; both trials failed. [Closed-loop repeat summary](flower-ppo-msaa0/summary.json)

The final fruit dataset and all compared policies use `render_samples=0`. The original successful flower repeat is not counted as a result of the corrected final pipeline. The official physics, controller and task-success conditions are unchanged. The current exporter and rollout loader record the rendering profile and reject a mismatched dataset profile.

The retained summaries document the finding. Large per-control dumps and one-off probe sources are omitted from the current source release. [Current data pipeline](../../../stage2.md) and [final comparison](../fruit-completion-v1/README.md) describe the supported path.
