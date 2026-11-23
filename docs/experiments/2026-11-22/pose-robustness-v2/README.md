# Pose dataset checks and shared-policy development

This directory retains the checks needed to interpret the final fruit experiment. Earlier joint-space pilots, one-episode fits and alternate recipes are separate diagnostics; their checkpoints do not initialize the final reported PPO model.

## Data and controller alignment

The [training contract audit](training-contract-audit.json) checks the real pose labels through normalization and shared controller decoding. It records 47 fruit episodes / 6,986 frames and 17 flower episodes / 2,918 frames, including literal instruction counts. Five validation episodes / 744 frames remain separate. The final recipe uses original Euler labels and upstream state-conditioning settings.

The [two-update GPU smoke check](two-task-smoke.json) demonstrated backward updates, saved checkpoints and an exact restored action prediction on the shared data. That earlier smoke recipe enabled VLM adapters and image augmentation; the final SFT90 recipe freezes the VLM and disables augmentation. [Final executed recipe](../../../single-gpu.md)

## Fixed development starts

The published reference and final shared SFT90 were evaluated on the same four starts for each retained task.

| Task | Published reference | Shared SFT90 |
| --- | ---: | ---: |
| Fruit selection | 2/4 | 3/4 |
| Flower insertion | 1/4 | 0/4 |

[Reference raw episodes](base-development.jsonl) also include the separate fridge diagnostic. [Shared SFT raw episodes](two-task-conservative-development.jsonl) retain all eight fruit/flower outcomes. [Paired summary](two-task-conservative-development-summary.json) records failure types, intervals and changed outcomes. None of these development results establish held-out improvement.

Flower did not pass the shared-policy completion gate. The finished scope therefore uses fruit PPO from the same shared SFT90, with separate reserved starts for its final evaluation. [Final experiment, all ordinary outcomes and language tests](../fruit-completion-v1/README.md)

## Retained conclusions

Expert action replay verifies data/control alignment; it does not prove that a trained policy can imitate the expert. More optimizer steps and smaller offline action errors did not reliably solve the autonomous failures in the earlier diagnostics. The current records keep this limitation explicit. [Renderer repeatability](../reproducibility/README.md) and [command-motion analysis](../../2026-11-23/training-and-motion/README.md) document two concrete integration checks.

One-off schedulers, interrupted worker logs, duplicate source/config snapshots and superseded development runs are omitted from this source release. The public Git history and local experiment archive retain the previous full export. The final evaluation records include every scored success and failure.
