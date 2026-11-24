"""Independently compare saved PPO actor tensors against their exact SFT parent."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    root = Path("runs/fruit-full-ppo-reward-compare-20261124")
    parent = Path("runs/fruit-full-train47-20261124/storage/sft/sft.pt")
    arm = sys.argv[1]
    if arm not in ("potential", "milestone"):
        raise ValueError("Use an existing reward arm")
    assert (root / f"{arm}-train.exit-code").read_text().strip() == "0"
    destination = root / f"{arm}-saved-actor-audit.json"
    assert not destination.exists()
    torch.set_num_threads(2)
    original = torch.load(parent, map_location="cpu", weights_only=True, mmap=True)
    parent_sha = sha(parent)
    assert original["metadata"]["train_full_action_expert"]
    assert original["metadata"]["trainable_parameter_count"] == 430098464
    path = root / "storage" / arm
    updates = [json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (path / "checkpoint-audits.jsonl").read_text().splitlines()]
    reports = []
    for iteration in (1, 2):
        checkpoint = path / f"ppo-{iteration}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        report = saved["ppo"]
        assert report["sft_sha256"] == parent_sha
        assert report["optimizer_iterations"] == iteration
        assert report["norm_stats_sha256"] == original["metadata"]["norm_stats_sha256"]
        assert report["base_sha256"] == original["metadata"]["base_sha256"]
        assert saved["metadata"] == original["metadata"]
        actor_names = set(original["policy"])
        assert actor_names <= set(saved["policy"])
        assert all(name.startswith("value_head.") for name in set(saved["policy"]) - actor_names)
        groups = {}
        for name, before in original["policy"].items():
            after = saved["policy"][name]
            assert before.dtype == after.dtype and before.shape == after.shape
            assert torch.isfinite(before).all() and torch.isfinite(after).all()
            delta = after.double() - before.double()
            role = "action_expert_full" if name.startswith("llm.") else "action_heads"
            group = groups.setdefault(role, dict(tensors=0, changed_tensors=0, changed_elements=0, max_abs_delta=0., delta_squared=0.))
            changed = int(torch.count_nonzero(delta))
            group["tensors"] += 1
            group["changed_tensors"] += int(changed > 0)
            group["changed_elements"] += changed
            group["max_abs_delta"] = max(group["max_abs_delta"], float(delta.abs().max()))
            group["delta_squared"] += float(delta.square().sum())
        for role, group in groups.items():
            group["delta_l2"] = math.sqrt(group.pop("delta_squared"))
            prior = report["actor_parameter_changes"][role]
            for key in ("tensors", "changed_tensors", "changed_elements"):
                assert group[key] == prior[key], (role, key)
            for key in ("max_abs_delta", "delta_l2"):
                assert math.isclose(group[key], prior[key], rel_tol=2e-6, abs_tol=1e-12), (role, key)
        applied = [row for row in updates if row["iteration"] <= iteration and row["optimizer_step_applied"]]
        actor = [row for row in applied if row["phase"] == "ppo" and row["actor_max_abs_delta_this_step"] > 0]
        current = [row for row in actor if row["iteration"] == iteration]
        assert len(current) == report["actor_optimizer_steps"] > 0
        assert len(applied) == report["total_optimizer_steps"]
        assert max(row["actor_grad_l2"] for row in actor) > 0
        assert sum(row["steps"] for row in rollouts if row["iteration"] <= iteration) == report["total_controls"]
        receipt, = [row for row in receipts if row["iteration"] == iteration]
        checkpoint_sha = sha(checkpoint)
        assert receipt["checkpoint_sha256"] == checkpoint_sha
        assert receipt["actor_and_critic_tensors_equal"]
        assert receipt["reload_normalized_prediction_max_error"] == 0
        assert "optimizer" not in saved and not report["training_resume_supported"]
        reports.append(dict(iteration=iteration, checkpoint_sha256=checkpoint_sha, actor_parameter_changes=groups, cumulative_actor_optimizer_steps=len(actor), iteration_actor_optimizer_steps=len(current), cumulative_critic_optimizer_steps=len(applied), actor_grad_l2_max=max(row["actor_grad_l2"] for row in actor), total_controls=report["total_controls"], optimizer_counters_independently_reconstructed=False, optimizer_evidence="Applied updates and gradients from logs; inference-onlycheckpoint has no optimizer state", matches_training_report=True))
    destination.write_text(json.dumps(dict(complete=True, audited_at=datetime.now(timezone.utc).isoformat(), sft_sha256=parent_sha, audit_source_sha256=sha(__file__), arm=arm, checkpoints=reports), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=True, arm=arm, checkpoints=len(reports), saved_actor_tensors_verified=True)), flush=True)


if __name__ == "__main__":
    main()
