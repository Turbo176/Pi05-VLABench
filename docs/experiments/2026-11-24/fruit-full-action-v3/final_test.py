"""Execute the locked three-policy comparison once on the reserved forty resets."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("runs/fruit-full-ppo-reward-compare-20261124")
RESERVED = Path("runs/fruit-long-sft-20261124/reserved-test")
PARENT = Path("runs/fruit-full-train47-20261124/storage/sft")
REFERENCE_CONFIG = Path("runs/fruit-recovery-20261124/sft-continuation-360/checkpoint-0180")
BASE = Path(".local/checkpoints/vlabench-primitive-rlinf/model.safetensors")
EVALUATOR = ROOT / "source/evaluate_pose_development.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    lock_path = ROOT / "evaluation/final-test-model-lock.json"
    lock = json.loads(lock_path.read_text())
    selection = json.loads((ROOT / "evaluation/selection.json").read_text())
    assert selection["complete"] and lock["selection_used_only_development"]
    assert lock["test_policy_calls_so_far"] == 0
    assert (ROOT / "evaluate.exit-code").read_text().strip() == "0"
    assert sha(RESERVED / "select_fruit-official.jsonl") == lock["reserved_test_manifest_sha256"]
    assert sha(BASE) == lock["upstream_reference"]["base_sha256"]
    assert sha(REFERENCE_CONFIG / "norm_stats.json") == lock["upstream_reference"]["normalization_sha256"]
    assert (REFERENCE_CONFIG / "norm_stats.json").read_bytes() == Path(".local/checkpoints/vlabench-primitive-rlinf/norm_stats.json").read_bytes()
    assert sha(PARENT / "sft.pt") == lock["sft"]["sha256"]
    assert sha(PARENT / "norm_stats.json") == lock["sft"]["normalization_sha256"]
    assert sha(lock["ppo"]["checkpoint"]) == lock["ppo"]["sha256"]
    cases = [json.loads(line) for line in (RESERVED / "select_fruit-official.jsonl").read_text().splitlines()]
    assert len(cases) == 40 and len({row["seed"] for row in cases}) == 40
    assert len({row["initial_state_sha256"] for row in cases}) == 40
    plan_path = ROOT / "final-test-execution-plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["evaluator_sha256"] == sha(EVALUATOR)
    assert plan["driver_sha256"] == sha(Path(__file__))
    destination = ROOT / "final-test"
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    save(destination / "launch.json", dict(started_at=datetime.now(timezone.utc).isoformat(), model_lock_sha256=sha(lock_path), execution_plan_sha256=sha(plan_path), models_fixed_before_test=True, max_concurrent_models=2, simulator_workers_per_model=2))
    jobs = [
        ("reference", "base", REFERENCE_CONFIG, None),
        ("sft", "sft", PARENT, None),
        ("ppo", "ppo", PARENT, Path(lock["ppo"]["checkpoint"])),
    ]

    def run(job):
        label, stage, configuration, weights = job
        command = [sys.executable, str(EVALUATOR), "--base", str(BASE), "--fork", os.environ["PI05_REFERENCE_FORK"], "--run", str(configuration), "--cases", str(RESERVED), "--protocol", str(RESERVED / "protocol.json"), "--output", str(destination / label), "--tasks", "select_fruit", "--suite", "ordinary", "--workers", "2", "--stage", stage]
        if weights is not None:
            command.extend(["--ppo-weights", str(weights)])
        save(destination / f"{label}-command.json", dict(label=label, command=command, started_at=datetime.now(timezone.utc).isoformat()))
        with (destination / f"{label}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        (destination / f"{label}.exit-code").write_text(str(result.returncode) + "\n")
        assert result.returncode == 0, f"{label} failed; inspect its log before any retry"
        completed = json.loads((destination / label / "completed.json").read_text())
        rows = [json.loads(line) for line in (destination / label / "episodes.jsonl").read_text().splitlines()]
        assert completed["cases"] == len(rows) == 40
        if label == "reference":
            assert completed["norm_stats_sha256"] == lock["upstream_reference"]["normalization_sha256"]
            assert "sft_sha256" not in completed
        else:
            assert completed["sft_sha256"] == lock["sft"]["sha256"]
        if label == "ppo":
            assert completed["ppo_sha256"] == lock["ppo"]["sha256"]
        report = dict(policy=label, successes=sum(row["success"] for row in rows), trials=40, controls=sum(row["steps"] for row in rows), completed_sha256=sha(destination / label / "completed.json"), episodes_sha256=sha(destination / label / "episodes.jsonl"))
        print(json.dumps(report), flush=True)
        return report

    reports = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            reports.append(future.result())
    save(destination / "completed.json", dict(complete=True, finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started, models_selected_only_on_development=True, reports=reports))


if __name__ == "__main__":
    main()
