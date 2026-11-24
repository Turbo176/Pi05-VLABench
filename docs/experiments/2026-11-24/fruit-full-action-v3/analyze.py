"""Verify the exported fruit comparison and plot its complete training histories."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

MANIFEST_SHA = "8d62b9aaee8fd018143cd82efbfeb8c301472adf4ab6e0c730a89cebf5e1da78"
ORIGINAL_NORM = "bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(wins, count):
    z = 1.959963984540054
    p = wins / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def score(records):
    wins = sum(row["success"] for row in records)
    return dict(successes=wins, trials=len(records), success_rate=wins/len(records), wilson_95=wilson(wins, len(records)), controls=sum(row["steps"] for row in records), ik_nonconverged_controls=sum(row["ik_nonconverged_steps"] for row in records), failures=dict(Counter(row["failure_type"] for row in records if not row["success"])))


def target(row):
    match = re.fullmatch(r"Put the (apple|banana|orange|pear) into the plate_seen", row["instruction"])
    require(match is not None, "Unexpected official instruction")
    return match.group(1)


def paired(first, second):
    gained = sum(not a["success"] and b["success"] for a,b in zip(first, second))
    lost = sum(a["success"] and not b["success"] for a,b in zip(first, second))
    discordant = gained + lost
    probability = min(1., 2*sum(math.comb(discordant, k) for k in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    return dict(gained_scenes=gained, lost_scenes=lost, unchanged_scenes=len(first)-discordant, success_rate_difference=(gained-lost)/len(first), exact_mcnemar_two_sided_p=probability)


def verify(evidence):
    experiment=evidence.parent.resolve()
    exported=read(evidence/"export-manifest.json")
    for relative,expected_file in exported["files"].items():
        path=(experiment/relative).resolve()
        path.relative_to(experiment)
        require(path.stat().st_size==expected_file["size"] and sha(path)==expected_file["sha256"], "Export hash mismatch: "+relative)
    manifest = rows(evidence / "test/cases.jsonl")
    require(sha(evidence / "test/cases.jsonl") == MANIFEST_SHA, "Reserved test manifest changed")
    expected = {row["seed"]:row for row in manifest}
    require(len(manifest) == len(expected) == 40, "Require forty unique seeds")
    require(len({row["initial_state_sha256"] for row in manifest}) == 40, "Duplicate test initialization")
    require(Counter(map(target, manifest)) == Counter(dict.fromkeys(("apple","banana","orange","pear"),10)), "Unbalanced test targets")
    lock = read(evidence / "test/model-lock.json")
    selection = read(evidence / "development/selection.json")
    require(selection["complete"] and lock["selection_used_only_development"] and lock["test_policy_calls_so_far"] == 0, "Incomplete or post-test selection")
    require(lock["upstream_reference"]["normalization_sha256"] == ORIGINAL_NORM, "Reference used modified normalization")
    require(selection["selected_candidate"]["sha256"] == lock["ppo"]["sha256"], "Selected PPO identity changed")
    ordered = sorted(expected)
    stages = {}
    for label, stage in (("reference","base"),("sft","sft"),("ppo","ppo")):
        records = rows(evidence / f"test/{label}.jsonl")
        indexed = {row["seed"]:row for row in records}
        require(len(records) == 40 and set(indexed) == set(expected), f"Incomplete {label} outcomes")
        for seed,row in indexed.items():
            require(all(row[key] == expected[seed][key] for key in ("initial_state_sha256","initial_policy_rgb_sha256","instruction","render_samples")), f"Unpaired {label} seed {seed}")
            require(row["stage"] == stage and row["control_hz"] == 10 and row["replan_steps"] == 5 and row["max_control_steps"] == 600, "Control contract changed")
            require(row["backend"] == "RLinf" and row["sampler"] == "JAX seeded noise", "Sampler or backend changed")
            require(0 < row["steps"] <= 600 and type(row["success"]) is bool, "Invalid outcome")
            require(row["base_sha256"] == lock["upstream_reference"]["base_sha256"], "Base weights changed")
            if label == "reference":
                require(row["norm_stats_sha256"] == ORIGINAL_NORM and "sft_sha256" not in row, "Reference restored trained weights or used new statistics")
            else:
                require(row["sft_sha256"] == lock["sft"]["sha256"] and row["norm_stats_sha256"] == lock["sft"]["normalization_sha256"], "SFT identity or norm changed")
            if label == "ppo":
                require(row["ppo_sha256"] == lock["ppo"]["sha256"], "Test PPO differs from model lock")
        stages[label] = [indexed[seed] for seed in ordered]
    handoff = read(evidence / "handoff/development.json")
    require(handoff["complete"] and handoff["optimizer_updates"] == 0 and handoff["matched_rollouts"] == 16 and all(row["matched"] for row in handoff["matches"]), "Zero-update handoff did not match all scenes")
    require(handoff["source_sft_sha256"] == lock["sft"]["sha256"], "Handoff used another SFT")
    sft = read(evidence / "sft/summary.json")
    sft_updates = rows(evidence / "sft/updates.jsonl")
    require(sft["updates"] == len(sft_updates) == 3000 and [row["update"] for row in sft_updates] == list(range(1,3001)), "Incomplete SFT updates")
    require(sft["reload_max_action_difference"] == 0 and sft["trainable_parameter_count"] == 430098464 and sft["train_full_action_expert"], "SFT trainable scope or restoration changed")
    gate = read(evidence / "sft/seen-gate.json")
    require(gate["complete"] and gate["passed"] and gate["successes"] == 3, "Seen reset gate failed")
    ppo_plan=read(evidence / "ppo-plan.json")
    amendment = ppo_plan["parent_qualification_amendment"]
    require(not amendment["prior_strict_success_count_upgrade_gate_passed"] and amendment["new_ppo_updates_at_amendment"] == amendment["reserved_test_policy_calls_at_amendment"] == 0, "Parent qualification record changed")
    for relative,digest in ppo_plan["source_sha256"].items():
        require(sha(experiment/"implementation/rl/source"/Path(relative).name)==digest,"Registered PPO source differs: "+relative)
    sft_plan=read(evidence/"sft/plan.json")
    for relative,digest in sft_plan["source_sha256"].items():
        require(sha(experiment/"implementation/sft"/Path(relative).name)==digest,"Registered SFT source differs: "+relative)
    require(sha(evidence/"sft/norm_stats.json")==lock["sft"]["normalization_sha256"],"Training normalization export differs")
    training = {}
    for arm in ("potential","milestone"):
        path = evidence / arm
        updates = rows(path / "updates.jsonl")
        rollouts = rows(path / "rollouts.jsonl")
        warmup = read(path / "warmup.json")
        independent = read(path / "saved-actor-audit.json")
        receipts = rows(path / "checkpoint-audits.jsonl")
        require(warmup["actor_unchanged"] and warmup["actor_max_abs_delta"] == 0 and warmup["critic_max_abs_delta"] > 0, "Warmup isolation failed")
        require(independent["complete"] and independent["sft_sha256"] == lock["sft"]["sha256"], "Saved actor audit missing")
        require(len(rollouts) == 12, "Require four warmup and eight PPO rollouts per reward arm")
        all_actor = [row for row in updates if row["phase"] == "ppo" and row["optimizer_step_applied"] and row["actor_max_abs_delta_this_step"] > 0]
        require(all_actor and max(row["actor_grad_l2"] for row in all_actor) > 0, "No real actor updates")
        per_iteration = []
        for iteration in (1,2):
            report = read(path / f"iteration-{iteration}.json")
            audits=[row for row in independent["checkpoints"] if row["iteration"]==iteration]
            if not audits:
                require(arm=="milestone" and iteration==2 and report["postupdate_full_batch_prefix_approx_kl"]>.02,"Unexpected missing saved checkpoint")
                require(not any(row["iteration"]==iteration for row in receipts),"Rejected iteration has a saved checkpoint receipt")
                outcome=read(evidence/"training-outcome.json")
                require(outcome["milestone_iteration2"]["rejected_before_save"] and outcome["no_guard_relaxation"],"Missing recorded rejection")
                require(report["total_controls"]==sum(row["steps"] for row in rollouts),"Rejected iteration cost omitted")
                per_iteration.append(dict(iteration=iteration,saved=False,rejected_for_full_batch_score=True,full_batch_prefix_score=report["postupdate_full_batch_prefix_approx_kl"],controls=report["total_controls"],actor_optimizer_steps=report["actor_optimizer_steps"]))
                continue
            audit, = [row for row in independent["checkpoints"] if row["iteration"] == iteration]
            receipt, = [row for row in receipts if row["iteration"] == iteration]
            applied = [row for row in updates if row["iteration"] <= iteration and row["optimizer_step_applied"]]
            actor = [row for row in all_actor if row["iteration"] <= iteration]
            current = [row for row in actor if row["iteration"] == iteration]
            require(report["sft_sha256"] == lock["sft"]["sha256"] and report["norm_stats_sha256"] == lock["sft"]["normalization_sha256"], "PPO parent changed")
            require(report["actor_optimizer_steps"] == len(current) > 0 and report["total_optimizer_steps"] == len(applied), "Optimizer log counts disagree")
            require(audit["cumulative_actor_optimizer_steps"] == len(actor) and audit["cumulative_critic_optimizer_steps"] == len(applied) and audit["matches_training_report"], "Independent audit disagrees")
            require(audit["checkpoint_sha256"] == receipt["checkpoint_sha256"] and receipt["reload_normalized_prediction_max_error"] == 0 and receipt["actor_and_critic_tensors_equal"], "Saved checkpoint reload failed")
            require(0 <= report["postupdate_full_batch_prefix_approx_kl"] <= .02, "Full-batch score guard exceeded")
            require(report["total_controls"] == sum(row["steps"] for row in rollouts if row["iteration"] <= iteration), "Interaction count differs")
            require(report["actor_parameter_changes"] == audit["actor_parameter_changes"] or all(all(math.isclose(report["actor_parameter_changes"][role][key], audit["actor_parameter_changes"][role][key], rel_tol=2e-6, abs_tol=1e-12) for key in report["actor_parameter_changes"][role]) for role in report["actor_parameter_changes"]), "Saved actor changes disagree")
            per_iteration.append(dict(iteration=iteration,saved=True, actor_optimizer_steps=len(current), cumulative_actor_optimizer_steps=len(actor), controls=report["total_controls"], full_batch_prefix_score=report["postupdate_full_batch_prefix_approx_kl"], checkpoint_sha256=audit["checkpoint_sha256"]))
        training[arm] = dict(iterations=per_iteration, total_controls=sum(row["steps"] for row in rollouts), actor_optimizer_steps=len(all_actor), critic_optimizer_steps=sum(row["optimizer_step_applied"] for row in updates), wall_seconds=read(path / "iteration-2.json")["wall_seconds"], checkpoint_training_resume_supported=False)
    selected = selection["selected_candidate"]
    selected_audit, = [row for row in training[selected["arm"]]["iterations"] if row["iteration"] == selected["iteration"]]
    require(selected_audit["checkpoint_sha256"] == selected["sha256"], "Selected checkpoint audit differs")
    starts = rows(evidence / "training-starts.jsonl")
    development = rows(evidence / "development/cases.jsonl")
    require(not {row["initial_state_sha256"] for row in starts} & {row["initial_state_sha256"] for row in manifest+development}, "PPO training initialization leakage")
    return dict(complete=True, verified_export_files=len(exported["files"]), test_manifest_sha256=MANIFEST_SHA, test=[dict(policy=label, **score(records), by_fruit={fruit:score([row for row in records if target(row) == fruit]) for fruit in ("apple","banana","orange","pear")}) for label,records in stages.items()], paired=dict(sft_minus_reference=paired(stages["reference"],stages["sft"]), ppo_minus_sft=paired(stages["sft"],stages["ppo"])), selected_ppo=selected, development_successes=selection["selected_development_successes"], development_trials=16, sft_training_seconds=sft["training_seconds"], ppo_training=training, total_reward_comparison_controls=sum(row["total_controls"] for row in training.values()), zero_update_matched_scenes=16, source_of_optimizer_steps="Logs and saved actor deltas; PPO inference checkpoints omit optimizer state"), stages


def plot(evidence, output, report, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white", "axes.titleweight":"bold"})
    labels = ("Upstream reference","Fruit SFT3000","PPO after SFT3000")
    colors = ("#64748b","#2563eb","#0d9488")
    fig, axes = plt.subplots(1,2,figsize=(12,4.7), gridspec_kw={"width_ratios":[1.1,1.5]})
    for i,entry in enumerate(report["test"]):
        rate=entry["success_rate"];lo,hi=entry["wilson_95"]
        axes[0].bar(i,rate*100,color=colors[i],width=.63)
        axes[0].errorbar(i,rate*100,yerr=np.array([[rate-lo],[hi-rate]])*100,color="#1e293b",capsize=5)
        axes[0].text(i,hi*100+3,f"{entry['successes']}/40",ha="center",weight="bold")
    axes[0].set(ylim=(0,105),ylabel="Official success (%)",xticks=range(3),xticklabels=("Reference","SFT","PPO"),title="40 paired scenes · Wilson 95% intervals")
    fruit_names=("apple","banana","orange","pear");x=np.arange(4)
    for i,entry in enumerate(report["test"]):
        values=[entry["by_fruit"][fruit]["successes"] for fruit in fruit_names]
        bars=axes[1].bar(x+(i-1)*.24,values,width=.23,label=labels[i],color=colors[i])
        axes[1].bar_label(bars,padding=2)
    axes[1].set(xticks=x,xticklabels=[name.title() for name in fruit_names],ylim=(0,12),ylabel="Official successes / 10",title="Same natural resets and 600-control budget")
    axes[1].legend(frameon=False,loc="upper right",fontsize=8)
    fig.tight_layout();fig.savefig(output/"success-rates.png",dpi=180);fig.savefig(output/"success-rates.svg");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    sft=rows(evidence/"sft/updates.jsonl");loss=np.array([row["loss"] for row in sft]);x=np.arange(1,len(loss)+1)
    axes[0,0].scatter(x,loss,s=3,alpha=.25,color="#2563eb",label="Every minibatch")
    median=[np.median(loss[max(0,i-99):i+1]) for i in range(len(loss))]
    axes[0,0].plot(x,median,color="#1e3a8a",linewidth=1.5,label="Trailing 100-update median")
    axes[0,0].set(yscale="log",xlabel="SFT optimizer update",ylabel="Flow-matching loss (log scale)",title="All 3,000 SFT updates")
    axes[0,0].legend(frameon=False,fontsize=8)
    for arm,color in (("potential","#7c3aed"),("milestone","#0d9488")):
        updates=rows(evidence/arm/"updates.jsonl");warmup=read(evidence/arm/"warmup.json")
        indices=np.arange(1,len(updates)+1)
        axes[0,1].plot(indices,[row["value_loss"] for row in updates],color=color,linewidth=.9,label=arm.title())
        actor=[(i,row) for i,row in zip(indices,updates) if row["phase"]=="ppo"]
        axes[1,0].plot([i for i,_ in actor],[row["actor_loss"] for _,row in actor],color=color,marker=".",linewidth=.8,label=arm.title())
        attempted=[(i,row) for i,row in actor if row.get("postupdate_approx_kl") is not None]
        axes[1,1].plot([i for i,_ in attempted],[row["postupdate_approx_kl"] for _,row in attempted],color=color,marker=".",linewidth=.8,label=arm.title())
        rejected=[(i,row) for i,row in actor if row["stopped_for_kl"]]
        for i,row in rejected:
            val=row.get("postupdate_approx_kl")
            if val is None:val=row["approx_kl"]
            axes[1,1].scatter(i,val,marker="x",color=color,s=45)
        for iteration in (1,2):
            summary=read(evidence/arm/f"iteration-{iteration}.json")
            axes[1,1].scatter(summary["total_optimizer_steps"],summary["postupdate_full_batch_prefix_approx_kl"],marker="D",facecolors="none",edgecolors=color,s=42,label="Whole rollout" if arm=="potential" and iteration==1 else None)
    axes[0,1].set(yscale="log",xlabel="Optimizer attempt (critic warmup included)",ylabel="Value loss (log scale)",title="Critic fitting and PPO")
    axes[1,0].set(xlabel="Optimizer attempt",ylabel="Clipped actor objective",title="PPO actor losses; both complete histories")
    axes[1,1].axhline(.02,color="#dc2626",linestyle="--",label="0.02 guard")
    axes[1,1].set(xlabel="Optimizer attempt",ylabel="Sampled prefix-score diagnostic",title="Update guard: minibatches and whole rollouts")
    for ax in axes.flat:ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(output/"training.png",dpi=180);plt.close(fig)
    ordered=sorted(stages["reference"],key=lambda row:row["seed"])
    image=np.array([[int(row["success"]) for row in stages[label]] for label in ("reference","sft","ppo")])
    fig,ax=plt.subplots(figsize=(14,3.4))
    from matplotlib.colors import ListedColormap
    ax.imshow(image,cmap=ListedColormap(("#e2e8f0","#0d9488")),vmin=0,vmax=1,aspect="auto")
    ax.set(yticks=range(3),yticklabels=labels,xticks=range(40),xticklabels=[row["seed"] for row in ordered],title="Every paired test outcome · green = official success, gray = failure")
    ax.tick_params(axis="x",labelrotation=90,labelsize=7)
    fig.tight_layout();fig.savefig(output/"paired-scenes.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    report,stages=verify(args.evidence)
    destination=args.output/"comparison.json"
    if args.verify_only:
        require(read(destination)==report,"Saved comparison differs from raw evidence")
    else:
        args.output.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        plot(args.evidence,args.output,report,stages)
    print(json.dumps(dict(verified=True,episodes=120,successes={row["policy"]:row["successes"] for row in report["test"]},reward_study_controls=report["total_reward_comparison_controls"])),flush=True)


if __name__=="__main__":
    main()
