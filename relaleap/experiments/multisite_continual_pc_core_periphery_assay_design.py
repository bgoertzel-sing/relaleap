"""Design a fail-closed multi-site continual PC/core-periphery assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/orthogonalized_sparse_core_periphery_interference_closeout/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/multisite_continual_pc_core_periphery_assay_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_evidence.csv",
    "site_schedule.csv",
    "assay_arms.csv",
    "observable_contract.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_ARMS = (
    "multisite_pc_core_periphery_candidate",
    "shared_core_only_ablation",
    "plastic_periphery_only_ablation",
    "equal_plasticity_core_periphery_ablation",
    "random_core_periphery_assignment_null",
    "dense_rank_norm_residual_control",
    "parameter_matched_mlp_residual_control",
    "low_rank_residual_control",
    "random_support_sparse_control",
    "frequency_support_sparse_control",
    "token_position_only_router_null",
    "shuffled_site_target_null",
)

REQUIRED_OBSERVABLES = (
    "heldout_ce_guardrail",
    "site_transfer_ce",
    "cross_site_retention",
    "anchor_kl_drift",
    "functional_flip_churn",
    "finite_update_commutator",
    "causal_intervention_fingerprint",
    "core_genericity_score",
    "periphery_specificity_score",
    "periphery_first_pruning_delta",
    "residual_l2_budget",
    "active_and_stored_parameter_budget",
    "leakage_null_rejection",
)

REQUIRED_SITES = ("copy", "reverse", "permute", "negate", "copy_revisit")


def run_multisite_continual_pc_core_periphery_assay_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write the local design contract for the next multi-site assay."""

    start = time.time()
    closeout = _read_json(closeout_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = _source_rows(closeout_path, closeout, strategy_review_path, strategy)
    site_rows = _site_schedule()
    arm_rows = _assay_arms()
    observable_rows = _observable_contract()
    gate_rows = _gate_rows(closeout, site_rows, arm_rows, observable_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "multisite_continual_pc_core_periphery_assay_design_recorded"
            if status == "pass"
            else "multisite_continual_pc_core_periphery_assay_design_failed_closed"
        ),
        "scientific_gate": (
            "ready_for_local_multisite_pc_core_periphery_assay_implementation"
            if status == "pass"
            else "blocked"
        ),
        "claim_status": "design_contract_only_no_training_gpu_or_promotion_evidence",
        "selected_next_step": (
            "implement the bounded local multi-site continual PC/core-periphery assay with matched dense/MLP/null controls"
            if status == "pass"
            else "repair source closeout or missing assay contract rows before implementation"
        ),
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "backend_policy": "local design/report only; RunPod and Colab remain blocked until a command-driven local assay exists and passes gates",
        "source_evidence": source_rows,
        "site_schedule": site_rows,
        "assay_arms": arm_rows,
        "observable_contract": observable_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "interpretation": _interpretation(status),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    closeout_path: Path,
    closeout: dict[str, Any],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "orthogonalized_sparse_core_periphery_interference_closeout",
            "path": str(closeout_path),
            "present": bool(closeout),
            "status": closeout.get("status", "missing"),
            "decision": closeout.get("decision", ""),
            "claim_status": closeout.get("claim_status", ""),
            "selected_next_action": closeout.get("selected_next_action", ""),
            "advance_to_gpu_validation": closeout.get("advance_to_gpu_validation", ""),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
            "selected_next_action": "",
            "advance_to_gpu_validation": False,
        },
    ]


def _site_schedule() -> list[dict[str, Any]]:
    rows = []
    for index, site in enumerate(REQUIRED_SITES, start=1):
        rows.append(
            {
                "site_index": index,
                "site": site,
                "phase_role": "revisit_retention_probe" if site.endswith("_revisit") else "adaptation_site",
                "target_family": "synthetic_prefix_safe_rule_stream",
                "task_id_visible_to_model": False,
                "shared_vocab_and_head": True,
                "update_target": "local_pc_hidden_delta_plus_ce_guardrail",
                "anchor_evaluation": "all earlier sites plus frozen anchor contexts",
                "required": True,
            }
        )
    return rows


def _assay_arms() -> list[dict[str, Any]]:
    specs = {
        "multisite_pc_core_periphery_candidate": ("candidate", "shared protected core plus plastic per-site periphery with PC hidden-delta updates"),
        "shared_core_only_ablation": ("mechanism_ablation", "remove plastic periphery while matching active rank where possible"),
        "plastic_periphery_only_ablation": ("mechanism_ablation", "remove protected shared core and keep periphery budget"),
        "equal_plasticity_core_periphery_ablation": ("mechanism_ablation", "same optimizer and consolidation for core and periphery"),
        "random_core_periphery_assignment_null": ("mechanism_null", "shuffle core/periphery assignment after training"),
        "dense_rank_norm_residual_control": ("dense_control", "dense residual matched on rank, norm, active compute, and storage"),
        "parameter_matched_mlp_residual_control": ("mlp_control", "causal-input MLP residual matched on parameters and active compute"),
        "low_rank_residual_control": ("dense_control", "low-rank adapter matched to active sparse rank"),
        "random_support_sparse_control": ("support_null", "random top-k sparse support with matched active/storage budgets"),
        "frequency_support_sparse_control": ("support_null", "train-frequency support sparse control"),
        "token_position_only_router_null": ("leakage_null", "router restricted to token and position features"),
        "shuffled_site_target_null": ("leakage_null", "misaligned site target/control updates to reject accounting artifacts"),
    }
    return [
        {
            "arm": arm,
            "family": family,
            "role": role,
            "matched_dimensions": "site_order, train_examples, heldout_examples, residual_l2, active_params, stored_params, optimizer_steps, seed",
            "required_outputs": "arm_metrics; phase_metrics; intervention_fingerprints; pruning_audit; commutator_matrix",
            "required": True,
        }
        for arm, (family, role) in specs.items()
    ]


def _observable_contract() -> list[dict[str, Any]]:
    specs = {
        "heldout_ce_guardrail": ("quality", "candidate must remain inside dense/MLP CE tolerance before mechanism claims"),
        "site_transfer_ce": ("transfer", "CE on heldout contexts from each site after later-site adaptation"),
        "cross_site_retention": ("retention", "retained performance on previous sites after sequential updates"),
        "anchor_kl_drift": ("retention", "KL drift on frozen anchor logits after each site update"),
        "functional_flip_churn": ("churn", "prediction flip fraction not explained by CE improvement"),
        "finite_update_commutator": ("interference", "order sensitivity across at least two site-update pairs"),
        "causal_intervention_fingerprint": ("causal", "necessity/sufficiency/selectivity under core/periphery interventions"),
        "core_genericity_score": ("mechanism", "core utility averaged across sites after periphery pruning"),
        "periphery_specificity_score": ("mechanism", "periphery utility concentrated on its originating site"),
        "periphery_first_pruning_delta": ("mechanism", "periphery-first pruning should hurt less off-site than core-first pruning"),
        "residual_l2_budget": ("budget", "candidate and controls compared under residual norm bands"),
        "active_and_stored_parameter_budget": ("budget", "active/storage budget recorded for every arm"),
        "leakage_null_rejection": ("validity", "token-position, random/frequency, and shuffled-target nulls must be rejected"),
    }
    return [
        {
            "observable": name,
            "family": family,
            "requirement": requirement,
            "required": True,
        }
        for name, (family, requirement) in specs.items()
    ]


def _gate_rows(
    closeout: dict[str, Any],
    site_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arms = {row["arm"] for row in arm_rows if row.get("required") is True}
    observables = {row["observable"] for row in observable_rows if row.get("required") is True}
    sites = {row["site"] for row in site_rows if row.get("required") is True}
    return [
        _criterion(
            "closeout_selected_multisite_design",
            closeout.get("status") == "pass"
            and closeout.get("selected_next_action") == "design_multisite_continual_pc_core_periphery_assay_before_gpu"
            and closeout.get("advance_to_gpu_validation") is False,
            "hard",
            "one-site closeout must explicitly select multi-site local design and block GPU",
            {
                "status": closeout.get("status", "missing"),
                "selected_next_action": closeout.get("selected_next_action", ""),
                "advance_to_gpu_validation": closeout.get("advance_to_gpu_validation", ""),
            },
            "regenerate the one-site closeout before designing the multi-site assay",
        ),
        _criterion(
            "site_schedule_has_revisit_and_hidden_boundaries",
            set(REQUIRED_SITES).issubset(sites)
            and any(row["phase_role"] == "revisit_retention_probe" for row in site_rows)
            and all(row["task_id_visible_to_model"] is False for row in site_rows),
            "hard",
            "assay must include at least four hidden sites plus a revisit retention phase",
            sorted(sites),
            "add the required hidden site schedule",
        ),
        _criterion(
            "assay_arms_complete",
            set(REQUIRED_ARMS).issubset(arms),
            "hard",
            "candidate, dense/MLP controls, sparse/null controls, and mechanism ablations are preregistered",
            sorted(arms),
            "add missing mandatory assay arms",
        ),
        _criterion(
            "observable_contract_complete",
            set(REQUIRED_OBSERVABLES).issubset(observables),
            "hard",
            "retention, commutator, causal fingerprint, pruning, CE, budget, and leakage-null observables are preregistered",
            sorted(observables),
            "add missing mandatory observables",
        ),
        _criterion(
            "local_only_before_gpu",
            True,
            "hard",
            "RunPod/Colab validation remains blocked until local assay rows exist and pass gates",
            "requires_gpu_now=false",
            "do not run GPU validation from this design-only step",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    expected: Any,
    actual: Any,
    failure_action: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "failure_action": "" if passed else failure_action,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in fields:
                fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields.get("notify_ben")).lower() == "true"
        or fields.get("strategic_change_level") == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_review_read": strategy["present"],
        "accepted": True,
        "deferred_or_rejected": "",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "reason": (
            "The latest review's trained-pilot recommendation has been completed and the resulting closeout selects the radical multi-site local fallback. "
            "This design records that direction without promoting or sending the branch to GPU."
        ),
    }


def _interpretation(status: str) -> str:
    if status == "pass":
        return (
            "This artifact is a design contract only. It redirects the failed one-site sparse core/periphery branch into a local multi-site continual-learning assay "
            "where forgetting, finite-update commutators, causal fingerprints, pruning selectivity, and dense/MLP/null controls are primary."
        )
    return "The multi-site assay design is incomplete; implementation and GPU validation remain blocked."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_evidence.csv", summary["source_evidence"])
    _write_csv(out_dir / "site_schedule.csv", summary["site_schedule"])
    _write_csv(out_dir / "assay_arms.csv", summary["assay_arms"])
    _write_csv(out_dir / "observable_contract.csv", summary["observable_contract"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-Site Continual PC/Core-Periphery Assay Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        "",
        summary["interpretation"],
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    if summary["strategy_review"].get("ben_notification_required"):
        lines.extend(["", "Ben notification is required by the strategy review header."])
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_multisite_continual_pc_core_periphery_assay_design(
        closeout_path=args.closeout,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
