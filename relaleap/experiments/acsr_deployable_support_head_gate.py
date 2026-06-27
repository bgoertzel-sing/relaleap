"""Gate deployable support-head evidence after ACSR columnability retirement."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_AUDIT_DIR = Path("results/audits/token_larger_support_wide_promoted_default_exhaustive_support")
DEFAULT_ACSR_GATE_DIR = Path("results/reports/acsr_support_discovery_gate_seed2")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_deployable_support_head_gate_local")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "support_head_metrics.csv",
    "null_controls.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_deployable_support_head_gate(
    *,
    source_audit_dir: Path = DEFAULT_SOURCE_AUDIT_DIR,
    acsr_gate_dir: Path = DEFAULT_ACSR_GATE_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_oracle_ce_headroom: float = 0.01,
    min_holdout_recovery_fraction: float = 0.5,
) -> dict[str, Any]:
    """Write a local deployable support-head gate from command artifacts."""

    start = time.time()
    audit = _read_json(source_audit_dir / "summary.json")
    prior_gate = _read_json(acsr_gate_dir / "summary.json")
    contextual_head = _as_dict(_as_dict(audit.get("audit")).get("contextual_router_support_head"))
    same_student = _as_dict(_as_dict(audit.get("audit")).get("contextual_router_support_intervention"))
    sequence_head = _as_dict(_as_dict(audit.get("audit")).get("contextual_router_support_sequence_head"))
    support_head_rows = _support_head_rows(contextual_head, same_student, sequence_head)
    null_rows = _null_rows(source_audit_dir, prior_gate)
    criteria = _criteria_rows(
        audit=audit,
        prior_gate=prior_gate,
        review=_strategy_review(strategy_review),
        support_head_rows=support_head_rows,
        null_rows=null_rows,
        min_oracle_ce_headroom=min_oracle_ce_headroom,
        min_holdout_recovery_fraction=min_holdout_recovery_fraction,
    )
    hard_failures = [row for row in criteria if not row["passed"] and row["severity"] == "hard"]
    blockers = [row for row in criteria if not row["passed"] and row["severity"] == "claim_blocker"]
    status = "pass" if not hard_failures else "fail"
    deployable_positive = status == "pass" and not blockers
    review = _strategy_review(strategy_review)
    summary = {
        "status": status,
        "decision": (
            "deployable_support_head_gate_positive_ready_for_local_repeat"
            if deployable_positive
            else (
                "deployable_support_head_gate_blocks_claim_pending_nulls_or_headroom"
                if status == "pass"
                else "deployable_support_head_gate_failed_closed"
            )
        ),
        "claim_status": (
            "deployable_support_discovery_local_gate_positive_not_gpu_validated"
            if deployable_positive
            else (
                "deployable_support_discovery_not_established_sparse_identity_retired"
                if status == "pass"
                else "deployable_support_head_gate_not_interpretable"
            )
        ),
        "source_audit_dir": str(source_audit_dir),
        "prior_acsr_gate_dir": str(acsr_gate_dir),
        "support_head_metrics": support_head_rows,
        "null_controls": null_rows,
        "gate_criteria": criteria,
        "failures": hard_failures,
        "claim_blockers": blockers,
        "aggregate_metrics": _aggregate_metrics(prior_gate, support_head_rows),
        "selected_next_step": _selected_next_step(status, deployable_positive),
        "next_command": (
            "./.venv-conda/bin/python -m relaleap.experiments.acsr_deployable_support_head_gate "
            "--out results/reports/acsr_deployable_support_head_gate_local"
        ),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "claim_boundaries": {
            "supported": [
                "same-student fixed-support forcing is represented by router-support intervention artifacts",
                "the local support head can be evaluated as a deployable mechanism probe",
                "sparse-support identity remains retired by the upstream ACSR gate",
            ],
            "not_supported": [
                "deployable support-discovery claim while shuffled-causal-feature support-head null is absent",
                "RunPod validation target while oracle CE headroom remains below the local gate",
                "default-router promotion or sparse-support identity revival",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _support_head_rows(
    contextual_head: dict[str, Any],
    same_student: dict[str, Any],
    sequence_head: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    rows.append(_head_row("learned_contextual_support_head", contextual_head))
    rows.append(_head_row("same_student_oracle_support_forcing", same_student))
    rows.append(_head_row("learned_sequence_support_head", sequence_head))
    return rows


def _head_row(name: str, packet: dict[str, Any]) -> dict[str, Any]:
    holdout = _as_dict(packet.get("holdout"))
    all_split = _as_dict(packet.get("all"))
    return {
        "component": name,
        "present": bool(packet),
        "selector": packet.get("selector", ""),
        "training_objective": packet.get("training_objective", ""),
        "train_split": packet.get("train_split", ""),
        "holdout_split": packet.get("holdout_split", ""),
        "holdout_router_loss": _number(holdout.get("router_loss")),
        "holdout_oracle_loss": _number(holdout.get("oracle_loss")),
        "holdout_intervention_loss": _number(holdout.get("intervention_loss")),
        "holdout_intervention_minus_router_loss": _number(holdout.get("intervention_minus_router_loss")),
        "holdout_intervention_oracle_regret": _number(holdout.get("intervention_oracle_regret")),
        "holdout_oracle_gap_recovery_fraction": _number(holdout.get("oracle_gap_recovery_fraction")),
        "all_oracle_gap_recovery_fraction": _number(all_split.get("oracle_gap_recovery_fraction")),
    }


def _null_rows(source_audit_dir: Path, prior_gate: dict[str, Any]) -> list[dict[str, Any]]:
    prior_nulls = _as_list(prior_gate.get("null_controls"))
    token_position = _prior_null(prior_nulls, "token_position_support_null")
    return [
        {
            "control": "token_position_support_null",
            "present": bool(token_position),
            "source": "prior_acsr_support_discovery_gate",
            "heldout_delta_vs_base_ce": token_position.get("heldout_delta_vs_base_ce", ""),
            "gap_vs_reference_heldout_ce_delta": token_position.get("gap_vs_reference_heldout_ce_delta", ""),
            "interpretation": "upstream ACSR token/position support null",
        },
        {
            "control": "shuffled_causal_feature_support_head_null",
            "present": (source_audit_dir / "shuffled_causal_feature_support_head.csv").is_file(),
            "source": str(source_audit_dir / "shuffled_causal_feature_support_head.csv"),
            "heldout_delta_vs_base_ce": "",
            "gap_vs_reference_heldout_ce_delta": "",
            "interpretation": "required learned support-head null; absent in current source artifacts",
        },
        {
            "control": "same_student_fixed_support_forcing",
            "present": (source_audit_dir / "router_support_intervention.csv").is_file(),
            "source": str(source_audit_dir / "router_support_intervention.csv"),
            "heldout_delta_vs_base_ce": "",
            "gap_vs_reference_heldout_ce_delta": "",
            "interpretation": "forces support choices through the same trained student values",
        },
    ]


def _criteria_rows(
    *,
    audit: dict[str, Any],
    prior_gate: dict[str, Any],
    review: dict[str, Any],
    support_head_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
    min_oracle_ce_headroom: float,
    min_holdout_recovery_fraction: float,
) -> list[dict[str, Any]]:
    learned = _row(support_head_rows, "learned_contextual_support_head")
    same_student = _row(support_head_rows, "same_student_oracle_support_forcing")
    metrics = _aggregate_metrics(prior_gate, support_head_rows)
    return [
        _criterion("strategy_review_consumed", review.get("status") == "read", "hard", "latest strategy review is read", review.get("status", ""), "strategy review was not read"),
        _criterion("source_audit_present", bool(audit), "hard", "support audit summary exists", bool(audit), "support audit summary is missing"),
        _criterion("prior_acsr_gate_present", bool(prior_gate), "hard", "prior ACSR support-discovery gate exists", bool(prior_gate), "prior ACSR gate is missing"),
        _criterion("identity_claim_retired", prior_gate.get("claim_status") == "deployable_support_discovery_not_established_sparse_identity_retired", "hard", "upstream ACSR gate keeps sparse identity retired", prior_gate.get("claim_status", ""), "upstream gate does not record retired sparse identity"),
        _criterion("learned_support_head_present", bool(learned.get("present")), "hard", "learned support-head artifact exists", learned.get("present", False), "learned support-head artifact missing"),
        _criterion("same_student_forcing_present", bool(same_student.get("present")) and _null_present(null_rows, "same_student_fixed_support_forcing"), "claim_blocker", "same-student fixed-support forcing exists", same_student.get("present", False), "same-student support forcing is absent"),
        _criterion("learned_head_holdout_improves_router", _negative(learned.get("holdout_intervention_minus_router_loss")), "claim_blocker", "learned head improves heldout router loss", learned.get("holdout_intervention_minus_router_loss"), "learned support head does not improve heldout router loss"),
        _criterion("learned_head_recovers_oracle_gap", _at_least(learned.get("holdout_oracle_gap_recovery_fraction"), min_holdout_recovery_fraction), "claim_blocker", "learned head recovers enough oracle gap on holdout", learned.get("holdout_oracle_gap_recovery_fraction"), "learned support head recovery is too small"),
        _criterion("oracle_support_headroom_positive", _at_most(metrics.get("sparse_oracle_minus_sparse_default_heldout_ce_delta"), -abs(min_oracle_ce_headroom)), "claim_blocker", "upstream oracle support has nontrivial CE headroom", metrics.get("sparse_oracle_minus_sparse_default_heldout_ce_delta"), "upstream oracle CE headroom is too small for a deployable target"),
        _criterion("token_position_null_present", _null_present(null_rows, "token_position_support_null"), "claim_blocker", "token/position support null exists", _null_present(null_rows, "token_position_support_null"), "token/position support null is absent"),
        _criterion("shuffled_feature_support_head_null_present", _null_present(null_rows, "shuffled_causal_feature_support_head_null"), "claim_blocker", "learned support head has shuffled-causal-feature null", _null_present(null_rows, "shuffled_causal_feature_support_head_null"), "shuffled-causal-feature support-head null is absent"),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _aggregate_metrics(prior_gate: dict[str, Any], support_head_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _as_dict(prior_gate.get("aggregate_metrics")).copy()
    learned = _row(support_head_rows, "learned_contextual_support_head")
    same_student = _row(support_head_rows, "same_student_oracle_support_forcing")
    metrics.update(
        {
            "learned_head_holdout_intervention_minus_router_loss": learned.get("holdout_intervention_minus_router_loss"),
            "learned_head_holdout_oracle_gap_recovery_fraction": learned.get("holdout_oracle_gap_recovery_fraction"),
            "same_student_holdout_oracle_gap_recovery_fraction": same_student.get("holdout_oracle_gap_recovery_fraction"),
        }
    )
    return metrics


def _selected_next_step(status: str, deployable_positive: bool) -> str:
    if status != "pass":
        return "repair missing source artifacts before interpreting deployable support-head evidence"
    if deployable_positive:
        return "repeat the deployable support-head gate locally with a fresh seed before considering RunPod"
    return (
        "add a shuffled-causal-feature support-head null and rerun the local deployable support-head gate; "
        "do not run RunPod yet"
    )


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "not_found", "recommendation_accepted": False}
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "status": "read",
        "recommendation_accepted": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("strategic_change_level") == "major" or review.get("notify_ben") == "true":
        return (
            "GPT-5.5-Pro review requested a major or notify-Ben shift. This gate accepts it: "
            "sparse-support identity stays retired, support discovery remains secondary, and Ben "
            "should be notified before treating support discovery as a primary claim."
        )
    return "No major strategy-review direction shift recorded for this gate."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "support_head_metrics.csv", summary["support_head_metrics"])
    _write_csv(out_dir / "null_controls.csv", summary["null_controls"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["aggregate_metrics"]
    lines = [
        "# ACSR Deployable Support-Head Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Learned-head heldout loss delta vs router: `{metrics.get('learned_head_holdout_intervention_minus_router_loss', '')}`",
        f"- Learned-head heldout oracle-gap recovery: `{metrics.get('learned_head_holdout_oracle_gap_recovery_fraction', '')}`",
        f"- Upstream oracle CE headroom: `{metrics.get('sparse_oracle_minus_sparse_default_heldout_ce_delta', '')}`",
        "",
        summary["direction_shift"],
        "",
        "This is a local command-driven gate. It does not run RunPod, and it does not revive the sparse-support identity claim.",
    ]
    if summary["claim_blockers"]:
        lines.extend(["", "## Claim Blockers"])
        for blocker in summary["claim_blockers"]:
            lines.append(f"- `{blocker['criterion']}`: {blocker['failure_reason']}")
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    for row in rows:
        if row.get("component") == component:
            return row
    return {}


def _prior_null(rows: list[Any], control: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("control") == control:
            return row
    return {}


def _null_present(rows: list[dict[str, Any]], control: str) -> bool:
    for row in rows:
        if row.get("control") == control:
            return bool(row.get("present"))
    return False


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _negative(value: Any) -> bool:
    number = _number(value)
    return number is not None and number < 0.0


def _at_least(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number <= threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-dir", type=Path, default=DEFAULT_SOURCE_AUDIT_DIR)
    parser.add_argument("--acsr-gate-dir", type=Path, default=DEFAULT_ACSR_GATE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-oracle-ce-headroom", type=float, default=0.01)
    parser.add_argument("--min-holdout-recovery-fraction", type=float, default=0.5)
    args = parser.parse_args()
    summary = run_acsr_deployable_support_head_gate(
        source_audit_dir=args.source_audit_dir,
        acsr_gate_dir=args.acsr_gate_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
        min_oracle_ce_headroom=args.min_oracle_ce_headroom,
        min_holdout_recovery_fraction=args.min_holdout_recovery_fraction,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
