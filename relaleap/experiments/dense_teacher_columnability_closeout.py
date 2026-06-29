"""Close out the current dense-teacher sparse-columnability rescue branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DENSE_PRIMARY_DIR = Path("results/reports/dense_primary_mechanism_assay")
DEFAULT_DISTILLATION_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_GATE_DIR = Path("results/reports/dense_teacher_columnability_gate")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_columnability_closeout")

CLOSEOUT_DECISION = "dense_teacher_sparse_columnability_branch_closed_for_failure_localization"
INSUFFICIENT_EVIDENCE_DECISION = "dense_teacher_sparse_columnability_closeout_failed_closed"
NEXT_STEP = (
    "scaffold dense_teacher_failure_localization.py with oracle-support, retrained-oracle, "
    "pair-composer/gated-value, dense/rank/norm control, and shuffled/random/token-position null rows"
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_criteria.csv",
    "process_state.csv",
    "notes.md",
)


def run_dense_teacher_columnability_closeout(
    *,
    dense_primary_dir: Path = DEFAULT_DENSE_PRIMARY_DIR,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    gate_dir: Path = DEFAULT_GATE_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record a fail-closed local closeout and select failure localization."""

    start = time.time()
    dense_primary = _read_json(dense_primary_dir / "summary.json")
    distillation = _read_json(distillation_dir / "summary.json")
    gate = _read_json(gate_dir / "summary.json")
    review = _strategy_review(strategy_review_path)
    process_rows = _process_rows()
    source_rows = [
        _source_row("dense_primary_mechanism_assay", dense_primary_dir / "summary.json", dense_primary),
        _source_row(
            "dense_teacher_residual_distillation_comparison",
            distillation_dir / "summary.json",
            distillation,
        ),
        _source_row("dense_teacher_columnability_gate", gate_dir / "summary.json", gate),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "read" if strategy_review_path.is_file() else "missing_optional",
            "decision": review.get("recommended_next_action", ""),
            "claim_status": (
                f"strategic_change_level={review.get('strategic_change_level', '')}; "
                f"notify_ben={review.get('notify_ben', '')}"
            ),
            "git_commit": "",
        },
    ]
    evidence = _evidence_snapshot(dense_primary, distillation, gate, review)
    criteria = _closeout_criteria(evidence, source_rows)
    failures = [row for row in criteria if not row["passed"]]
    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE_DECISION
        claim_status = "dense_teacher_closeout_source_evidence_incomplete"
        selected_next_step = "repair dense-teacher closeout source artifacts before interpretation"
        rationale = "The closeout cannot retire the branch until the comparison failure and local gate contract are present."
    else:
        status = "pass"
        decision = CLOSEOUT_DECISION
        claim_status = "dense_teacher_sparse_columnability_not_established_current_branch_retired"
        selected_next_step = NEXT_STEP
        rationale = (
            "The dense teacher strongly improves CE, but the sparse top-k2 students and ACSR-predicted "
            "support fail the dense-teacher gate. Calibrated and norm-budgeted rescue arms do not close "
            "the teacher CE gap or beat all null/control checks, so another loss rescue has low "
            "information gain. The next local evidence should assign failure to support prediction, "
            "value representability, or value composition."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "selected_next_step": selected_next_step,
        "source_rows": source_rows,
        "closeout_criteria": criteria,
        "process_state": process_rows,
        "evidence": evidence,
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "failures": failures,
        "rationale": rationale,
        "backend_policy": "RunPod/Colab remain idle; this is a local closeout and failure-localization selection.",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence_snapshot(
    dense_primary: dict[str, Any],
    distillation: dict[str, Any],
    gate: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    variant_rows = _list(distillation.get("variant_rows"))
    base = _float_or_none(distillation.get("base_ce_loss"))
    teacher = _float_or_none(distillation.get("dense_teacher_ce_loss"))
    acsr = _find_arm(variant_rows, "acsr_predicted_future_support", scale=1.0)
    contextual = _find_arm(variant_rows, "promoted_contextual_topk2_ce_mse_distill", scale=1.0)
    mse_only = _find_arm(variant_rows, "promoted_contextual_topk2_mse_only_distill", scale=1.0)
    norm_budget = _find_arm(
        variant_rows,
        "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
        scale=1.0,
    )
    norm_budget_scaled = _find_arm(
        variant_rows,
        "norm_budgeted_promoted_contextual_topk2_ce_mse_distill_teacher_scale_0p25",
        scale=0.25,
    )
    return {
        "dense_primary_status": dense_primary.get("status"),
        "dense_primary_decision": dense_primary.get("decision"),
        "dense_primary_git_commit": dense_primary.get("git_commit"),
        "primary_teacher_arm": dense_primary.get("primary_arm"),
        "distillation_status": distillation.get("status"),
        "distillation_decision": distillation.get("decision"),
        "distillation_git_commit": distillation.get("git_commit"),
        "distillation_gate_passed": _as_dict(distillation.get("gate_status")).get(
            "passes_dense_teacher_distillation_gate"
        ),
        "gate_status": gate.get("status"),
        "gate_scientific_gate": gate.get("scientific_gate"),
        "gate_decision": gate.get("decision"),
        "gate_git_commit": gate.get("git_commit"),
        "base_ce_loss": base,
        "dense_teacher_ce_loss": teacher,
        "dense_teacher_ce_improvement": _delta(base, teacher),
        "acsr_ce_loss": _float_or_none(acsr.get("ce_loss")),
        "acsr_teacher_logit_mse": _float_or_none(acsr.get("teacher_logit_mse")),
        "contextual_ce_loss": _float_or_none(contextual.get("ce_loss")),
        "contextual_teacher_logit_mse": _float_or_none(contextual.get("teacher_logit_mse")),
        "mse_only_ce_loss": _float_or_none(mse_only.get("ce_loss")),
        "norm_budget_ce_loss": _float_or_none(norm_budget.get("ce_loss")),
        "norm_budget_teacher_logit_mse": _float_or_none(norm_budget.get("teacher_logit_mse")),
        "norm_budget_residual_norm_ratio": _float_or_none(norm_budget.get("residual_norm_ratio")),
        "norm_budget_scaled_ce_loss": _float_or_none(norm_budget_scaled.get("ce_loss")),
        "norm_budget_scaled_teacher_logit_mse": _float_or_none(norm_budget_scaled.get("teacher_logit_mse")),
        "norm_budget_scaled_residual_norm_ratio": _float_or_none(norm_budget_scaled.get("residual_norm_ratio")),
        "failure_criteria": [
            row.get("criterion")
            for row in _list(distillation.get("failures"))
            if row.get("criterion")
        ],
        "strategy_verdict": review.get("verdict"),
        "strategy_recommended_next_action": review.get("recommended_next_action"),
        "ben_notification_required": review.get("ben_notification_required"),
    }


def _closeout_criteria(
    evidence: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_missing = [row["source"] for row in source_rows if row["source"] != "strategy_review" and not row["present"]]
    return [
        _criterion(
            "source_artifacts_present",
            not source_missing,
            "dense primary, distillation comparison, and columnability gate summaries are present",
            source_missing,
            "required source summary missing",
        ),
        _criterion(
            "dense_teacher_improves_base",
            evidence["dense_teacher_ce_improvement"] is not None and evidence["dense_teacher_ce_improvement"] > 0.0,
            "dense teacher CE must improve over base CE before branch closeout is meaningful",
            {
                "base_ce_loss": evidence["base_ce_loss"],
                "dense_teacher_ce_loss": evidence["dense_teacher_ce_loss"],
                "improvement": evidence["dense_teacher_ce_improvement"],
            },
            "dense teacher did not improve base CE",
        ),
        _criterion(
            "distillation_gate_failed_closed",
            evidence["distillation_status"] == "fail" and evidence["distillation_gate_passed"] is False,
            "dense-teacher sparse-student comparison must fail its scientific gate",
            {
                "status": evidence["distillation_status"],
                "gate_passed": evidence["distillation_gate_passed"],
            },
            "distillation comparison has not failed closed",
        ),
        _criterion(
            "local_contract_completed",
            evidence["gate_status"] == "pass" and evidence["gate_scientific_gate"] == "ready_for_local_validation",
            "columnability gate contract is complete, so the failure is scientific rather than missing accounting",
            {
                "status": evidence["gate_status"],
                "scientific_gate": evidence["gate_scientific_gate"],
            },
            "local dense-teacher contract is not complete",
        ),
        _criterion(
            "norm_budget_rescue_not_supported",
            evidence["norm_budget_ce_loss"] is not None
            and evidence["contextual_ce_loss"] is not None
            and evidence["norm_budget_ce_loss"] > evidence["contextual_ce_loss"],
            "norm-budgeted rescue does not improve the promoted contextual sparse student",
            {
                "norm_budget_ce_loss": evidence["norm_budget_ce_loss"],
                "contextual_ce_loss": evidence["contextual_ce_loss"],
                "norm_budget_residual_norm_ratio": evidence["norm_budget_residual_norm_ratio"],
            },
            "norm-budgeted sparse rescue is not clearly negative",
        ),
        _criterion(
            "failure_localization_selected_by_strategy_review",
            "failure-localization" in str(evidence["strategy_recommended_next_action"]).lower()
            or "failure localization" in str(evidence["strategy_recommended_next_action"]).lower()
            or str(evidence["strategy_verdict"]).upper() == "PIVOT",
            "external review recommends pivoting from rescue losses to local failure localization",
            {
                "verdict": evidence["strategy_verdict"],
                "recommended_next_action": evidence["strategy_recommended_next_action"],
            },
            "strategy review does not support the closeout direction",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status", ""),
        "decision": packet.get("decision", ""),
        "claim_status": packet.get("claim_status", ""),
        "git_commit": packet.get("git_commit", ""),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "present": path.is_file(),
        "path": str(path),
        "strategic_change_level": "",
        "notify_ben": False,
        "ben_notification_required": False,
        "recommended_next_action": "",
        "verdict": "",
    }
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "notify_ben":
            data[key] = value.lower() == "true"
        elif key in {"strategic_change_level", "recommended_next_action", "verdict"}:
            data[key] = value
    data["ben_notification_required"] = (
        data.get("notify_ben") is True or data.get("strategic_change_level") == "major"
    )
    return data


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": review.get("strategic_change_level", ""),
        "verdict": review.get("verdict", ""),
        "ben_should_be_notified": bool(review.get("ben_notification_required")),
        "direction": "stop dense-teacher sparse-student rescue stacking; localize representation/routing/value-composition failure",
        "recommendation_disposition": "accepted",
        "disposition_reason": (
            "Oracle-support/value-composition upper bounds are a higher-information local test than another loss variant."
        ),
    }


def _process_rows() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", "relaleap-cli-loop|python -m relaleap"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        output = ""
    pids: list[str] = []
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            pids.append(parts[0])
            rows.append({"pid": parts[0], "command": parts[1] if len(parts) > 1 else ""})
    blank_rows = rows and all(not row["command"] for row in rows)
    if pids and blank_rows:
        try:
            ps_output = subprocess.check_output(
                ["ps", "-p", ",".join(pids), "-o", "pid=,command="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            rows = []
            for line in ps_output.splitlines():
                parts = line.strip().split(maxsplit=1)
                rows.append({"pid": parts[0], "command": parts[1] if len(parts) > 1 else ""})
        except subprocess.CalledProcessError:
            pass
    return rows


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_criteria.csv", summary["closeout_criteria"])
    _write_csv(out_dir / "process_state.csv", summary["process_state"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    shift = summary["direction_shift"]
    lines = [
        "# Dense Teacher Columnability Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben should be notified: `{shift['ben_should_be_notified']}`",
        "",
        "## Evidence",
        "",
        f"- Dense teacher CE: `{evidence['dense_teacher_ce_loss']}`",
        f"- Base CE: `{evidence['base_ce_loss']}`",
        f"- ACSR CE / teacher-logit MSE: `{evidence['acsr_ce_loss']}` / `{evidence['acsr_teacher_logit_mse']}`",
        f"- Promoted contextual CE / teacher-logit MSE: `{evidence['contextual_ce_loss']}` / `{evidence['contextual_teacher_logit_mse']}`",
        f"- Norm-budgeted CE / residual norm ratio: `{evidence['norm_budget_ce_loss']}` / `{evidence['norm_budget_residual_norm_ratio']}`",
        "",
        str(summary["rationale"]),
        "",
        "## Direction Shift",
        "",
        (
            "GPT-5.5-Pro recommended a major pivot from dense-teacher sparse-student rescue losses "
            "to failure localization. This run accepts that recommendation; Ben should be notified."
        ),
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _find_arm(rows: list[dict[str, Any]], arm: str, *, scale: float) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm and _float_or_none(row.get("teacher_scale")) == scale:
            return row
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-primary-dir", type=Path, default=DEFAULT_DENSE_PRIMARY_DIR)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_columnability_closeout(
        dense_primary_dir=args.dense_primary_dir,
        distillation_dir=args.distillation_dir,
        gate_dir=args.gate_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
