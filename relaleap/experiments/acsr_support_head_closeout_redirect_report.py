"""Close out negative ACSR support-head evidence and redirect the next step."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DEPLOYABLE_GATE = Path("results/reports/acsr_deployable_support_head_gate_local/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_support_head_closeout_redirect")

SUPPORT_HEAD_NEGATIVE = "acsr_support_head_negative_closeout_redirect_selected"
INSUFFICIENT_EVIDENCE = "acsr_support_head_closeout_failed_closed"
FINITE_UPDATE_DENSE_CONTROL_ACTION = "finite_update_commutator_dense_control_assay"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "redirect_criteria.csv",
    "notes.md",
)


def run_acsr_support_head_closeout_redirect_report(
    *,
    deployable_gate_path: Path = DEFAULT_DEPLOYABLE_GATE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Freeze the local deployable support-head claim and select one branch."""

    start = time.time()
    gate = _read_json_object(deployable_gate_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("deployable_support_head_gate", deployable_gate_path, gate),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    metrics = _metrics(gate)
    criteria = _criteria(gate, metrics)
    hard_failures = [row for row in criteria if not row["passed"] and row["severity"] == "hard"]
    claim_failures = [row for row in criteria if not row["passed"] and row["severity"] == "claim_blocker"]

    if hard_failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_step = "repair or rerun the local deployable support-head gate before redirecting ACSR"
        rationale = (
            "The closeout cannot redirect the research branch because the source "
            "deployable support-head gate is missing, failing, or not the expected "
            "local negative gate."
        )
    else:
        status = "pass"
        decision = SUPPORT_HEAD_NEGATIVE
        selected_next_action = FINITE_UPDATE_DENSE_CONTROL_ACTION
        next_step = (
            "write a command-driven finite-update commutator assay for the ACSR "
            "residual path against matched dense residual controls, using CE only "
            "as a guardrail and reporting order sensitivity, symmetric KL, logit "
            "MSE, retention/churn, and dense-minus-sparse paired deltas"
        )
        rationale = (
            "The null-complete deployable support-head gate is locally negative: "
            "the learned head beats the newly added nulls but the oracle headroom "
            "is too small and sequence-heldout support-head forcing is unfavorable. "
            "Support discovery is therefore frozen as a claim under this regime; "
            "the next non-CE mechanism question is whether residual updates reduce "
            "finite-update interference versus dense controls."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "claim_statuses": {
            "deployable_support_discovery": (
                "frozen_negative_tiny_headroom_and_sequence_holdout_failure"
                if status == "pass"
                else "not_interpretable"
            ),
            "sparse_support_identity": (
                "retired_by_upstream_gate" if status == "pass" else "not_interpretable"
            ),
            "runpod_validation": "deferred_no_gpu_target",
            "ben_notification": "required" if strategy["ben_notification_required"] else "not_required",
        },
        "source_rows": source_rows,
        "redirect_criteria": criteria,
        "failures": hard_failures,
        "claim_blockers_preserved": claim_failures,
        "metrics": metrics,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _metrics(gate: dict[str, Any]) -> dict[str, Any]:
    aggregate = gate.get("aggregate_metrics") if isinstance(gate.get("aggregate_metrics"), dict) else {}
    rows = gate.get("support_head_metrics") if isinstance(gate.get("support_head_metrics"), list) else []
    sequence = _row(rows, "learned_sequence_support_head")
    null_rows = gate.get("null_controls") if isinstance(gate.get("null_controls"), list) else []
    return {
        "gate_status": gate.get("status"),
        "gate_decision": gate.get("decision"),
        "gate_claim_status": gate.get("claim_status"),
        "learned_head_holdout_delta_vs_router": _float_or_none(
            aggregate.get("learned_head_holdout_intervention_minus_router_loss")
        ),
        "learned_head_holdout_oracle_gap_recovery": _float_or_none(
            aggregate.get("learned_head_holdout_oracle_gap_recovery_fraction")
        ),
        "same_student_holdout_oracle_gap_recovery": _float_or_none(
            aggregate.get("same_student_holdout_oracle_gap_recovery_fraction")
        ),
        "upstream_oracle_ce_headroom": _float_or_none(
            aggregate.get("sparse_oracle_minus_sparse_default_heldout_ce_delta")
        ),
        "sequence_head_delta_vs_router": _float_or_none(
            sequence.get("holdout_intervention_minus_router_loss")
        ),
        "sequence_head_oracle_gap_recovery": _float_or_none(
            sequence.get("holdout_oracle_gap_recovery_fraction")
        ),
        "shuffled_causal_feature_null_present": _null_present(
            null_rows,
            "shuffled_causal_feature_support_head_null",
        ),
        "token_position_null_present": _null_present(null_rows, "token_position_support_null"),
    }


def _criteria(gate: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "deployable_gate_present",
            bool(gate),
            "hard",
            "source deployable support-head gate exists",
            bool(gate),
            "deployable support-head gate summary is missing",
        ),
        _criterion(
            "deployable_gate_passed_local_artifact_checks",
            metrics["gate_status"] == "pass",
            "hard",
            "source gate passed artifact and hard source checks",
            metrics["gate_status"],
            "source deployable support-head gate did not pass hard checks",
        ),
        _criterion(
            "deployable_gate_is_negative",
            metrics["gate_decision"] == "deployable_support_head_gate_blocks_claim_pending_nulls_or_headroom",
            "hard",
            "source gate blocks the deployable support-head claim",
            metrics["gate_decision"],
            "source deployable support-head gate is not the expected negative closeout source",
        ),
        _criterion(
            "sparse_identity_retired",
            metrics["gate_claim_status"] == "deployable_support_discovery_not_established_sparse_identity_retired",
            "hard",
            "upstream ACSR sparse-support identity remains retired",
            metrics["gate_claim_status"],
            "source gate did not preserve sparse-support identity retirement",
        ),
        _criterion(
            "null_complete",
            metrics["shuffled_causal_feature_null_present"] and metrics["token_position_null_present"],
            "hard",
            "shuffled-causal-feature and token/position support-head nulls exist",
            {
                "shuffled_causal_feature_null_present": metrics["shuffled_causal_feature_null_present"],
                "token_position_null_present": metrics["token_position_null_present"],
            },
            "support-head null controls are incomplete",
        ),
        _criterion(
            "oracle_headroom_tiny",
            _greater_than(metrics["upstream_oracle_ce_headroom"], -0.01),
            "claim_blocker",
            "oracle-support CE headroom is too small to justify GPU validation",
            metrics["upstream_oracle_ce_headroom"],
            "oracle-support headroom is not tiny; reroute may need strategic review",
        ),
        _criterion(
            "sequence_holdout_unfavorable",
            _greater_than(metrics["sequence_head_delta_vs_router"], 0.0),
            "claim_blocker",
            "sequence-heldout learned support-head forcing is unfavorable",
            metrics["sequence_head_delta_vs_router"],
            "sequence-heldout learned support head was not unfavorable",
        ),
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


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": "",
            "notify_ben": "",
            "recommended_next_action": "",
            "verdict": "",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "ben_notification_required": (
            header.get("notify_ben", "").lower() == "true"
            or header.get("strategic_change_level", "").lower() == "major"
        ),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; closeout relies on command artifacts only."
    if strategy["ben_notification_required"]:
        return (
            "Latest GPT-5.5-Pro review required Ben notification or marked a major "
            "shift; this closeout records that requirement while keeping the run "
            "bounded and local."
        )
    return (
        "Latest GPT-5.5-Pro review was accepted: the null-complete support-head "
        "gate has been rerun locally, RunPod is deferred, and this report records "
        "the resulting support-discovery freeze."
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "redirect_criteria.csv", summary["redirect_criteria"])
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
    metrics = summary["metrics"]
    lines = [
        "# ACSR Support-Head Closeout Redirect",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Learned-head heldout delta vs router: `{metrics['learned_head_holdout_delta_vs_router']}`",
        f"- Learned-head oracle-gap recovery: `{metrics['learned_head_holdout_oracle_gap_recovery']}`",
        f"- Upstream oracle CE headroom: `{metrics['upstream_oracle_ce_headroom']}`",
        f"- Sequence-head heldout delta vs router: `{metrics['sequence_head_delta_vs_router']}`",
        "",
        summary["rationale"],
        "",
        summary["strategy_review_handling"],
        "",
        "RunPod remains deferred because this report does not define a GPU validation target.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _row(rows: list[Any], component: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("component") == component:
            return row
    return {}


def _null_present(rows: list[Any], control: str) -> bool:
    for row in rows:
        if isinstance(row, dict) and row.get("control") == control:
            return bool(row.get("present"))
    return False


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _greater_than(value: Any, threshold: float) -> bool:
    number = _float_or_none(value)
    return number is not None and number > threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-gate", type=Path, default=DEFAULT_DEPLOYABLE_GATE)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_support_head_closeout_redirect_report(
        deployable_gate_path=args.deployable_gate,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
