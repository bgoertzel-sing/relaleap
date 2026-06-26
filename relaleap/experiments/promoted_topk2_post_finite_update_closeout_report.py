"""Close out promoted top-k-2 finite-update evidence and select one next step."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)
from relaleap.experiments.promoted_topk2_pairwise_value_interaction_localization_audit import (
    PAIRWISE_VALUE_INTERACTION_DIFFUSE,
)
from relaleap.experiments.promoted_topk2_retention_synthesis_gate import (
    CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
)


DEFAULT_PAIRWISE_LOCALIZATION = Path(
    "results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit/summary.json"
)
DEFAULT_FINITE_UPDATE_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json"
)
DEFAULT_RETENTION_SYNTHESIS = Path(
    "results/reports/token_larger_promoted_topk2_retention_synthesis_gate/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_post_finite_update_closeout"
)

POST_FINITE_UPDATE_CLOSEOUT_SELECTED = "post_finite_update_closeout_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "extend_causal_fingerprint_control_matrix_with_finite_update_fields"


def run_promoted_topk2_post_finite_update_closeout_report(
    *,
    pairwise_localization_path: Path = DEFAULT_PAIRWISE_LOCALIZATION,
    finite_update_report_path: Path = DEFAULT_FINITE_UPDATE_REPORT,
    retention_synthesis_path: Path = DEFAULT_RETENTION_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Fail-closed closeout after the finite-update order-control audit."""

    start = time.time()
    pairwise = _read_json_object(pairwise_localization_path)
    finite_update = _read_json_object(finite_update_report_path)
    retention = _read_json_object(retention_synthesis_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("pairwise_value_interaction_localization", pairwise_localization_path, pairwise),
        _source_row("finite_update_order_control", finite_update_report_path, finite_update),
        _source_row("retention_synthesis_gate", retention_synthesis_path, retention),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence_snapshot(pairwise=pairwise, finite_update=finite_update, retention=retention)
    failures = _failures(source_rows=source_rows, evidence=evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_step = "repair_missing_or_inconsistent_post_finite_update_sources"
        rationale = (
            "The post-finite-update closeout cannot select a next step because a "
            "required report is missing, failing, or inconsistent with the current "
            "diffuse-value / bounded-CE finite-update interpretation."
        )
    else:
        status = "pass"
        decision = POST_FINITE_UPDATE_CLOSEOUT_SELECTED
        selected_next_action = SELECTED_NEXT_ACTION
        next_step = (
            "implement and run a no-training causal fingerprint/control matrix "
            "extension that includes per-token forward-vs-reverse CE, symmetric KL, "
            "logit MSE, support-set, token-position, support-churn, and residual-delta "
            "strata for promoted top-k-2 plus rank-matched top-k-1, random fixed "
            "top-k-2, and dense active-rank controls"
        )
        rationale = (
            "The value-interaction localization audit is diffuse, so another value "
            "or router mitigation family is not authorized. The finite-update audit "
            "shows material residual/logit order sensitivity with CE bounded at the "
            "aggregate guardrail and existing per-token CE/KL strata. The next useful "
            "bounded step is therefore to promote those finite-update fields into the "
            "causal fingerprint/control matrix before any causal-cooperation claim."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "claim_statuses": {
            "contextual_topk2_router": "operational_default_support_routing",
            "topk2_causal_cooperation": "blocked_pending_extended_control_matrix",
            "value_router_mitigation_family": "closed_for_now_diffuse_or_not_established",
            "finite_update_order_sensitivity": "material_residual_logit_risk_ce_guardrail_bounded",
        },
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "selected_next_step_csv": str(out_dir / "selected_next_step.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(
        out_dir / "selected_next_step.csv",
        ["selected_next_action", "next_step"],
        [{"selected_next_action": selected_next_action, "next_step": next_step}],
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence_snapshot(
    *,
    pairwise: dict[str, Any],
    finite_update: dict[str, Any],
    retention: dict[str, Any],
) -> dict[str, Any]:
    finite_metrics = finite_update.get("metrics", {})
    pairwise_metrics = pairwise.get("metrics", {})
    return {
        "pairwise_decision": pairwise.get("decision"),
        "pairwise_localization_status": pairwise.get("localization_status"),
        "pairwise_top_pair_abs_synergy_share": _float_or_none(
            pairwise_metrics.get("top_pair_abs_synergy_share")
        ),
        "pairwise_top3_discovery_confirmation_overlap": _float_or_none(
            pairwise_metrics.get("discovery_confirmation_top3_overlap")
        ),
        "finite_update_decision": finite_update.get("decision"),
        "topk2_mean_commutator_anchor_ce_abs_delta": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_ce_abs_delta")
        ),
        "topk2_mean_commutator_anchor_logit_mse": _float_or_none(
            finite_metrics.get("topk2_mean_commutator_anchor_logit_mse")
        ),
        "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio")
        ),
        "topk2_to_dense_mean_commutator_anchor_logit_mse_ratio": _float_or_none(
            finite_metrics.get("topk2_to_dense_mean_commutator_anchor_logit_mse_ratio")
        ),
        "per_token_commutator_row_count": _float_or_none(
            finite_metrics.get("per_token_commutator_row_count")
        ),
        "per_token_commutator_ce_abs_delta_mean": _float_or_none(
            finite_metrics.get("per_token_commutator_ce_abs_delta_mean")
        ),
        "per_token_commutator_symmetric_kl_mean": _float_or_none(
            finite_metrics.get("per_token_commutator_symmetric_kl_mean")
        ),
        "retention_synthesis_decision": retention.get("decision"),
        "retention_synthesis_next_step": retention.get("next_step"),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:3]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expected = {
        "pairwise_decision": PAIRWISE_VALUE_INTERACTION_DIFFUSE,
        "pairwise_localization_status": "diffuse",
        "finite_update_decision": FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
        "retention_synthesis_decision": CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    required_numeric = (
        "topk2_mean_commutator_anchor_ce_abs_delta",
        "topk2_mean_commutator_anchor_logit_mse",
        "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio",
        "topk2_to_dense_mean_commutator_anchor_logit_mse_ratio",
        "per_token_commutator_row_count",
        "per_token_commutator_ce_abs_delta_mean",
        "per_token_commutator_symmetric_kl_mean",
    )
    for field in required_numeric:
        if evidence.get(field) is None:
            failures.append(
                {
                    "source": "evidence_snapshot",
                    "field": field,
                    "expected": "numeric value",
                    "actual": None,
                }
            )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted the recommendation to use the refreshed pairwise localization "
            "audit as a fail-closed gate; the diffuse result blocks new value/router "
            "mitigation families and sends the loop to finite-update control-matrix "
            "extension"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Promoted Top-k-2 Post-Finite-Update Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Pairwise localization: `{evidence['pairwise_localization_status']}`",
        f"- Finite-update decision: `{evidence['finite_update_decision']}`",
        "- Top-k-2/top-k-1 commutator logit-MSE ratio: "
        f"`{evidence['topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio']}`",
        "- Per-token commutator CE abs-delta mean: "
        f"`{evidence['per_token_commutator_ce_abs_delta_mean']}`",
        "- Per-token commutator symmetric KL mean: "
        f"`{evidence['per_token_commutator_symmetric_kl_mean']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["next_step"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairwise-localization",
        type=Path,
        default=DEFAULT_PAIRWISE_LOCALIZATION,
    )
    parser.add_argument(
        "--finite-update-report",
        type=Path,
        default=DEFAULT_FINITE_UPDATE_REPORT,
    )
    parser.add_argument(
        "--retention-synthesis",
        type=Path,
        default=DEFAULT_RETENTION_SYNTHESIS,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_promoted_topk2_post_finite_update_closeout_report(
        pairwise_localization_path=args.pairwise_localization,
        finite_update_report_path=args.finite_update_report,
        retention_synthesis_path=args.retention_synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
