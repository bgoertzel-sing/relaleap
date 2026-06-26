"""Close the promoted top-k-2 value/router mitigation family."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_POST_VALUE_CLOSEOUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_post_value_router_mitigation_closeout"
)
DEFAULT_PAIRWISE_LOCALIZATION_DIR = Path(
    "results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit"
)
DEFAULT_ACTIVE_TOPK1_SELECTION_DIR = Path(
    "results/reports/token_larger_active_topk1_next_evidence_selection"
)
DEFAULT_RETENTION_FOLLOWUP_DIR = Path(
    "results/reports/token_larger_active_topk1_retention_functional_churn_followup"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_promoted_topk2_post_localization_closeout"
)

POST_LOCALIZATION_CLOSEOUT = "promoted_topk2_value_router_family_closed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_RETENTION_CAUSAL_AUDIT = "retention_causal_audit_design"
SELECTED_RETENTION_BACKEND_REPEAT = "retention_churn_backend_repeat_if_needed"
SELECTED_MATCHED_DECONFOUNDING = "matched_deconfounding"


def run_promoted_topk2_post_localization_closeout_report(
    *,
    post_value_closeout_dir: Path = DEFAULT_POST_VALUE_CLOSEOUT_DIR,
    pairwise_localization_dir: Path = DEFAULT_PAIRWISE_LOCALIZATION_DIR,
    active_topk1_selection_dir: Path = DEFAULT_ACTIVE_TOPK1_SELECTION_DIR,
    retention_followup_dir: Path = DEFAULT_RETENTION_FOLLOWUP_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Close failed mitigation work and select the next non-mitigation branch."""

    start = time.time()
    packets = {
        "post_value_router_mitigation_closeout": _read_json_object(
            post_value_closeout_dir / "summary.json"
        ),
        "pairwise_value_interaction_localization": _read_json_object(
            pairwise_localization_dir / "summary.json"
        ),
        "active_topk1_next_evidence_selection": _read_json_object(
            active_topk1_selection_dir / "summary.json"
        ),
        "active_topk1_retention_functional_churn_followup": _read_json_object(
            retention_followup_dir / "summary.json"
        ),
    }
    paths = {
        "post_value_router_mitigation_closeout": post_value_closeout_dir / "summary.json",
        "pairwise_value_interaction_localization": pairwise_localization_dir
        / "summary.json",
        "active_topk1_next_evidence_selection": active_topk1_selection_dir
        / "summary.json",
        "active_topk1_retention_functional_churn_followup": retention_followup_dir
        / "summary.json",
    }
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [_source_row(name, paths[name], packets[name]) for name in paths]
    source_rows.append(
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
        }
    )
    metrics = _metrics(packets)
    closure_rows = _closure_rows(packets, metrics)
    failures = _failures(source_rows, packets)
    selected_next_branch = _selected_next_branch(packets, metrics, failures)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        rationale = (
            "The post-localization closeout cannot be interpreted because one "
            "or more required command-generated source artifacts is missing, "
            "failing, or inconsistent."
        )
        next_step = "repair missing post-localization closeout source artifacts"
    else:
        status = "pass"
        decision = POST_LOCALIZATION_CLOSEOUT
        claim_status = "topk2_value_router_mitigation_family_closed_no_promotion"
        rationale = _rationale(selected_next_branch, metrics)
        next_step = _next_step(selected_next_branch)

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_branch": selected_next_branch,
        "branch_options": [
            SELECTED_RETENTION_CAUSAL_AUDIT,
            SELECTED_RETENTION_BACKEND_REPEAT,
            SELECTED_MATCHED_DECONFOUNDING,
        ],
        "selection_gate": {
            "requires_gpu_now": selected_next_branch
            == SELECTED_RETENTION_BACKEND_REPEAT,
            "new_training_required": False,
            "adds_new_mitigation_family": False,
            "topk2_value_router_family_closed": bool(
                status == "pass"
                and metrics["pairwise_localization_decision"]
                == "pairwise_value_interaction_diffuse"
            ),
            "retention_followup_already_completed": bool(
                metrics["retention_followup_decision"]
                == "retention_functional_churn_bracket_supported"
            ),
        },
        "claim_statuses": {
            "contextual_topk2_router": "operational_default_train_time_support_selection",
            "topk2_causal_cooperation": "not_supported",
            "router_policy_mitigation": "closed_not_established",
            "value_router_mitigation_family": (
                "closed_not_established" if status == "pass" else INSUFFICIENT_EVIDENCE
            ),
            "pairwise_value_interaction": "not_localized",
            "active_rank_matched_topk1": "active_retention_churn_bracket",
        },
        "source_rows": source_rows,
        "closure_rows": closure_rows,
        "metrics": metrics,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "closure_rows_csv": str(out_dir / "closure_rows.csv"),
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
        out_dir / "closure_rows.csv",
        ["branch", "source_decision", "key_metric", "key_value", "disposition"],
        closure_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _metrics(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    localization = packets["pairwise_value_interaction_localization"]
    selection = packets["active_topk1_next_evidence_selection"]
    retention = packets["active_topk1_retention_functional_churn_followup"]
    localization_metrics = localization.get("metrics", {})
    retention_aggregates = retention.get("aggregates", {})
    return {
        "post_value_closeout_decision": packets[
            "post_value_router_mitigation_closeout"
        ].get("decision"),
        "post_value_selected_next_action": packets[
            "post_value_router_mitigation_closeout"
        ].get("selected_next_action"),
        "pairwise_localization_decision": localization.get("decision"),
        "pairwise_localization_status": localization.get("localization_status"),
        "top3_pair_abs_synergy_share": localization_metrics.get(
            "top3_pair_abs_synergy_share"
        ),
        "dominant_column_abs_synergy_share": localization_metrics.get(
            "dominant_column_abs_synergy_share"
        ),
        "frequency_control_primary_denominator_count": localization_metrics.get(
            "frequency_control_primary_denominator_count"
        ),
        "active_topk1_selected_experiment": selection.get("selected_experiment"),
        "matched_control_coverage_adequate": (
            selection.get("signals", {}).get("matched_control_coverage_adequate")
            if isinstance(selection.get("signals"), dict)
            else None
        ),
        "retention_followup_decision": retention.get("decision"),
        "retention_min_support_churn_advantage_topk1_vs_topk2": retention_aggregates.get(
            "min_support_churn_advantage_topk1_vs_topk2"
        ),
        "retention_min_commutator_advantage_topk1_vs_topk2": retention_aggregates.get(
            "min_commutator_anchor_advantage_topk1_vs_topk2"
        ),
        "retention_min_transfer_advantage_topk1_vs_dense": retention_aggregates.get(
            "min_transfer_advantage_topk1_vs_dense"
        ),
        "retention_ce_guardrail_all_packets": retention_aggregates.get(
            "ce_guardrail_all_packets"
        ),
    }


def _closure_rows(
    packets: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "branch": "router_policy_and_value_mitigation",
            "source_decision": metrics["post_value_closeout_decision"],
            "key_metric": "selected_next_action",
            "key_value": metrics["post_value_selected_next_action"],
            "disposition": "closed_not_established",
        },
        {
            "branch": "pairwise_value_interaction_localization",
            "source_decision": metrics["pairwise_localization_decision"],
            "key_metric": "localization_status",
            "key_value": metrics["pairwise_localization_status"],
            "disposition": "closed_no_new_mitigation_family",
        },
        {
            "branch": "active_topk1_next_evidence_selection",
            "source_decision": packets["active_topk1_next_evidence_selection"].get(
                "decision"
            ),
            "key_metric": "selected_experiment",
            "key_value": metrics["active_topk1_selected_experiment"],
            "disposition": "accepted_retention_branch",
        },
        {
            "branch": "active_topk1_retention_functional_churn_followup",
            "source_decision": metrics["retention_followup_decision"],
            "key_metric": "min_support_churn_advantage_topk1_vs_topk2",
            "key_value": metrics[
                "retention_min_support_churn_advantage_topk1_vs_topk2"
            ],
            "disposition": "completed_local_bracket",
        },
    ]


def _selected_next_branch(
    packets: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str | None:
    if failures:
        return None
    if metrics["active_topk1_selected_experiment"] != "retention_churn":
        return SELECTED_MATCHED_DECONFOUNDING
    if metrics["retention_followup_decision"] == "retention_functional_churn_bracket_supported":
        return SELECTED_RETENTION_CAUSAL_AUDIT
    return SELECTED_RETENTION_BACKEND_REPEAT


def _rationale(branch: str | None, metrics: dict[str, Any]) -> str:
    if branch == SELECTED_RETENTION_CAUSAL_AUDIT:
        return (
            "The current top-k-2 value/router mitigation family should close: "
            "router/value mitigations did not establish a promotion path and the "
            "pairwise localization audit was diffuse, so adding another "
            "mitigation family is underjustified. The active top-k-1 selector "
            "chose retention/churn, and the local four-control follow-up already "
            "supports that bracket with lower support churn and commutator risk "
            "under CE guardrails. The next bounded step is therefore a "
            "discriminative causal-retention audit design, not another top-k-2 "
            "mitigation."
        )
    if branch == SELECTED_RETENTION_BACKEND_REPEAT:
        return (
            "The active top-k-1 selector chose retention/churn, but the selected "
            "follow-up is not yet completed locally, so the next bounded step is "
            "to run or repair that retention/churn follow-up before interpreting "
            "the branch."
        )
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return (
            "The active top-k-1 selector did not choose retention/churn, so the "
            "loop should return to matched deconfounding coverage rather than "
            "designing a retention audit."
        )
    return "No branch selected."


def _next_step(branch: str | None) -> str:
    if branch == SELECTED_RETENTION_CAUSAL_AUDIT:
        return (
            "design one command-driven causal-retention audit that uses the "
            "completed local retention/churn bracket to test whether active "
            "rank-matched top-k-1's lower churn corresponds to reusable causal "
            "corrections under matched contexts and CE guardrails"
        )
    if branch == SELECTED_RETENTION_BACKEND_REPEAT:
        return (
            "run or repair the selected retention/functional-churn follow-up "
            "with the four required controls before adding new interpretation"
        )
    if branch == SELECTED_MATCHED_DECONFOUNDING:
        return "repair matched deconfounding/control coverage before retention claims"
    return "repair source artifacts before selecting a branch"


def _failures(
    source_rows: list[dict[str, Any]],
    packets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required_sources = source_rows[:4]
    for row in required_sources:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "summary_json",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
        elif row["status"] != "pass":
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": "pass",
                    "actual": row["status"],
                }
            )
    expectations = (
        (
            "post_value_router_mitigation_closeout",
            "decision",
            "promoted_topk2_mitigation_closeout_no_promotion",
        ),
        (
            "post_value_router_mitigation_closeout",
            "selected_next_action",
            "pairwise_value_interaction_localization_audit",
        ),
        (
            "pairwise_value_interaction_localization",
            "decision",
            "pairwise_value_interaction_diffuse",
        ),
        (
            "active_topk1_next_evidence_selection",
            "decision",
            "active_topk1_next_evidence_selected",
        ),
    )
    for source, field, expected in expectations:
        actual = packets[source].get(field)
        if actual != expected:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
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
        "claim_status": packet.get("claim_status") or packet.get("claim_gate"),
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
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {}
    for line in lines[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted: this closeout uses the already completed active-rank-matched "
            "top-k-1 selection report, preserves the retention/churn branch favored "
            "by the review, and records that no additional top-k-2 mitigation "
            "family should be opened from diffuse localization evidence"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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
    lines = [
        "# Promoted Top-k-2 Post-Localization Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next branch: `{summary['selected_next_branch']}`",
        f"- Rationale: {summary['rationale']}",
        f"- Next step: {summary['next_step']}",
        "",
        "## Strategy Review",
        "",
        f"- Present: `{summary['strategy_review']['present']}`",
        f"- Strategic change level: `{summary['strategy_review']['strategic_change_level']}`",
        f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
        f"- Incorporation: {summary['strategy_review']['incorporation']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- `{failure.get('source')}` `{failure.get('field')}`: "
                f"expected `{failure.get('expected')}`, actual `{failure.get('actual')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Close the promoted top-k-2 value/router mitigation family after "
            "pairwise localization and select one non-mitigation next branch."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--post-value-closeout-dir",
        type=Path,
        default=DEFAULT_POST_VALUE_CLOSEOUT_DIR,
    )
    parser.add_argument(
        "--pairwise-localization-dir",
        type=Path,
        default=DEFAULT_PAIRWISE_LOCALIZATION_DIR,
    )
    parser.add_argument(
        "--active-topk1-selection-dir",
        type=Path,
        default=DEFAULT_ACTIVE_TOPK1_SELECTION_DIR,
    )
    parser.add_argument(
        "--retention-followup-dir",
        type=Path,
        default=DEFAULT_RETENTION_FOLLOWUP_DIR,
    )
    parser.add_argument(
        "--strategy-review",
        type=Path,
        default=DEFAULT_STRATEGY_REVIEW,
    )
    args = parser.parse_args()
    summary = run_promoted_topk2_post_localization_closeout_report(
        out_dir=args.out,
        post_value_closeout_dir=args.post_value_closeout_dir,
        pairwise_localization_dir=args.pairwise_localization_dir,
        active_topk1_selection_dir=args.active_topk1_selection_dir,
        retention_followup_dir=args.retention_followup_dir,
        strategy_review_path=args.strategy_review,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
