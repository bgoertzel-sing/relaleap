"""Close or redirect the low-churn MLP sparse-factorization ceiling path."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DECISION_AUDIT = Path("results/reports/low_churn_mlp_sparse_factorization_decision_audit/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_closeout")

VALUE_DICTIONARY_RESCUE_ACTION = "design_value_dictionary_capacity_rescue_before_gpu"
REPAIR_ACTION = "repair_low_churn_mlp_sparse_factorization_ceiling_closeout_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_ceiling_closeout(
    *,
    decision_audit_path: Path = DEFAULT_DECISION_AUDIT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed closeout for the current vector-centroid ceiling."""

    start = time.time()
    audit = _read_json(decision_audit_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("low_churn_mlp_sparse_factorization_decision_audit", decision_audit_path, audit),
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
        },
    ]
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(audit, strategy)
    selected = not failures and all(row["passed"] for row in decision_matrix if row["required"])
    selected_next_action = VALUE_DICTIONARY_RESCUE_ACTION if selected else REPAIR_ACTION
    status = "pass" if selected else "fail"
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_vector_centroid_ceiling_closed"
            if selected
            else "low_churn_mlp_sparse_factorization_ceiling_closeout_failed_closed"
        ),
        "claim_status": (
            "current_sparse_factorization_ceiling_closed_value_dictionary_rescue_selected"
            if selected
            else "sparse_factorization_ceiling_closeout_sources_incomplete_or_inconsistent"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "design a local value-dictionary capacity rescue with richer reusable values, dense/low-rank controls, and target non-columnability gates before any GPU validation"
            if selected
            else "repair or regenerate the sparse-factorization decision audit before selecting a next step"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout/design only; RunPod and Colab remain blocked by sparse-factorization evidence",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": _candidate_actions(selected_next_action),
        "failures": failures + [row for row in decision_matrix if row["required"] and not row["passed"]],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": _rationale(audit, selected),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(audit: dict[str, Any], strategy: dict[str, Any]) -> list[dict[str, Any]]:
    global_r2 = _float(audit.get("global_dictionary_oracle_r2"))
    learned_r2 = _float(audit.get("learned_router_heldout_r2"))
    return [
        {
            "signal": "decision_audit_blocks_gpu",
            "required": True,
            "passed": (
                audit.get("status") == "pass"
                and audit.get("advance_to_gpu_validation") is False
                and audit.get("promotion_allowed") is False
                and audit.get("requires_gpu_now") is False
            ),
            "actual": {
                "decision": audit.get("decision"),
                "claim_status": audit.get("claim_status"),
                "selected_next_action": audit.get("selected_next_action"),
            },
            "expected": "decision audit must be complete and explicitly block GPU promotion",
        },
        {
            "signal": "exact_oracle_labeled_nondeployable",
            "required": True,
            "passed": audit.get("exact_oracle_nondeployable") is True,
            "actual": {"exact_oracle_nondeployable": audit.get("exact_oracle_nondeployable")},
            "expected": "exact per-row oracle must be treated as target leakage, not reusable-column evidence",
        },
        {
            "signal": "learned_router_negative_or_far_from_oracle",
            "required": True,
            "passed": audit.get("learned_router_blocks_gpu") is True and (learned_r2 is None or learned_r2 < 0.5),
            "actual": {
                "learned_router_blocks_gpu": audit.get("learned_router_blocks_gpu"),
                "learned_router_heldout_r2": learned_r2,
                "oracle_learned_r2_gap": audit.get("oracle_learned_r2_gap"),
            },
            "expected": "deployable learned support remains below the local R2 gate",
        },
        {
            "signal": "global_dictionary_ceiling_weak",
            "required": True,
            "passed": global_r2 is not None and global_r2 < 0.5,
            "actual": {
                "global_dictionary_oracle_r2": global_r2,
                "blocking_blame": audit.get("blocking_blame", []),
            },
            "expected": "oracle support over reusable dictionary should fail the current heldout R2 threshold",
        },
        {
            "signal": "strategy_review_accepts_no_gpu_audit_direction",
            "required": False,
            "passed": (
                not strategy["present"]
                or "decision/blame audit" in strategy["recommended_next_action"]
                or "sparse-factorization" in strategy["recommended_next_action"]
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "optional strategy review should be compatible with local sparse-factorization closeout",
        },
    ]


def _candidate_actions(selected_next_action: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": VALUE_DICTIONARY_RESCUE_ACTION,
            "disposition": "selected" if selected_next_action == VALUE_DICTIONARY_RESCUE_ACTION else "blocked",
            "reason": "The current vector-centroid reusable dictionary is weak even with oracle support, so support-router redesign alone is not the next coherent step.",
            "next_step": "write a local value-dictionary rescue design with richer reusable values and explicit closeout gates",
            "claim_status": "value_dictionary_capacity_rescue_selected_before_gpu",
        },
        {
            "candidate_action": "redesign_prefix_safe_support_router_before_gpu",
            "disposition": "rejected",
            "reason": "Learned support is weak, but the reusable global-dictionary oracle is also weak, so router-only work would not address the present ceiling.",
            "next_step": "only reopen router design if a stronger reusable dictionary oracle clears the local threshold",
            "claim_status": "router_only_redesign_deferred",
        },
        {
            "candidate_action": "run_runpod_sparse_factorization_validation",
            "disposition": "rejected",
            "reason": "Exact oracle is nondeployable and both learned/deployable and reusable-dictionary evidence block GPU validation.",
            "next_step": "keep backend unused until a local value-dictionary or target-columnability gate passes",
            "claim_status": "gpu_validation_blocked",
        },
    ]


def _rationale(audit: dict[str, Any], selected: bool) -> str:
    if not selected:
        return "The closeout could not establish the required local source evidence, so it fails closed."
    return (
        "The current sparse-factorization ceiling is closed for the vector-centroid reusable-dictionary path: "
        "exact oracle reconstruction is nondeployable, the learned router remains below the deployable gate, "
        "and oracle support over a reusable global dictionary is also below the local heldout R2 threshold. "
        "The next bounded step is a local value-dictionary capacity rescue design, not RunPod validation or "
        "router-only tuning."
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


def _source_failures(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"source": row["source"], "reason": f"{row['path']} is missing"}
        for row in rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    values = {
        "present": path.is_file(),
        "path": str(path),
        "strategic_change_level": _header_value(text, "strategic_change_level"),
        "notify_ben": _header_value(text, "notify_ben"),
        "recommended_next_action": _header_value(text, "recommended_next_action"),
        "verdict": _header_value(text, "verdict"),
    }
    values["ben_notification_required"] = (
        str(values["notify_ben"]).lower() == "true"
        or str(values["strategic_change_level"]).lower() == "major"
    )
    return values


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; closeout relies on command-generated local artifacts."
    if strategy["ben_notification_required"]:
        return "Strategy review requested Ben notification or recorded a major change; this closeout preserves that note."
    return "Strategy review recommendation accepted as compatible with local no-GPU sparse-factorization closeout."


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
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


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Low-Churn MLP Sparse-Factorization Ceiling Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
            "",
            summary["rationale"],
            "",
        ]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-audit", type=Path, default=DEFAULT_DECISION_AUDIT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_sparse_factorization_ceiling_closeout(
        decision_audit_path=args.decision_audit,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
