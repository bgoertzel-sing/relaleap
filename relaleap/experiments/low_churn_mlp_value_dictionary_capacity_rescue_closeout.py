"""Close out the low-churn MLP value-dictionary capacity rescue."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PREGATE_DIR = Path("results/reports/low_churn_mlp_value_dictionary_capacity_rescue_pregate")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_value_dictionary_capacity_rescue_closeout")

REPAIR_ACTION = "repair_value_dictionary_capacity_rescue_closeout_sources"
NEXT_BRANCH_SELECTOR_ACTION = "select_next_post_value_dictionary_local_branch_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_low_churn_mlp_value_dictionary_capacity_rescue_closeout(
    *,
    pregate_dir: Path = DEFAULT_PREGATE_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the value-dictionary pregate and write a conservative closeout."""

    start = time.time()
    summary_path = pregate_dir / "summary.json"
    candidate_path = pregate_dir / "candidate_metrics.csv"
    gate_path = pregate_dir / "gate_criteria.csv"
    pregate = _read_json(summary_path)
    candidates = _read_csv(candidate_path)
    gates = _read_csv(gate_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("value_dictionary_capacity_rescue_pregate_summary", summary_path, pregate, 1 if pregate else 0),
        _source("value_dictionary_capacity_rescue_candidate_metrics", candidate_path, {}, len(candidates)),
        _source("value_dictionary_capacity_rescue_gate_criteria", gate_path, {}, len(gates)),
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
            "row_count": 1 if strategy["present"] else 0,
        },
    ]
    closeout_rows = _closeout_rows(pregate, candidates, gates, strategy)
    failures = _source_failures(source_rows) + [
        row for row in closeout_rows if row["required"] and not row["passed"]
    ]
    selected = not failures
    selected_next_action = NEXT_BRANCH_SELECTOR_ACTION if selected else REPAIR_ACTION
    status = "pass" if selected else "fail"
    candidate_actions = _candidate_actions(selected_next_action)
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_value_dictionary_capacity_rescue_closed"
            if selected
            else "low_churn_mlp_value_dictionary_capacity_rescue_closeout_failed_closed"
        ),
        "claim_status": (
            "target_aware_value_dictionary_rescue_closed_no_gpu"
            if selected
            else "value_dictionary_capacity_rescue_closeout_sources_incomplete_or_inconsistent"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "add a compact post-value-dictionary branch selector before any new architecture or GPU validation"
            if selected
            else "repair or regenerate the value-dictionary pregate artifacts, then rerun this closeout"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local closeout only; RunPod and Colab remain blocked by target-aware/null/control gates",
        "source_rows": source_rows,
        "closeout_rows": closeout_rows,
        "candidate_actions": candidate_actions,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": _rationale(selected, pregate),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _closeout_rows(
    pregate: dict[str, Any],
    candidates: list[dict[str, str]],
    gates: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    best_sparse = _as_dict(pregate.get("best_sparse_oracle"))
    best_valid_null = _as_dict(pregate.get("best_valid_null"))
    best_control = _as_dict(pregate.get("best_capacity_control"))
    sparse_r2 = _float(pregate.get("best_sparse_oracle_r2"))
    null_r2 = _float(best_valid_null.get("heldout_reconstruction_r2"))
    control_r2 = _best_budget_matched_control_r2(candidates)
    valid_null_delta = _float(pregate.get("valid_null_delta_r2"))
    failed_science_gates = [
        row.get("criterion", "")
        for row in gates
        if row.get("gate_type") == "scientific_advancement" and str(row.get("passed")).lower() != "true"
    ]
    return [
        {
            "signal": "pregate_completed_and_blocks_gpu",
            "required": True,
            "passed": (
                pregate.get("status") == "pass"
                and pregate.get("advance_to_gpu_validation") is False
                and pregate.get("promotion_allowed") is False
                and pregate.get("requires_gpu_now") is False
            ),
            "actual": {
                "decision": pregate.get("decision"),
                "claim_status": pregate.get("claim_status"),
                "selected_next_action": pregate.get("selected_next_action"),
            },
            "expected": "pregate must be complete and explicitly block GPU validation",
        },
        {
            "signal": "best_sparse_is_target_aware_nondeployable",
            "required": True,
            "passed": (
                best_sparse.get("deployable") is False
                and best_sparse.get("target_access_at_eval") == "target_residual_vector"
                and best_sparse.get("support_source") == "heldout_target_nearest_code"
            ),
            "actual": {
                "candidate": best_sparse.get("candidate"),
                "deployable": best_sparse.get("deployable"),
                "target_access_at_eval": best_sparse.get("target_access_at_eval"),
                "support_source": best_sparse.get("support_source"),
            },
            "expected": "best sparse row must be labeled as a target-aware nondeployable ceiling",
        },
        {
            "signal": "valid_target_aware_null_ties_sparse",
            "required": True,
            "passed": (
                valid_null_delta is not None
                and abs(valid_null_delta) <= 1e-12
                and null_r2 is not None
                and sparse_r2 is not None
                and abs(null_r2 - sparse_r2) <= 1e-12
                and best_valid_null.get("valid_null_for_target_access") is True
            ),
            "actual": {
                "best_sparse_oracle_r2": sparse_r2,
                "best_valid_null_r2": null_r2,
                "valid_null_delta_r2": valid_null_delta,
                "best_valid_null": best_valid_null.get("candidate"),
            },
            "expected": "best sparse must not beat the valid target-aware null",
        },
        {
            "signal": "budget_matched_low_rank_dominates_sparse",
            "required": True,
            "passed": control_r2 is not None and sparse_r2 is not None and control_r2 >= sparse_r2 + 0.10,
            "actual": {
                "best_sparse_oracle_r2": sparse_r2,
                "best_budget_matched_low_rank_r2": control_r2,
                "best_capacity_control": best_control.get("candidate"),
            },
            "expected": "budget-matched low-rank control should dominate the sparse ceiling by at least 0.10 R2",
        },
        {
            "signal": "scientific_advancement_gates_fail_as_expected",
            "required": True,
            "passed": {
                "richer_sparse_oracle_min_r2",
                "dense_low_rank_control_not_dominant",
                "shuffled_and_route_nulls_rejected",
            }.issubset(set(failed_science_gates)),
            "actual": {"failed_scientific_gates": failed_science_gates},
            "expected": "threshold, control dominance, and null-rejection gates must all fail",
        },
        {
            "signal": "strategy_review_supports_closeout",
            "required": False,
            "passed": (
                not strategy["present"]
                or "null" in strategy["recommended_next_action"].lower()
                or "commit" in strategy["recommended_next_action"].lower()
                or strategy["verdict"] in {"FIX", "HOLD", ""}
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
                "verdict": strategy["verdict"],
            },
            "expected": "strategy review should be compatible with a no-GPU local closeout",
        },
    ]


def _best_budget_matched_control_r2(candidates: list[dict[str, str]]) -> float | None:
    values = [
        _float(row.get("heldout_reconstruction_r2"))
        for row in candidates
        if row.get("family") == "capacity_control" and row.get("budget_match_group") == "budget_matched_low_rank"
    ]
    return max([value for value in values if value is not None], default=None)


def _candidate_actions(selected_next_action: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": NEXT_BRANCH_SELECTOR_ACTION,
            "disposition": "selected" if selected_next_action == NEXT_BRANCH_SELECTOR_ACTION else "blocked",
            "reason": "The target-aware value-dictionary rescue is closed by a valid-null tie and low-rank control dominance.",
            "next_step": "select exactly one local post-value-dictionary branch before any new architecture work",
            "claim_status": "post_value_dictionary_local_branch_selection_needed",
        },
        {
            "candidate_action": "run_runpod_value_dictionary_validation",
            "disposition": "rejected",
            "reason": "The best sparse row is nondeployable, ties the valid target-aware null, and loses to budget-matched low-rank controls.",
            "next_step": "keep RunPod unused until a local mechanism gate passes",
            "claim_status": "gpu_validation_blocked",
        },
        {
            "candidate_action": "extend_target_aware_dictionary_rescue",
            "disposition": "rejected",
            "reason": "Further target-aware vector quantization would not address prefix-safe reusable support or causal mechanism evidence.",
            "next_step": "only reopen if a train-only deployable support/control design is specified by a later branch selector",
            "claim_status": "target_aware_dictionary_extension_closed",
        },
    ]


def _rationale(selected: bool, pregate: dict[str, Any]) -> str:
    if not selected:
        return "The closeout could not establish the required pregate evidence, so it fails closed."
    return (
        "The richer low-churn value dictionary improved the local sparse ceiling, but it did not rescue the branch: "
        f"best sparse R2 {pregate.get('best_sparse_oracle_r2')} ties the valid target-aware null with delta "
        f"{pregate.get('valid_null_delta_r2')}, remains nondeployable, and is dominated by low-rank capacity controls. "
        "GPU validation and further target-aware dictionary expansion are blocked."
    )


def _source(source: str, path: Path, payload: dict[str, Any], row_count: int) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "read" if path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "row_count": row_count,
    }


def _source_failures(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"source": row["source"], "reason": f"{row['path']} is missing or empty"}
        for row in rows
        if row["source"] != "strategy_review" and (not row["present"] or int(row.get("row_count", 0) or 0) <= 0)
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
        return "Strategy review requested Ben notification or a major direction change; this closeout records that flag without GPU action."
    return "Strategy review recommendation accepted: no GPU, close the target-aware dictionary rescue, and keep null/budget labels explicit."


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
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
            "# Low-Churn MLP Value-Dictionary Capacity Rescue Closeout",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            "",
            summary["rationale"],
            "",
        ]
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
    parser.add_argument("--pregate-dir", type=Path, default=DEFAULT_PREGATE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_low_churn_mlp_value_dictionary_capacity_rescue_closeout(
        pregate_dir=args.pregate_dir,
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
