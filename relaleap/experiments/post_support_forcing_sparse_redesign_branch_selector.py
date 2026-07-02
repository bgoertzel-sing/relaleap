"""Select the next step after support-forcing/pruning and redesign branches.

This selector is intentionally report-only. It reconciles the dense-teacher
support-forcing/pruning closeout with later local sparse value-support redesign
artifacts, then chooses exactly one bounded next action without running GPU
validation or opening a duplicate mechanism branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SUPPORT_CLOSEOUT = Path("results/reports/dense_teacher_support_forcing_pruning_closeout/summary.json")
DEFAULT_SPARSE_FACTOR_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_sparse_factorization_ceiling_closeout/summary.json"
)
DEFAULT_VALUE_DICTIONARY_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_value_dictionary_capacity_rescue_closeout/summary.json"
)
DEFAULT_POST_VALUE_SELECTOR = Path("results/reports/post_value_dictionary_branch_selector/summary.json")
DEFAULT_MECHANISM_INVENTORY = Path("results/reports/mechanism_source_inventory/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_support_forcing_sparse_redesign_branch_selector")

REPAIR_ACTION = "repair_post_support_forcing_sparse_redesign_sources"
STRATEGY_REFRESH_ACTION = "request_strategy_review_before_new_sparse_value_support_redesign"
LOW_CHURN_FACTOR_ACTION = "reopen_low_churn_sparse_factorization_ceiling"
VALUE_DICTIONARY_ACTION = "reopen_target_aware_value_dictionary_rescue"
GPU_ACTION = "launch_gpu_validation_for_sparse_value_support_redesign"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_support_forcing_sparse_redesign_branch_selector(
    *,
    support_closeout_path: Path = DEFAULT_SUPPORT_CLOSEOUT,
    sparse_factor_closeout_path: Path = DEFAULT_SPARSE_FACTOR_CLOSEOUT,
    value_dictionary_closeout_path: Path = DEFAULT_VALUE_DICTIONARY_CLOSEOUT,
    post_value_selector_path: Path = DEFAULT_POST_VALUE_SELECTOR,
    mechanism_inventory_path: Path = DEFAULT_MECHANISM_INVENTORY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a deterministic local branch selector artifact."""

    start = time.time()
    support_closeout = _read_json(support_closeout_path)
    sparse_factor = _read_json(sparse_factor_closeout_path)
    value_closeout = _read_json(value_dictionary_closeout_path)
    post_value = _read_json(post_value_selector_path)
    inventory = _read_json(mechanism_inventory_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("support_forcing_pruning_closeout", support_closeout_path, support_closeout),
        _source_row("low_churn_sparse_factorization_closeout", sparse_factor_closeout_path, sparse_factor),
        _source_row("value_dictionary_capacity_rescue_closeout", value_dictionary_closeout_path, value_closeout),
        _source_row("post_value_dictionary_branch_selector", post_value_selector_path, post_value),
        _source_row("mechanism_source_inventory", mechanism_inventory_path, inventory),
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
        },
    ]
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(support_closeout, sparse_factor, value_closeout, post_value, inventory, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_support_forcing_sparse_redesign_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate missing post-support-forcing sparse-redesign source artifacts"
        claim_status = "post_support_forcing_sparse_redesign_sources_incomplete"
        rationale = "The selector cannot choose a branch until required local source artifacts are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_support_forcing_sparse_redesign_branch_selected"
        selected_next_action = selected_row["candidate_action"]
        selected_next_step = selected_row["next_step"]
        claim_status = selected_row["claim_status"]
        rationale = selected_row["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "training_executed": False,
        "backend_policy": "local branch selection only; Colab and RunPod remain blocked",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "direction_shift": _direction_shift(strategy),
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(
    support_closeout: dict[str, Any],
    sparse_factor: dict[str, Any],
    value_closeout: dict[str, Any],
    post_value: dict[str, Any],
    inventory: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    support_evidence = _as_dict(support_closeout.get("evidence"))
    return [
        {
            "signal": "support_forcing_pruning_branch_closed",
            "status": support_closeout.get("status"),
            "decision": support_closeout.get("decision"),
            "claim_status": support_closeout.get("claim_status"),
            "supports_redesign_without_gpu": (
                support_closeout.get("status") == "pass"
                and support_closeout.get("selected_next_step")
                == "select a new local sparse value/support redesign branch with stronger flat-value controls before any backend validation"
                and support_closeout.get("requires_gpu_now") is False
                and support_closeout.get("advance_to_gpu_validation") is False
            ),
            "observed": {
                "learned_r2": support_evidence.get("learned_r2"),
                "flat_r2": support_evidence.get("flat_r2"),
                "oracle_r2": support_evidence.get("oracle_r2"),
                "sparse_specific_gate_passed": support_evidence.get("sparse_specific_gate_passed"),
            },
        },
        {
            "signal": "low_churn_sparse_factorization_consumed",
            "status": sparse_factor.get("status"),
            "decision": sparse_factor.get("decision"),
            "claim_status": sparse_factor.get("claim_status"),
            "supports_no_factorization_reopen": (
                sparse_factor.get("status") == "pass"
                and sparse_factor.get("selected_next_action") == "design_value_dictionary_capacity_rescue_before_gpu"
                and sparse_factor.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": sparse_factor.get("selected_next_action")},
        },
        {
            "signal": "target_aware_value_dictionary_consumed",
            "status": value_closeout.get("status"),
            "decision": value_closeout.get("decision"),
            "claim_status": value_closeout.get("claim_status"),
            "supports_no_value_dictionary_reopen": (
                value_closeout.get("status") == "pass"
                and value_closeout.get("selected_next_action")
                == "select_next_post_value_dictionary_local_branch_before_gpu"
                and value_closeout.get("requires_gpu_now") is False
                and value_closeout.get("advance_to_gpu_validation") is False
            ),
            "observed": {"selected_next_action": value_closeout.get("selected_next_action")},
        },
        {
            "signal": "post_value_selector_already_requests_strategy_refresh",
            "status": post_value.get("status"),
            "decision": post_value.get("decision"),
            "claim_status": post_value.get("claim_status"),
            "supports_strategy_refresh": (
                post_value.get("status") == "pass"
                and post_value.get("selected_next_action")
                == "request_strategy_review_before_new_post_value_dictionary_branch"
                and post_value.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": post_value.get("selected_next_action")},
        },
        {
            "signal": "mechanism_inventory_all_local_gates_closed",
            "status": inventory.get("status"),
            "decision": inventory.get("decision"),
            "claim_status": inventory.get("claim_status"),
            "supports_strategy_refresh": (
                inventory.get("status") == "pass"
                and inventory.get("selected_next_action") == "request_strategy_review_before_new_mechanism_branch"
                and inventory.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": inventory.get("selected_next_action")},
        },
        {
            "signal": "external_strategy_review_read",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_no_gpu": "gpu" not in str(strategy["recommended_next_action"]).lower()
            or "no gpu" in str(strategy["recommended_next_action"]).lower(),
            "observed": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
            },
        },
    ]


def _candidate_actions(
    decision_matrix: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required source artifacts are missing",
                "repair or regenerate the missing local reports",
                "source_artifact_repair_required",
            )
        ]

    support_closed = _signal(decision_matrix, "support_forcing_pruning_branch_closed").get(
        "supports_redesign_without_gpu"
    )
    factor_consumed = _signal(decision_matrix, "low_churn_sparse_factorization_consumed").get(
        "supports_no_factorization_reopen"
    )
    value_consumed = _signal(decision_matrix, "target_aware_value_dictionary_consumed").get(
        "supports_no_value_dictionary_reopen"
    )
    post_value_refresh = _signal(decision_matrix, "post_value_selector_already_requests_strategy_refresh").get(
        "supports_strategy_refresh"
    )
    inventory_refresh = _signal(decision_matrix, "mechanism_inventory_all_local_gates_closed").get(
        "supports_strategy_refresh"
    )

    if support_closed and factor_consumed and value_consumed and post_value_refresh and inventory_refresh:
        return [
            _candidate(
                STRATEGY_REFRESH_ACTION,
                "selected",
                (
                    "support-forcing/pruning closed the current sparse-specific claim, and the already-run "
                    "low-churn sparse-factorization plus target-aware value-dictionary redesigns are also "
                    "closed locally; opening another value/support mechanism now would duplicate consumed "
                    "branches without fresh strategy input"
                ),
                "run a fresh external strategy review before opening another sparse value/support redesign or GPU validation",
                "all_current_sparse_value_support_redesigns_closed_strategy_refresh_selected",
            ),
            _candidate(
                LOW_CHURN_FACTOR_ACTION,
                "rejected",
                "the low-churn sparse-factorization ceiling branch already closed and redirected to value-dictionary rescue",
                "do not reopen this branch without new external direction or new source evidence",
                "already_consumed",
            ),
            _candidate(
                VALUE_DICTIONARY_ACTION,
                "rejected",
                "the target-aware value-dictionary rescue already closed and selected post-value strategy refresh",
                "do not extend this branch without new external direction or new source evidence",
                "already_consumed",
            ),
            _candidate(
                GPU_ACTION,
                "rejected",
                "no local sparse value/support redesign has beaten flat-value controls or enabled promotion",
                "do not run Colab or RunPod validation",
                "gpu_blocked_by_local_gates",
            ),
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "source artifacts are present but do not encode the expected closed local redesign state",
            "inspect source reports and regenerate the stale selector inputs",
            "unexpected_source_state",
        )
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row["source"],
            "path": row["path"],
            "reason": "missing required source artifact",
        }
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _signal(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("signal") == name:
            return row
    return {}


def _candidate(
    action: str,
    disposition: str,
    reason: str,
    next_step: str,
    claim_status: str,
) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": "none",
            "notify_ben": False,
            "verdict": "",
            "recommended_next_action": "",
        }
    header: dict[str, Any] = {
        "present": True,
        "strategic_change_level": "none",
        "notify_ben": False,
        "verdict": "",
        "recommended_next_action": "",
    }
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "notify_ben":
            header[key] = value.lower() == "true"
        elif key in {"strategic_change_level", "verdict", "recommended_next_action"}:
            header[key] = value
    return header


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; the selector used local source artifacts only."
    return (
        "Read the latest GPT-5.5-Pro review and kept its dense-teacher/control recommendation accepted. "
        "Because the local dense-teacher sparse value/support redesign chain is now closed, this selector "
        "does not invent another branch or launch GPU validation."
    )


def _direction_shift(strategy: dict[str, Any]) -> dict[str, Any]:
    ben_notify = bool(strategy["notify_ben"] or strategy["strategic_change_level"] == "major")
    return {
        "strategic_change_level": strategy["strategic_change_level"],
        "ben_should_be_notified": ben_notify,
        "recommendation_disposition": "accepted" if strategy["present"] else "not_applicable",
        "direction": "all current local sparse value/support redesign branches are closed; request strategy refresh before a new branch",
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
    selected_action = selected[0]["candidate_action"] if selected else summary["selected_next_action"]
    text = f"""# Post Support-Forcing Sparse Redesign Branch Selector

- Status: {summary["status"]}
- Decision: {summary["decision"]}
- Claim status: {summary["claim_status"]}
- Selected action: {selected_action}
- Selected next step: {summary["selected_next_step"]}
- GPU validation remains blocked: requires_gpu_now=false, advance_to_gpu_validation=false, promotion_allowed=false.
- Strategy review handling: {summary["strategy_review_handling"]}

## Rationale

{summary["rationale"]}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_support_forcing_sparse_redesign_branch_selector(out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
