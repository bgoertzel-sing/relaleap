"""Select the next step after the value-dictionary rescue closes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_VALUE_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_value_dictionary_capacity_rescue_closeout/summary.json"
)
DEFAULT_SPARSE_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_sparse_factorization_ceiling_closeout/summary.json"
)
DEFAULT_CONTEXT_CLOSEOUT = Path("results/reports/context_contrastive_core_periphery_closeout/summary.json")
DEFAULT_MECHANISM_INVENTORY = Path("results/reports/mechanism_source_inventory/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_value_dictionary_branch_selector")

REPAIR_ACTION = "repair_post_value_dictionary_branch_selector_sources"
STRATEGY_REFRESH_ACTION = "request_strategy_review_before_new_post_value_dictionary_branch"
PREFIX_SAFE_DISTILLER_ACTION = "design_train_only_prefix_safe_sparse_distiller_pregate"
RUNPOD_ACTION = "run_runpod_value_dictionary_or_sparse_validation"
TARGET_AWARE_EXTENSION_ACTION = "extend_target_aware_value_dictionary_rescue"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_value_dictionary_branch_selector(
    *,
    value_closeout_path: Path = DEFAULT_VALUE_CLOSEOUT,
    sparse_closeout_path: Path = DEFAULT_SPARSE_CLOSEOUT,
    context_closeout_path: Path = DEFAULT_CONTEXT_CLOSEOUT,
    mechanism_inventory_path: Path = DEFAULT_MECHANISM_INVENTORY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose one bounded no-GPU step after the target-aware dictionary branch."""

    start = time.time()
    value_closeout = _read_json(value_closeout_path)
    sparse_closeout = _read_json(sparse_closeout_path)
    context_closeout = _read_json(context_closeout_path)
    mechanism_inventory = _read_json(mechanism_inventory_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("value_dictionary_capacity_rescue_closeout", value_closeout_path, value_closeout),
        _source_row("sparse_factorization_ceiling_closeout", sparse_closeout_path, sparse_closeout),
        _source_row("context_contrastive_core_periphery_closeout", context_closeout_path, context_closeout),
        _source_row("mechanism_source_inventory", mechanism_inventory_path, mechanism_inventory),
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
    decision_matrix = _decision_matrix(
        value_closeout,
        sparse_closeout,
        context_closeout,
        mechanism_inventory,
        strategy,
    )
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_value_dictionary_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate missing post-value-dictionary source artifacts"
        claim_status = "post_value_dictionary_sources_incomplete"
        rationale = "The selector cannot choose a bounded next step until required local source reports are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_value_dictionary_branch_selected"
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
        "backend_policy": "local branch selection only; RunPod and Colab remain blocked until a fresh local mechanism gate passes",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "failures": failures,
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


def _decision_matrix(
    value_closeout: dict[str, Any],
    sparse_closeout: dict[str, Any],
    context_closeout: dict[str, Any],
    mechanism_inventory: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    review_action = str(strategy["recommended_next_action"]).lower()
    return [
        {
            "signal": "value_dictionary_rescue_closed",
            "status": value_closeout.get("status"),
            "decision": value_closeout.get("decision"),
            "claim_status": value_closeout.get("claim_status"),
            "supports_no_gpu": (
                value_closeout.get("status") == "pass"
                and value_closeout.get("selected_next_action")
                == "select_next_post_value_dictionary_local_branch_before_gpu"
                and value_closeout.get("requires_gpu_now") is False
                and value_closeout.get("advance_to_gpu_validation") is False
            ),
            "observed": {
                "selected_next_action": value_closeout.get("selected_next_action"),
                "promotion_allowed": value_closeout.get("promotion_allowed"),
            },
        },
        {
            "signal": "sparse_factorization_ceiling_closed",
            "status": sparse_closeout.get("status"),
            "decision": sparse_closeout.get("decision"),
            "claim_status": sparse_closeout.get("claim_status"),
            "supports_no_dictionary_reopen": (
                sparse_closeout.get("status") == "pass"
                and sparse_closeout.get("selected_next_action")
                == "design_value_dictionary_capacity_rescue_before_gpu"
                and sparse_closeout.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": sparse_closeout.get("selected_next_action")},
        },
        {
            "signal": "context_contrastive_branch_consumed",
            "status": context_closeout.get("status"),
            "decision": context_closeout.get("decision"),
            "claim_status": context_closeout.get("claim_status"),
            "supports_no_context_duplicate": (
                context_closeout.get("status") == "pass"
                and context_closeout.get("selected_next_action")
                == "design_low_churn_mlp_sparse_factorization_ceiling"
                and context_closeout.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": context_closeout.get("selected_next_action")},
        },
        {
            "signal": "mechanism_inventory_strategy_needed",
            "status": mechanism_inventory.get("status"),
            "decision": mechanism_inventory.get("decision"),
            "claim_status": mechanism_inventory.get("claim_status"),
            "supports_strategy_refresh": (
                mechanism_inventory.get("status") == "pass"
                and mechanism_inventory.get("selected_next_action")
                == "request_strategy_review_before_new_mechanism_branch"
                and mechanism_inventory.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_action": mechanism_inventory.get("selected_next_action")},
        },
        {
            "signal": "external_strategy_review_scope",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_refresh_before_new_architecture": (
                not strategy["present"]
                or "pregate" in review_action
                or "null" in review_action
                or "close" in review_action
            ),
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

    value_closed = _signal(decision_matrix, "value_dictionary_rescue_closed").get("supports_no_gpu")
    sparse_closed = _signal(decision_matrix, "sparse_factorization_ceiling_closed").get(
        "supports_no_dictionary_reopen"
    )
    context_consumed = _signal(decision_matrix, "context_contrastive_branch_consumed").get(
        "supports_no_context_duplicate"
    )
    inventory_needs_strategy = _signal(decision_matrix, "mechanism_inventory_strategy_needed").get(
        "supports_strategy_refresh"
    )
    review_scope_stale = _signal(decision_matrix, "external_strategy_review_scope").get(
        "supports_refresh_before_new_architecture"
    )

    if value_closed and sparse_closed and context_consumed and inventory_needs_strategy and review_scope_stale:
        return [
            _candidate(
                STRATEGY_REFRESH_ACTION,
                "selected",
                (
                    "the target-aware value-dictionary rescue, the sparse-factorization ceiling, and the "
                    "context-contrastive branch are all closed locally; the remaining inventory says a fresh "
                    "strategy review is needed before opening another non-duplicative mechanism branch"
                ),
                "run a fresh external strategy review before choosing any new local mechanism branch or GPU validation",
                "post_value_dictionary_all_local_gates_closed_strategy_refresh_selected",
            ),
            _candidate(
                PREFIX_SAFE_DISTILLER_ACTION,
                "deferred",
                "a train-only prefix-safe sparse distiller is plausible but would be a new branch after all local gates closed",
                "consider only after a fresh strategy review or explicit Ben direction selects it",
                "deferred_pending_strategy_refresh",
            ),
            _candidate(
                RUNPOD_ACTION,
                "rejected",
                "no local mechanism gate passed; GPU validation would only repeat nondeployable or closed ceilings",
                "keep RunPod and Colab unused",
                "gpu_validation_blocked",
            ),
            _candidate(
                TARGET_AWARE_EXTENSION_ACTION,
                "rejected",
                "the value-dictionary closeout already closed target-aware vector quantization after a valid-null tie",
                "do not extend target-aware dictionary capacity without a deployable train-only design",
                "target_aware_dictionary_extension_closed",
            ),
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "source artifacts are present but do not encode the expected closed post-value-dictionary state",
            "inspect source reports and rerun the stale selector inputs",
            "branch_state_inconsistent",
        )
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status") if payload else "missing",
        "decision": payload.get("decision") if payload else "",
        "claim_status": payload.get("claim_status") if payload else "",
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _signal(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == name), {})


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    fields: dict[str, Any] = {
        "present": bool(text),
        "strategic_change_level": "unknown",
        "notify_ben": "unknown",
        "recommended_next_action": "",
        "verdict": "",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in fields:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true"
        or fields["strategic_change_level"] == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy.get("ben_notification_required"):
        return (
            "Read the latest external review; it requests Ben notification or a major shift. "
            "This selector records that condition while keeping GPU/default changes blocked."
        )
    return (
        "Read the latest external review. Its prior no-RunPod/null-audit advice has been satisfied "
        "by the pregate and closeout, so this selector defers any new architecture to a fresh review."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post Value-Dictionary Branch Selector",
            "",
            f"- Status: {summary['status']}",
            f"- Decision: {summary['decision']}",
            f"- Claim status: {summary['claim_status']}",
            f"- Selected next action: {summary['selected_next_action']}",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: requires_gpu_now=false, promotion_allowed=false, advance_to_gpu_validation=false.",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
            "## Rationale",
            "",
            str(summary["rationale"]),
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-closeout", type=Path, default=DEFAULT_VALUE_CLOSEOUT)
    parser.add_argument("--sparse-closeout", type=Path, default=DEFAULT_SPARSE_CLOSEOUT)
    parser.add_argument("--context-closeout", type=Path, default=DEFAULT_CONTEXT_CLOSEOUT)
    parser.add_argument("--mechanism-inventory", type=Path, default=DEFAULT_MECHANISM_INVENTORY)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_post_value_dictionary_branch_selector(
        value_closeout_path=args.value_closeout,
        sparse_closeout_path=args.sparse_closeout,
        context_closeout_path=args.context_closeout,
        mechanism_inventory_path=args.mechanism_inventory,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
