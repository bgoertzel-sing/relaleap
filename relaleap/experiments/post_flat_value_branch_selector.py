"""Select the next local branch after sparse and flat value branches close."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LEARNED_ROUTER_CLOSEOUT = Path("results/reports/learned_router_sparse_value_closeout/summary.json")
DEFAULT_FLAT_VALUE_CLOSEOUT = Path(
    "results/reports/same_router_flat_value_commutator_mitigation_closeout/summary.json"
)
DEFAULT_CORE_CLOSEOUT = Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")
DEFAULT_DENSE_TEACHER_CONTROL = Path("results/reports/dense_teacher_control_mechanism_assay/summary.json")
DEFAULT_MECHANISM_FACTOR_REPEAT = Path("results/reports/mechanism_factorized_continual_learning_repeat/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_flat_value_branch_selector")

DENSE_TEACHER_DISTILLATION_ACTION = "run_local_dense_teacher_residual_distillation_comparison"
CONTRASTIVE_CORE_PERIPHERY_ACTION = "design_new_contrastive_core_periphery_mechanism_before_gpu"
REPAIR_ACTION = "repair_post_flat_value_branch_selector_sources"
REOPEN_FLAT_VALUE_ACTION = "reopen_flat_value_capacity_only_with_new_commutator_signal"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_post_flat_value_branch_selector(
    *,
    learned_router_closeout_path: Path = DEFAULT_LEARNED_ROUTER_CLOSEOUT,
    flat_value_closeout_path: Path = DEFAULT_FLAT_VALUE_CLOSEOUT,
    core_closeout_path: Path = DEFAULT_CORE_CLOSEOUT,
    dense_teacher_control_path: Path = DEFAULT_DENSE_TEACHER_CONTROL,
    mechanism_factor_repeat_path: Path = DEFAULT_MECHANISM_FACTOR_REPEAT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose one bounded local mechanism path after local value branches close."""

    start = time.time()
    learned = _read_json(learned_router_closeout_path)
    flat = _read_json(flat_value_closeout_path)
    core = _read_json(core_closeout_path)
    dense = _read_json(dense_teacher_control_path)
    factorized = _read_json(mechanism_factor_repeat_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("learned_router_sparse_value_closeout", learned_router_closeout_path, learned),
        _source_row("same_router_flat_value_commutator_mitigation_closeout", flat_value_closeout_path, flat),
        _source_row("core_periphery_negative_evidence_closeout", core_closeout_path, core),
        _source_row("dense_teacher_control_mechanism_assay", dense_teacher_control_path, dense),
        _source_row("mechanism_factorized_continual_learning_repeat", mechanism_factor_repeat_path, factorized),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    decision_matrix = _decision_matrix(learned, flat, core, dense, factorized, strategy)
    failures = _source_failures(source_rows)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_flat_value_branch_selector_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing post-flat-value branch-selector source artifacts"
        claim_status = "post_flat_value_branch_sources_incomplete"
        rationale = "The selector cannot choose a branch until required local source artifacts are present."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_flat_value_branch_selected"
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
        "backend_policy": "local branch selection only; RunPod and Colab remain blocked until a local mechanism gate passes",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy, selected_next_action),
        "failures": failures,
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
    learned: dict[str, Any],
    flat: dict[str, Any],
    core: dict[str, Any],
    dense: dict[str, Any],
    factorized: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "signal": "learned_router_sparse_values_closed",
            "status": learned.get("status"),
            "decision": learned.get("decision"),
            "claim_status": learned.get("claim_status"),
            "supports_redirect": (
                learned.get("status") == "pass"
                and learned.get("decision") == "learned_router_sparse_value_branch_closed"
                and learned.get("advance_to_gpu_validation") is False
            ),
            "observed": {"selected_next_action": learned.get("selected_next_action")},
        },
        {
            "signal": "flat_value_capacity_closed_as_generic_capacity",
            "status": flat.get("status"),
            "decision": flat.get("decision"),
            "claim_status": flat.get("claim_status"),
            "supports_redirect": (
                flat.get("status") == "pass"
                and flat.get("selected_next_action") == "close_flat_value_capacity_as_generic_capacity_before_gpu"
                and flat.get("advance_to_gpu_validation") is False
            ),
            "observed": {"selected_next_action": flat.get("selected_next_action")},
        },
        {
            "signal": "core_periphery_current_attempt_demoted",
            "status": core.get("status"),
            "decision": core.get("decision"),
            "claim_status": core.get("claim_status"),
            "supports_new_core_attempt_only_as_design": (
                core.get("status") == "pass"
                and core.get("selected_next_action") == "demote_current_core_periphery_mechanism_to_diagnostic_status"
                and core.get("requires_gpu_now") is False
            ),
            "observed": {"next_step": core.get("next_step")},
        },
        {
            "signal": "dense_teacher_control_blocked_but_distillation_comparison_available",
            "status": dense.get("status"),
            "decision": dense.get("decision"),
            "claim_status": dense.get("claim_status"),
            "supports_dense_teacher_local_comparison": (
                dense.get("status") == "pass"
                and dense.get("scientific_gate") == "blocked"
                and dense.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_step": dense.get("selected_next_step")},
        },
        {
            "signal": "mechanism_factorized_sparse_retention_closed",
            "status": factorized.get("status"),
            "decision": factorized.get("decision"),
            "claim_status": factorized.get("claim_status"),
            "supports_not_reopening_factorized_sparse_now": (
                factorized.get("status") == "pass"
                and factorized.get("claim_status") == "mechanism_factorized_sparse_retention_not_established"
                and factorized.get("requires_gpu_now") is False
            ),
            "observed": {"selected_next_step": factorized.get("selected_next_step")},
        },
        {
            "signal": "external_strategy_review",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_no_gpu": "runpod" in strategy["recommended_next_action"].lower()
            or "gpu" in strategy["recommended_next_action"].lower(),
            "observed": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
            },
        },
    ]


def _candidate_actions(
    decision_matrix: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required branch-selection source artifacts are missing",
                "repair or regenerate the missing local source reports",
                "source_artifact_repair_required",
            )
        ]

    learned_closed = _supports(decision_matrix, "learned_router_sparse_values_closed")
    flat_closed = _supports(decision_matrix, "flat_value_capacity_closed_as_generic_capacity")
    core_demoted = _signal(decision_matrix, "core_periphery_current_attempt_demoted").get(
        "supports_new_core_attempt_only_as_design"
    )
    dense_available = _signal(
        decision_matrix,
        "dense_teacher_control_blocked_but_distillation_comparison_available",
    ).get("supports_dense_teacher_local_comparison")
    factorized_closed = _signal(decision_matrix, "mechanism_factorized_sparse_retention_closed").get(
        "supports_not_reopening_factorized_sparse_now"
    )

    if learned_closed and flat_closed and dense_available and factorized_closed:
        return [
            _candidate(
                DENSE_TEACHER_DISTILLATION_ACTION,
                "selected",
                (
                    "learned sparse values and same-router flat values are closed before GPU, the existing "
                    "core/periphery and mechanism-factorized sparse attempts already have negative closeouts, "
                    "and the dense-teacher residual distillation comparison is the unrun local mechanism test "
                    "with explicit sparse, dense, router, null, norm, and teacher-scale controls"
                ),
                (
                    "run the bounded local dense-teacher residual distillation comparison and evaluate its "
                    "fail-closed control gates before any backend validation"
                ),
                "dense_teacher_residual_distillation_local_comparison_selected_no_gpu",
            ),
            _candidate(
                CONTRASTIVE_CORE_PERIPHERY_ACTION,
                "deferred",
                (
                    "Ben's core/periphery direction remains scientifically relevant, but the current "
                    "core/periphery mechanism was already demoted; a new contrastive design should wait "
                    "until the dense-teacher residual comparison clarifies the value/residual target"
                ),
                "reopen only with a materially new contrastive core/periphery design and preregistered controls",
                "core_periphery_repair_deferred_after_negative_closeout",
            ),
            _candidate(
                REOPEN_FLAT_VALUE_ACTION,
                "rejected",
                "flat-value capacity is closed as generic capacity without a passing commutator signal",
                "only reopen after new direct order-averaged or norm-clipped commutator evidence",
                "flat_value_reopen_requires_new_commutator_signal",
            ),
        ]

    if learned_closed and flat_closed and core_demoted:
        return [
            _candidate(
                CONTRASTIVE_CORE_PERIPHERY_ACTION,
                "selected",
                "dense-teacher comparison sources are not coherent, leaving only a local design-level sparse mechanism redirect",
                "design a new contrastive core/periphery mechanism before any GPU validation",
                "contrastive_core_periphery_design_selected_due_to_dense_source_gap",
            )
        ]

    return [
        _candidate(
            REPAIR_ACTION,
            "selected",
            "the local closeout signals do not yet support a coherent post-flat-value branch",
            "repair or regenerate the local closeout source reports",
            "post_flat_value_evidence_incomplete",
        )
    ]


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _supports(rows: list[dict[str, Any]], signal: str) -> bool:
    row = _signal(rows, signal)
    return bool(row.get("supports_redirect"))


def _signal(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == signal), {})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "strategic_change_level": "minor",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify_ben = header.get("notify_ben", "false").lower() == "true"
    level = header.get("strategic_change_level", "minor")
    return {
        "present": True,
        "strategic_change_level": level,
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "ben_notification_required": notify_ben or level == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any], selected_next_action: str) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; used local closeout artifacts only."
    if strategy["ben_notification_required"]:
        return (
            "Read the external review and recorded its Ben-notification flag. This selector remains local "
            f"and selected `{selected_next_action}` without GPU validation."
        )
    return (
        "Read the external review. Its no-RunPod hidden-classifier recommendation remains satisfied by "
        "prior fail-closed artifacts; this selector chooses the next downstream local branch."
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    notes = [
        "# Post-Flat-Value Branch Selector",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Rationale: {summary['rationale']}",
        "",
        "GPU validation remains blocked. This report only selects one local mechanism path after sparse learned values and flat-value capacity failed local gates.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-router-closeout", type=Path, default=DEFAULT_LEARNED_ROUTER_CLOSEOUT)
    parser.add_argument("--flat-value-closeout", type=Path, default=DEFAULT_FLAT_VALUE_CLOSEOUT)
    parser.add_argument("--core-closeout", type=Path, default=DEFAULT_CORE_CLOSEOUT)
    parser.add_argument("--dense-teacher-control", type=Path, default=DEFAULT_DENSE_TEACHER_CONTROL)
    parser.add_argument("--mechanism-factor-repeat", type=Path, default=DEFAULT_MECHANISM_FACTOR_REPEAT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_post_flat_value_branch_selector(
        learned_router_closeout_path=args.learned_router_closeout,
        flat_value_closeout_path=args.flat_value_closeout,
        core_closeout_path=args.core_closeout,
        dense_teacher_control_path=args.dense_teacher_control,
        mechanism_factor_repeat_path=args.mechanism_factor_repeat,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
