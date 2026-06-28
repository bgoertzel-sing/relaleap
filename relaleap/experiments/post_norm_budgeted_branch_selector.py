"""Select the next local branch after the norm-budgeted sparse branch stops."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_NORM_SYNTHESIS = Path("results/reports/norm_budgeted_churn_strata_synthesis/summary.json")
DEFAULT_CL_REPEAT = Path("results/reports/mechanism_factorized_continual_learning_repeat/summary.json")
DEFAULT_COMMUTATOR = Path("results/reports/acsr_finite_update_commutator_assay/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_norm_budgeted_branch_selector")

TASK_FREE_CL_ACTION = "design_task_free_continual_learning_dense_control_assay"
DENSE_TEACHER_ACTION = "pivot_to_dense_teacher_control_mechanism_assay"
STRONG_COMMUTATOR_ACTION = "run_stronger_bounded_finite_update_commutator_assay"
REPAIR_SOURCES_ACTION = "repair_missing_post_norm_budgeted_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "decision_matrix.csv",
    "notes.md",
)


def run_post_norm_budgeted_branch_selector(
    *,
    norm_synthesis_path: Path = DEFAULT_NORM_SYNTHESIS,
    cl_repeat_path: Path = DEFAULT_CL_REPEAT,
    commutator_path: Path = DEFAULT_COMMUTATOR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose one bounded next local branch after sparse norm-target evidence stops."""

    start = time.time()
    norm_synthesis = _read_json(norm_synthesis_path)
    cl_repeat = _read_json(cl_repeat_path)
    commutator = _read_json(commutator_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("norm_budgeted_churn_strata_synthesis", norm_synthesis_path, norm_synthesis),
        _source_row("mechanism_factorized_cl_repeat", cl_repeat_path, cl_repeat),
        _source_row("finite_update_commutator", commutator_path, commutator),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}",
        },
    ]
    failures = _source_failures(source_rows)
    decision_matrix = _decision_matrix(norm_synthesis, cl_repeat, commutator, strategy)
    candidate_actions = _candidate_actions(decision_matrix, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]
    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_norm_budgeted_branch_selector_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        next_step = "repair missing post-norm-budgeted source artifacts"
        claim_status = "post_norm_budgeted_branch_sources_incomplete"
        rationale = "The selector cannot choose a scientific branch until required local source artifacts are present."
    else:
        status = "pass"
        decision = "post_norm_budgeted_branch_selected"
        selected_next_action = selected[0]["candidate_action"]
        next_step = selected[0]["next_step"]
        claim_status = selected[0]["claim_status"]
        rationale = selected[0]["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "selected_next_step": next_step,
        "claim_status": claim_status,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "backend_policy": "local branch selection only; no RunPod/Colab repeat until a local mechanism assay clears dense/null/churn/norm gates",
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
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
    norm_synthesis: dict[str, Any],
    cl_repeat: dict[str, Any],
    commutator: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    commutator_metrics = _as_dict(commutator.get("metrics"))
    return [
        {
            "signal": "norm_budgeted_sparse_branch",
            "status": norm_synthesis.get("status"),
            "decision": norm_synthesis.get("decision"),
            "claim_status": norm_synthesis.get("claim_status"),
            "supports_continuing_sparse_norm_target": bool(norm_synthesis.get("runpod_repeat_warranted")),
            "observed": norm_synthesis.get("interpretation", ""),
        },
        {
            "signal": "task_free_continual_learning_repeat",
            "status": cl_repeat.get("status"),
            "decision": cl_repeat.get("decision"),
            "claim_status": cl_repeat.get("claim_status"),
            "supports_sparse_retention_branch": cl_repeat.get("topk2_tradeoff_repeat_status") == "survives_second_seed",
            "observed": cl_repeat.get("selected_next_step", ""),
        },
        {
            "signal": "finite_update_commutator",
            "status": commutator.get("status"),
            "decision": commutator.get("decision"),
            "claim_status": commutator.get("claim_status"),
            "supports_stronger_commutator": _float(commutator_metrics.get("sparse_mean_logit_mse")) is not None
            and (_float(commutator_metrics.get("sparse_mean_logit_mse")) or 0.0) >= 0.005,
            "observed": {
                "sparse_mean_logit_mse": commutator_metrics.get("sparse_mean_logit_mse"),
                "dense_mean_logit_mse": commutator_metrics.get("dense_mean_logit_mse"),
                "next_step": commutator.get("next_step", ""),
            },
        },
        {
            "signal": "external_strategy_review",
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": f"verdict={strategy['verdict']}",
            "supports_dense_control_pivot": "dense" in strategy["recommended_next_action"].lower()
            or "mlp" in strategy["recommended_next_action"].lower(),
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
        return [_candidate(REPAIR_SOURCES_ACTION, "selected", "required source artifacts are missing", "repair missing local source artifacts", "incomplete")]

    sparse_norm_continues = bool(_signal(decision_matrix, "norm_budgeted_sparse_branch").get("supports_continuing_sparse_norm_target"))
    cl_supported = bool(_signal(decision_matrix, "task_free_continual_learning_repeat").get("supports_sparse_retention_branch"))
    commutator_supported = bool(_signal(decision_matrix, "finite_update_commutator").get("supports_stronger_commutator"))

    if commutator_supported and not cl_supported:
        return [
            _candidate(
                STRONG_COMMUTATOR_ACTION,
                "selected",
                "the sparse commutator is material enough to make update-order sensitivity the primary bounded question",
                "run one stronger bounded finite-update commutator assay before changing architecture direction",
                "finite_update_order_sensitivity_followup_allowed_local_only",
            ),
            _candidate(TASK_FREE_CL_ACTION, "deferred", "commutator magnitude is the narrower live mechanism question", "revisit after commutator follow-up", "deferred"),
            _candidate(DENSE_TEACHER_ACTION, "deferred", "dense-control pivot waits for commutator clarification", "revisit after commutator follow-up", "deferred"),
        ]

    if cl_supported or sparse_norm_continues:
        return [
            _candidate(
                TASK_FREE_CL_ACTION,
                "selected",
                "some sparse retention/norm signal survived local guards but still needs task-free dense/null controls",
                "design a task-free continual-learning dense-control assay with ACSR retained only as a diagnostic arm",
                "task_free_cl_dense_control_design_allowed_local_only",
            ),
            _candidate(DENSE_TEACHER_ACTION, "deferred", "task-free retention is the more direct unresolved mechanism question", "revisit after task-free CL design", "deferred"),
            _candidate(STRONG_COMMUTATOR_ACTION, "rejected", "commutator evidence is not the active unresolved signal", "only rerun if commutator magnitude becomes primary", "rejected"),
        ]

    return [
        _candidate(
            DENSE_TEACHER_ACTION,
            "selected",
            "sparse norm-target, sparse retention, and finite-update order-sensitivity branches are all locally blocked, so the next coherent mechanism candidate should be dense/MLP control-first",
            "design one dense-teacher or MLP residual-control mechanism assay with matched residual-L2, anchor KL, flip/functional churn, and intervention-fingerprint gates",
            "sparse_branches_locally_blocked_dense_control_pivot_selected",
        ),
        _candidate(TASK_FREE_CL_ACTION, "rejected", "the existing CL repeat did not replicate the sparse top-k2 tradeoff", "only reconsider after new dense/null-controlled CL evidence", "rejected"),
        _candidate(STRONG_COMMUTATOR_ACTION, "rejected", "the existing sparse commutator is too small for a sparse mechanism claim", "only rerun if update-order sensitivity becomes the primary question", "rejected"),
    ]


def _candidate(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _signal(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal") == signal), {})


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _source_failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": row["source"], "field": "source_artifact", "reason": f"{row['path']} is missing"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Post Norm-Budgeted Branch Selector",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            summary["rationale"],
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": "missing",
            "notify_ben": "false",
            "ben_notification_required": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    text = path.read_text(encoding="utf-8")
    header: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    notify = header.get("notify_ben", "false").lower() == "true"
    return {
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", "false"),
        "ben_notification_required": notify or header.get("strategic_change_level", "").lower() == "major",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "missing optional strategy review; no recommendation incorporated"
    if strategy["ben_notification_required"]:
        return "review consumed; direction shift recorded and Ben should be notified"
    return "review consumed; no major direction shift or Ben notification requested"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-synthesis", type=Path, default=DEFAULT_NORM_SYNTHESIS)
    parser.add_argument("--cl-repeat", type=Path, default=DEFAULT_CL_REPEAT)
    parser.add_argument("--commutator", type=Path, default=DEFAULT_COMMUTATOR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_post_norm_budgeted_branch_selector(
        norm_synthesis_path=args.norm_synthesis,
        cl_repeat_path=args.cl_repeat,
        commutator_path=args.commutator,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "selected_next_action": summary["selected_next_action"]}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
