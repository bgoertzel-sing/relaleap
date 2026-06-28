"""Select the local branch after core/periphery and ACSR sparse paths are demoted."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CORE_CLOSEOUT = Path("results/reports/core_periphery_negative_evidence_closeout/summary.json")
DEFAULT_ACSR_SELECTOR = Path("results/reports/acsr_post_negative_branch_selector/summary.json")
DEFAULT_DENSE_PRIMARY = Path("results/reports/dense_primary_mechanism_assay/summary.json")
DEFAULT_CAUSAL_ROUTER_SYNTHESIS = Path(
    "results/reports/token_larger_causal_contextual_router_distillation_synthesis/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/post_core_periphery_contextual_dense_branch_selector")

DENSE_MECHANISM_ACTION = "continue_dense_mlp_mechanism_track_with_causal_router_diagnostics"
REPAIR_SOURCES_ACTION = "repair_post_core_periphery_branch_source_artifacts"
CORE_REPAIR_ACTION = "design_new_contrastive_core_periphery_mechanism_before_gpu"
REOPEN_ACSR_ACTION = "reopen_sparse_acsr_promotion_path"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "candidate_actions.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_post_core_periphery_contextual_dense_branch_selector(
    *,
    core_closeout_path: Path = DEFAULT_CORE_CLOSEOUT,
    acsr_selector_path: Path = DEFAULT_ACSR_SELECTOR,
    dense_primary_path: Path = DEFAULT_DENSE_PRIMARY,
    causal_router_synthesis_path: Path = DEFAULT_CAUSAL_ROUTER_SYNTHESIS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Choose exactly one bounded local branch from the current source artifacts."""

    start = time.time()
    core = _read_json(core_closeout_path)
    acsr = _read_json(acsr_selector_path)
    dense = _read_json(dense_primary_path)
    causal = _read_json(causal_router_synthesis_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("core_periphery_closeout", core_closeout_path, core),
        _source_row("acsr_post_negative_selector", acsr_selector_path, acsr),
        _source_row("dense_primary_mechanism_assay", dense_primary_path, dense),
        _source_row("causal_contextual_router_distillation_synthesis", causal_router_synthesis_path, causal),
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
    evidence = _evidence(core, acsr, dense, causal, strategy)
    criteria = _criteria(source_rows, evidence)
    failures = [row for row in criteria if not row["passed"]]
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "post_core_periphery_branch_selection_failed_closed"
        selected_next_action = REPAIR_SOURCES_ACTION
        next_step = "repair missing or contradictory post-core/periphery source artifacts"
        claim_status = "post_core_periphery_branch_source_evidence_incomplete"
        rationale = "Required local branch-selection evidence is missing or contradictory."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "post_core_periphery_contextual_dense_branch_selected"
        selected_next_action = selected_row["candidate_action"]
        next_step = selected_row["next_step"]
        claim_status = selected_row["claim_status"]
        rationale = selected_row["reason"]

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_action": selected_next_action,
        "next_step": next_step,
        "requires_gpu_now": False,
        "backend_policy": "local branch-selection report only; RunPod remains blocked until a local mechanism gate passes",
        "source_rows": source_rows,
        "evidence": evidence,
        "criteria": criteria,
        "candidate_actions": candidate_actions,
        "strategy_review": strategy,
        "strategy_response": _strategy_response(evidence, strategy),
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


def _evidence(
    core: dict[str, Any],
    acsr: dict[str, Any],
    dense: dict[str, Any],
    causal: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "core_status": core.get("status"),
        "core_selected_next_action": core.get("selected_next_action"),
        "core_claim_status": core.get("claim_status"),
        "core_requires_gpu_now": core.get("requires_gpu_now"),
        "acsr_status": acsr.get("status"),
        "acsr_selected_next_action": acsr.get("selected_next_action"),
        "acsr_dense_controls_active": _nested(acsr, "claim_statuses", "dense_residual_controls"),
        "dense_status": dense.get("status"),
        "dense_primary_arm": dense.get("primary_arm"),
        "dense_primary_family": dense.get("primary_family"),
        "dense_claim_status": dense.get("claim_status"),
        "causal_status": causal.get("status"),
        "causal_claim_status": causal.get("claim_status"),
        "causal_selected_next_step": causal.get("selected_next_step"),
        "strategy_verdict": strategy.get("verdict"),
        "strategy_recommended_next_action": strategy.get("recommended_next_action"),
        "ben_notification_required": strategy.get("ben_notification_required"),
        "strategy_major_pivot": strategy.get("strategic_change_level") == "major",
    }


def _criteria(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    required_present = all(row["present"] for row in source_rows[:4])
    core_demoted = (
        evidence["core_status"] == "pass"
        and evidence["core_selected_next_action"] == "demote_current_core_periphery_mechanism_to_diagnostic_status"
        and evidence["core_requires_gpu_now"] is False
    )
    acsr_retired = (
        evidence["acsr_status"] == "pass"
        and evidence["acsr_selected_next_action"] == "retire_acsr_promotion_in_favor_of_dense_residual_controls"
    )
    dense_selectable = (
        evidence["dense_status"] == "pass"
        and isinstance(evidence["dense_primary_arm"], str)
        and bool(evidence["dense_primary_arm"])
    )
    causal_available = evidence["causal_status"] == "pass"
    return [
        _criterion(
            "required_source_artifacts_present",
            required_present,
            "core, ACSR, dense primary, and causal-router synthesis summaries present",
            ",".join(str(row["present"]) for row in source_rows[:4]),
        ),
        _criterion(
            "core_periphery_current_mechanism_demoted",
            core_demoted,
            "current core/periphery mechanism demoted and GPU blocked",
            f"{evidence['core_status']}; {evidence['core_selected_next_action']}; gpu={evidence['core_requires_gpu_now']}",
        ),
        _criterion(
            "acsr_promotion_path_retired",
            acsr_retired,
            "ACSR promotion path retired in favor of dense residual controls",
            f"{evidence['acsr_status']}; {evidence['acsr_selected_next_action']}",
        ),
        _criterion(
            "dense_primary_assay_selectable",
            dense_selectable,
            "dense/MLP primary mechanism assay passed and selected a primary arm",
            f"{evidence['dense_status']}; {evidence['dense_primary_arm']}",
        ),
        _criterion(
            "causal_router_synthesis_available_as_diagnostic",
            causal_available,
            "causal-router distillation synthesis remains available as diagnostic context, not a default claim",
            f"{evidence['causal_status']}; {evidence['causal_claim_status']}",
        ),
    ]


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_SOURCES_ACTION,
                "selected",
                "required branch-selection sources are missing or inconsistent",
                "repair or rerun the missing local source reports",
                "source_artifact_repair_required",
            ),
            _candidate(
                DENSE_MECHANISM_ACTION,
                "blocked",
                "the dense track cannot be selected until required source gates are coherent",
                "rerun after source repair",
                "source_artifact_repair_required",
            ),
            _candidate(
                CORE_REPAIR_ACTION,
                "blocked",
                "the latest core/periphery status cannot be interpreted from incomplete sources",
                "rerun after source repair",
                "source_artifact_repair_required",
            ),
        ]
    return [
        _candidate(
            DENSE_MECHANISM_ACTION,
            "selected",
            (
                "the current core/periphery mechanism is demoted, ACSR promotion is retired by dense controls, "
                "and the dense primary assay has a local primary arm"
            ),
            (
                "run a bounded local follow-up around the selected dense/MLP primary arm, using causal-router "
                "distillation only as diagnostic context and keeping RunPod blocked"
            ),
            "dense_mlp_mechanism_track_selected_no_gpu_or_default_change",
        ),
        _candidate(
            CORE_REPAIR_ACTION,
            "deferred",
            (
                "the latest review recommends a contrastive periphery repair, but the current closeout already "
                "demoted this mechanism after useful-periphery evidence failed dense/MLP retention and pruning gates"
            ),
            "resume only if Ben asks for another core/periphery mechanism attempt or a new design changes the local gates",
            "core_periphery_repair_deferred_after_negative_closeout",
        ),
        _candidate(
            REOPEN_ACSR_ACTION,
            "rejected",
            "ACSR support identity and dense-teacher compression are not supported against current dense controls",
            "do not run ACSR promotion or GPU validation from this branch",
            "acsr_promotion_path_remains_retired",
        ),
    ]


def _strategy_response(evidence: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_recommendation": strategy.get("recommended_next_action"),
        "deferred_or_rejected": "deferred",
        "reason": (
            "The recommendation is scientifically sensible as a possible future mechanism design, "
            "but it is not the next bounded step because the latest command-driven closeout demoted "
            "the current core/periphery mechanism and selected a dense/control redirect."
        ),
        "ben_should_be_notified": bool(evidence.get("ben_notification_required")),
    }


def _criterion(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
    }


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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision"),
        "claim_status": payload.get("claim_status"),
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        summary["source_rows"],
    )
    _write_csv(
        out_dir / "candidate_actions.csv",
        ["candidate_action", "disposition", "reason", "next_step", "claim_status"],
        summary["candidate_actions"],
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        summary["criteria"],
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post Core/Periphery Contextual Dense Branch Selector",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Next step: {summary['next_step']}",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Ben notification: `{summary['strategy_response']['ben_should_be_notified']}`",
        "",
        summary["rationale"],
        "",
        "Strategy review response: "
        f"{summary['strategy_response']['deferred_or_rejected']} - {summary['strategy_response']['reason']}",
        "",
        "This is a local branch-selection artifact. It does not promote a default router and does not "
        "turn the current core/periphery or ACSR sparse evidence into GPU evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in fields:
                fields[key] = value
    fields["ben_notification_required"] = (
        str(fields.get("notify_ben")).lower() == "true"
        or fields.get("strategic_change_level") == "major"
    )
    return fields


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-closeout", type=Path, default=DEFAULT_CORE_CLOSEOUT)
    parser.add_argument("--acsr-selector", type=Path, default=DEFAULT_ACSR_SELECTOR)
    parser.add_argument("--dense-primary", type=Path, default=DEFAULT_DENSE_PRIMARY)
    parser.add_argument("--causal-router-synthesis", type=Path, default=DEFAULT_CAUSAL_ROUTER_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_post_core_periphery_contextual_dense_branch_selector(
        core_closeout_path=args.core_closeout,
        acsr_selector_path=args.acsr_selector,
        dense_primary_path=args.dense_primary,
        causal_router_synthesis_path=args.causal_router_synthesis,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
