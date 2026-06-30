"""Close out or redirect the flat-value commutator mitigation branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROBE = Path("results/reports/same_router_flat_value_commutator_mitigation_probe/summary.json")
DEFAULT_VARIANT_ROWS = Path("results/reports/same_router_flat_value_commutator_mitigation_probe/variant_rows.csv")
DEFAULT_GATE_ROWS = Path("results/reports/same_router_flat_value_commutator_mitigation_probe/gate_rows.csv")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_commutator_mitigation_closeout")

REPAIR_ACTION = "repair_flat_value_commutator_mitigation_closeout_sources"
CLOSE_GENERIC_ACTION = "close_flat_value_capacity_as_generic_capacity_before_gpu"
DESIGN_DENSE_TEACHER_ACTION = "design_dense_teacher_residual_distillation_or_core_periphery_branch"
REPEAT_ACTION = "repeat_flat_value_commutator_mitigation_probe_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_same_router_flat_value_commutator_mitigation_closeout(
    *,
    probe_path: Path = DEFAULT_PROBE,
    variant_rows_path: Path = DEFAULT_VARIANT_ROWS,
    gate_rows_path: Path = DEFAULT_GATE_ROWS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the mitigation probe and select one local non-GPU redirect."""

    start = time.time()
    probe = _read_json(probe_path)
    variant_rows = _read_csv(variant_rows_path)
    gate_rows = _read_csv(gate_rows_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("same_router_flat_value_commutator_mitigation_probe", probe_path, probe),
        _source_csv("same_router_flat_value_commutator_mitigation_variant_rows", variant_rows_path, variant_rows),
        _source_csv("same_router_flat_value_commutator_mitigation_gate_rows", gate_rows_path, gate_rows),
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
    evidence = _evidence(probe, variant_rows, gate_rows, strategy)
    failures = _failures(source_rows, evidence)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(evidence, failures)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "flat_value_commutator_mitigation_closeout_failed_closed"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair flat-value commutator mitigation closeout source artifacts"
        claim_status = "source_artifacts_incomplete"
        rationale = "Required probe source artifacts are missing, contradictory, or require Ben notification."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "flat_value_commutator_mitigation_branch_closed_or_redirected"
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked until a new local branch clears null/control and budget gates",
        "source_rows": source_rows,
        "evidence": evidence,
        "closeout_rows": closeout_rows,
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


def _evidence(
    probe: dict[str, Any],
    variant_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    gates = {row.get("gate", ""): _bool_or_none(row.get("passes")) for row in gate_rows}
    measured_variants = [row for row in variant_rows if _bool_or_none(row.get("measured")) is True]
    missing_required = [
        row
        for row in variant_rows
        if _bool_or_none(row.get("required_variant")) is True
        and _bool_or_none(row.get("measured")) is False
    ]
    passing_variants = [
        row
        for row in variant_rows
        if _bool_or_none(row.get("measured")) is True
        and _bool_or_none(row.get("variant_passes")) is True
    ]
    anchor = next(
        (row for row in variant_rows if row.get("variant") == "flat_value_commutator_penalty_probe"),
        {},
    )
    return {
        "probe_status": probe.get("status"),
        "probe_decision": probe.get("decision"),
        "probe_claim_status": probe.get("claim_status"),
        "probe_selected_next_action": probe.get("selected_next_action"),
        "measured_passing_variant_count": _first_int(
            probe.get("measured_passing_variant_count"),
            len(passing_variants),
        ),
        "missing_required_variant_count": _first_int(
            probe.get("missing_required_variant_count"),
            len(missing_required),
        ),
        "measured_variant_count": len(measured_variants),
        "required_variants_missing": bool(missing_required),
        "passing_variants_present": bool(passing_variants),
        "at_least_one_measured_mitigation_variant": gates.get("at_least_one_measured_mitigation_variant"),
        "no_required_variants_missing": gates.get("no_required_variants_missing"),
        "measured_variant_passes_all_gates": gates.get("measured_variant_passes_all_gates"),
        "anchor_proxy_commutator_budget_passes": gates.get("anchor_proxy_commutator_budget_passes"),
        "anchor_commutator_ratio_to_budget": _float_or_none(anchor.get("commutator_ratio_to_budget")),
        "anchor_failure_reasons": anchor.get("failure_reasons", ""),
        "ben_notification_required": strategy["notify_ben"]
        or strategy["strategic_change_level"] == "major",
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "flat_value_commutator_mitigation",
            "source_decision": evidence["probe_decision"],
            "disposition": "closed_before_gpu",
            "reason": (
                "measured anchor proxy fails commutator budget and required order-averaged/norm-clipped variants are missing"
                if evidence["required_variants_missing"]
                else "no measured mitigation clears CE, norm, churn, and commutator gates"
            ),
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "same_router_flat_value_capacity",
            "source_decision": evidence["probe_decision"],
            "disposition": "closed_as_generic_capacity_counterexample",
            "reason": "flat-value capacity improved CE controls but did not establish a low-interference residual mechanism",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "dense_teacher_or_core_periphery_local_design",
            "source_decision": "local_closeout",
            "disposition": "redirect_target",
            "reason": "current sparse/flat value branches are blocked by null, control, or commutator evidence",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "gpu_validation",
            "source_decision": "local_closeout",
            "disposition": "blocked",
            "reason": "no local flat-value commutator mitigation permits GPU validation",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    ]


def _candidate_actions(
    evidence: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required closeout sources are missing, contradictory, or require Ben notification",
                "repair flat-value commutator mitigation closeout source artifacts",
                "source_repair_required",
            )
        ]
    if evidence["passing_variants_present"] and evidence["required_variants_missing"] is False:
        return [
            _candidate(
                REPEAT_ACTION,
                "selected",
                "a measured mitigation variant cleared all local gates and needs a repeat before backend spend",
                "repeat flat-value commutator mitigation on an adjacent local seed before any GPU validation",
                "flat_value_commutator_mitigation_repeat_required",
            )
        ]
    return [
        _candidate(
            CLOSE_GENERIC_ACTION,
            "selected",
            (
                "the flat-value mitigation did not clear the finite-update commutator budget and required direct "
                "mitigation variants are missing, so the flat-value capacity signal should be closed as generic "
                "capacity before GPU validation"
            ),
            "close flat-value capacity as generic capacity and redirect to a new local mechanism branch",
            "flat_value_capacity_closed_as_generic_capacity",
        ),
        _candidate(
            DESIGN_DENSE_TEACHER_ACTION,
            "deferred",
            "the natural next local branch is dense-teacher residual distillation or core/periphery design, but first record this closeout",
            "after closeout, select a dense-teacher residual distillation or core/periphery local design step from status",
            "deferred_next_branch",
        ),
        _candidate(
            REPEAT_ACTION,
            "rejected",
            "repeating the current mitigation would duplicate a branch with no passing measured variant and missing required rows",
            "only reconsider after implementing direct order-averaged or stricter norm-clipped mitigation rows",
            "rejected",
        ),
    ]


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    if evidence["probe_status"] != "pass":
        failures.append({"source": "probe", "reason": "probe_status_not_pass"})
    if evidence["ben_notification_required"]:
        failures.append({"source": "strategy_review", "reason": "ben_notification_required"})
    return failures


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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_json(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing"),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _source_csv(source: str, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(rows),
        "status": "present" if rows else "missing",
        "decision": "",
        "claim_status": "",
        "row_count": len(rows),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "strategic_change_level": "minor",
            "notify_ben": False,
            "recommended_next_action": "",
            "verdict": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return {
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", "minor"),
        "notify_ben": header.get("notify_ben", "false").lower() == "true",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; proceeded with local fail-closed probe artifacts."
    return (
        "Read the latest external review. Its no-RunPod hidden-classifier recommendation remains satisfied; "
        "this closeout records the downstream flat-value commutator mitigation failure without launching GPU validation."
    )


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


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
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    evidence = summary["evidence"]
    notes = [
        "# Flat-Value Commutator Mitigation Closeout",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Measured passing variants: `{evidence['measured_passing_variant_count']}`",
        f"- Missing required variants: `{evidence['missing_required_variant_count']}`",
        f"- Anchor commutator ratio to budget: `{evidence['anchor_commutator_ratio_to_budget']}`",
        f"- Anchor failure reasons: `{evidence['anchor_failure_reasons']}`",
        f"- Rationale: {summary['rationale']}",
        "",
        "GPU validation remains blocked. The flat-value capacity signal is closed as generic capacity unless a new direct mitigation row changes the local commutator evidence.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--variant-rows", type=Path, default=DEFAULT_VARIANT_ROWS)
    parser.add_argument("--gate-rows", type=Path, default=DEFAULT_GATE_ROWS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_commutator_mitigation_closeout(
        probe_path=args.probe,
        variant_rows_path=args.variant_rows,
        gate_rows_path=args.gate_rows,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
