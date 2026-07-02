"""Close out explicit order averaging for promoted top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ORDER_AVERAGING_PROBE = Path(
    "results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json"
)
DEFAULT_INVENTORY = Path("results/reports/commutator_dense_teacher_source_inventory/summary.json")
DEFAULT_MULTISITE_CLOSEOUT = Path("results/reports/multisite_continual_pc_core_periphery_closeout/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/token_larger_promoted_topk2_order_averaging_closeout")

REPAIR_ACTION = "repair_order_averaging_closeout_sources"
CLOSE_ACTION = "close_order_averaging_before_gpu"
REQUEST_SELECTOR_ACTION = "request_strategy_or_selector_for_deployable_non_router_mechanism"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_promoted_topk2_order_averaging_closeout(
    *,
    order_averaging_probe_path: Path = DEFAULT_ORDER_AVERAGING_PROBE,
    inventory_path: Path = DEFAULT_INVENTORY,
    multisite_closeout_path: Path = DEFAULT_MULTISITE_CLOSEOUT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record that order averaging is diagnostic only and block GPU."""

    start = time.time()
    probe = _read_json(order_averaging_probe_path)
    inventory = _read_json(inventory_path)
    multisite = _read_json(multisite_closeout_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("explicit_order_averaging_mitigation_probe", order_averaging_probe_path, probe),
        _source_row("commutator_dense_teacher_source_inventory", inventory_path, inventory),
        _source_row("multisite_pc_core_periphery_closeout", multisite_closeout_path, multisite),
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
    evidence = _evidence(probe, inventory, multisite, strategy)
    failures = _failures(source_rows, evidence)
    closeout_rows = _closeout_rows(evidence)
    candidate_actions = _candidate_actions(failures, evidence)
    selected = [row for row in candidate_actions if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "promoted_topk2_order_averaging_closeout_failed_closed"
        claim_status = "order_averaging_closeout_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair missing or inconsistent order-averaging closeout sources"
        rationale = "Required source artifacts are missing or inconsistent, so no scientific redirect is selected."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "promoted_topk2_order_averaging_closed_no_gpu"
        claim_status = selected_row["claim_status"]
        selected_next_action = selected_row["candidate_action"]
        selected_next_step = selected_row["next_step"]
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
        "backend_policy": "local closeout only; RunPod and Colab remain blocked",
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
    inventory: dict[str, Any],
    multisite: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    probe_evidence = probe.get("evidence", {})
    gates = {row.get("gate"): row.get("passes") for row in probe.get("gate_rows", [])}
    return {
        "probe_status": probe.get("status"),
        "probe_decision": probe.get("decision"),
        "probe_selected_next_action": probe.get("selected_next_action"),
        "probe_requires_gpu_now": probe.get("requires_gpu_now"),
        "probe_advance_to_gpu_validation": probe.get("advance_to_gpu_validation"),
        "probe_promotion_allowed": probe.get("promotion_allowed"),
        "order_average_ratio": probe_evidence.get(
            "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio"
        ),
        "order_average_ce_delta_vs_best": probe_evidence.get(
            "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order"
        ),
        "flat_value_order_averaging_control_present": gates.get(
            "flat_value_order_averaging_control_present"
        ),
        "promotion_or_gpu_allowed_gate": gates.get("promotion_or_gpu_allowed"),
        "inventory_selected_next_action": inventory.get("selected_next_action"),
        "inventory_claim_status": inventory.get("claim_status"),
        "multisite_selected_next_action": multisite.get("selected_next_action"),
        "multisite_claim_status": multisite.get("claim_status"),
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:2]:
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
        "probe_status": "pass",
        "probe_decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
        "probe_selected_next_action": "record_order_averaging_matched_control_closeout_no_gpu",
        "probe_requires_gpu_now": False,
        "probe_advance_to_gpu_validation": False,
        "probe_promotion_allowed": False,
        "inventory_selected_next_action": "run_explicit_order_averaging_mitigation_probe_locally",
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    return failures


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "explicit_forward_reverse_order_averaging",
            "disposition": "closed_as_nondeployable_diagnostic",
            "reason": "uses both update orders and therefore is not a deployable finite-update rule",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "promoted_contextual_topk2_causal_cooperation",
            "disposition": "not_supported",
            "reason": "flat-value order-averaging control is missing and sparse-specific mechanism evidence fails closed",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "gpu_validation",
            "disposition": "blocked",
            "reason": "local order-averaging closeout does not permit RunPod or Colab validation",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        {
            "branch": "next_deployable_non_router_mechanism",
            "disposition": "selector_required",
            "reason": "current selectors culminated in order averaging, and that selected branch is now closed as diagnostic only",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    ]


def _candidate_actions(
    failures: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, str]]:
    if failures:
        return [
            _candidate(
                REPAIR_ACTION,
                "selected",
                "required closeout sources are missing or inconsistent",
                "repair order-averaging closeout source artifacts",
                "source_repair_required",
            )
        ]
    return [
        _candidate(
            CLOSE_ACTION,
            "selected",
            (
                "explicit order averaging clears only a diagnostic commutator proxy; because it is nondeployable, "
                "lacks a flat-value order-averaging control, and blocks GPU/promotion, record the closeout now"
            ),
            (
                "after this closeout, use a fresh strategy review or command-generated selector to choose exactly "
                "one deployable non-router mechanism branch before any GPU validation"
            ),
            "order_averaging_closed_selector_required_for_next_deployable_mechanism",
        ),
        _candidate(
            REQUEST_SELECTOR_ACTION,
            "next_after_closeout",
            "no current selector-backed deployable non-router branch remains after the selected order-averaging branch is closed",
            "request or run a selector for the next deployable non-router mechanism branch",
            "next_branch_selector_required",
        ),
        _candidate(
            "launch_gpu_validation_for_order_averaging",
            "rejected",
            "local artifacts explicitly set requires_gpu_now=false and advance_to_gpu_validation=false",
            "do not use RunPod or Colab for this branch",
            "gpu_validation_blocked",
        ),
        _candidate(
            "promote_topk2_causal_cooperation",
            "rejected",
            "matched-control evidence does not establish sparse-specific causal cooperation",
            "do not promote top-k-2 causal cooperation from order-averaging evidence",
            "promotion_blocked",
        ),
    ]


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


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
        "selected_next_action": packet.get("selected_next_action"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "verdict": None,
            "recommended_next_action": None,
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:16]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "verdict": header.get("verdict"),
        "recommended_next_action": header.get("recommended_next_action"),
        "ben_notification_required": bool(notify_ben)
        or header.get("strategic_change_level") == "major",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external review was present; closeout used command-generated artifacts only."
    if strategy["ben_notification_required"]:
        return (
            "Read the external review and preserved its notify/major flag; this closeout still blocks GPU and "
            "records that Ben should be notified before a direction shift."
        )
    return (
        "Read the external review. Its recommendation to run the local explicit order-averaging probe has now "
        "been completed and closed as diagnostic-only; no recommendation is rejected."
    )


def _read_json(path: Path) -> dict[str, Any]:
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


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Promoted Top-k-2 Order-Averaging Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Selected next step: {summary['selected_next_step']}",
        f"- GPU validation remains blocked: `{not summary['advance_to_gpu_validation']}`",
        "",
        "## Evidence",
        "",
        f"- Order-average ratio: `{evidence['order_average_ratio']}`",
        f"- Order-average CE delta versus best order: `{evidence['order_average_ce_delta_vs_best']}`",
        f"- Flat-value order-averaging control present: `{evidence['flat_value_order_averaging_control_present']}`",
        f"- Prior inventory selected: `{evidence['inventory_selected_next_action']}`",
        "",
        "## Interpretation",
        "",
        summary["rationale"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-averaging-probe", type=Path, default=DEFAULT_ORDER_AVERAGING_PROBE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--multisite-closeout", type=Path, default=DEFAULT_MULTISITE_CLOSEOUT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_promoted_topk2_order_averaging_closeout(
        order_averaging_probe_path=args.order_averaging_probe,
        inventory_path=args.inventory,
        multisite_closeout_path=args.multisite_closeout,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
