"""Probe local flat-value finite-update commutator mitigations."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DESIGN = Path("results/reports/same_router_flat_value_commutator_mitigation_design/summary.json")
DEFAULT_DIAGNOSTIC = Path("results/reports/same_router_flat_value_capacity_diagnostic")
DEFAULT_SYNTHETIC_DIR = Path("results/reports/synthetic_mechanism_causal_modularity")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/same_router_flat_value_commutator_mitigation_probe")

REPAIR_ACTION = "repair_flat_value_commutator_mitigation_probe_sources"
CLOSE_ACTION = "close_flat_value_commutator_mitigation_before_gpu"
REPEAT_ACTION = "repeat_flat_value_commutator_mitigation_probe_before_gpu"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "variant_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_same_router_flat_value_commutator_mitigation_probe(
    *,
    design_path: Path = DEFAULT_DESIGN,
    diagnostic_dir: Path = DEFAULT_DIAGNOSTIC,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Evaluate measured local mitigation rows and fail closed on missing variants."""

    start = time.time()
    design = _read_json(design_path)
    paths = {
        "diagnostic_budget_rows": diagnostic_dir / "budget_rows.csv",
        "diagnostic_control_rows": diagnostic_dir / "control_rows.csv",
        "diagnostic_gate_rows": diagnostic_dir / "gate_rows.csv",
        "synthetic_arm_metrics": synthetic_dir / "arm_metrics.csv",
        "synthetic_commutator_rows": synthetic_dir / "commutator_rows.csv",
        "synthetic_forgetting_rows": synthetic_dir / "forgetting_rows.csv",
    }
    rows = {name: _read_csv(path) for name, path in paths.items()}
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_json("same_router_flat_value_commutator_mitigation_design", design_path, design),
        *[_source_csv(name, path, rows[name]) for name, path in paths.items()],
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
            "row_count": "",
        },
    ]
    failures = _source_failures(source_rows, design, strategy)
    variant_rows = _variant_rows(
        arm_metrics=rows["synthetic_arm_metrics"],
        commutator_rows=rows["synthetic_commutator_rows"],
        forgetting_rows=rows["synthetic_forgetting_rows"],
        diagnostic_budget_rows=rows["diagnostic_budget_rows"],
        diagnostic_control_rows=rows["diagnostic_control_rows"],
        source_failures=failures,
    )
    gate_rows = _gate_rows(variant_rows, failures)
    measured_passes = [row for row in variant_rows if row.get("measured") is True and row.get("variant_passes") is True]
    missing_required = [row for row in variant_rows if row.get("measured") is False and row.get("required_variant") is True]
    probe_passes = bool(measured_passes) and not missing_required and all(row["passes"] is True for row in gate_rows)
    status = "fail" if failures else "pass"
    decision = (
        "same_router_flat_value_commutator_mitigation_probe_failed_closed"
        if failures
        else "same_router_flat_value_commutator_mitigation_probe_passed_repeat_before_gpu"
        if probe_passes
        else "same_router_flat_value_commutator_mitigation_probe_gpu_blocked"
    )
    selected_next_action = REPAIR_ACTION if failures else REPEAT_ACTION if probe_passes else CLOSE_ACTION
    selected_next_step = (
        "repair flat-value commutator mitigation probe source artifacts"
        if failures
        else "repeat the passing flat-value commutator mitigation on an adjacent local seed before GPU validation"
        if probe_passes
        else "close the current flat-value commutator mitigation branch before GPU; order-averaged and strict norm-clipped rows are still missing and the measured anchor proxy fails the commutator budget"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": (
            "source_artifacts_incomplete"
            if failures
            else "flat_value_commutator_mitigation_signal_needs_repeat"
            if probe_passes
            else "flat_value_commutator_mitigation_not_established"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local probe only; RunPod and Colab remain blocked until a measured mitigation passes all local gates",
        "source_rows": source_rows,
        "variant_rows": variant_rows,
        "gate_rows": gate_rows,
        "measured_passing_variant_count": len(measured_passes),
        "missing_required_variant_count": len(missing_required),
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _variant_rows(
    *,
    arm_metrics: list[dict[str, str]],
    commutator_rows: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    diagnostic_budget_rows: list[dict[str, str]],
    diagnostic_control_rows: list[dict[str, str]],
    source_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if source_failures:
        return [
            {
                "variant": "source_repair",
                "measured": False,
                "required_variant": False,
                "variant_passes": False,
                "failure_reasons": "source_artifacts_incomplete",
                "requires_gpu_now": False,
                "advance_to_gpu_validation": False,
            }
        ]
    by_arm = {row.get("arm", ""): row for row in arm_metrics}
    controls = {row.get("arm", ""): row for row in diagnostic_control_rows}
    budgets = {row.get("budget", ""): row for row in diagnostic_budget_rows}
    reference_ce = _metric(by_arm.get("promoted_contextual_topk2", {}), "holdout_ce")
    token_ce = _metric(by_arm.get("token_position_router_topk2", {}), "holdout_ce")
    dense_ce = _metric(by_arm.get("dense_rank_norm_matched", {}), "holdout_ce")
    low_churn_ce = _metric(by_arm.get("low_churn_mlp_active_matched", {}), "holdout_ce")
    fixed_ce = _metric(by_arm.get("fixed_support_topk2", {}), "holdout_ce")
    random_ce = _metric(by_arm.get("random_support_topk2", {}), "holdout_ce")
    norm_budget = _metric(budgets.get("residual_norm", {}), "reference_budget_value")
    churn_budget = _metric(budgets.get("functional_churn", {}), "reference_budget_value")
    commutator_budget = _metric(budgets.get("finite_update_commutator", {}), "reference_budget_value")

    rows = [
        _measured_variant(
            variant="unmitigated_flat_value",
            arm="flat_column_value_mlp_topk2",
            role="baseline_not_mitigation",
            by_arm=by_arm,
            commutator_rows=commutator_rows,
            forgetting_rows=forgetting_rows,
            reference_ce=reference_ce,
            token_ce=token_ce,
            dense_ce=dense_ce,
            low_churn_ce=low_churn_ce,
            fixed_ce=fixed_ce,
            random_ce=random_ce,
            norm_budget=norm_budget,
            churn_budget=churn_budget,
            commutator_budget=commutator_budget,
            required_variant=False,
        ),
        _measured_variant(
            variant="flat_value_commutator_penalty_probe",
            arm="flat_column_value_mlp_anchor_topk2",
            role="measured_anchor_kl_proxy",
            by_arm=by_arm,
            commutator_rows=commutator_rows,
            forgetting_rows=forgetting_rows,
            reference_ce=reference_ce,
            token_ce=token_ce,
            dense_ce=dense_ce,
            low_churn_ce=low_churn_ce,
            fixed_ce=fixed_ce,
            random_ce=random_ce,
            norm_budget=norm_budget,
            churn_budget=churn_budget,
            commutator_budget=commutator_budget,
            required_variant=True,
        ),
        _missing_variant(
            "flat_value_order_averaged_updates",
            "no direct order-averaged flat-value intervention row exists in the command-generated packet",
        ),
        _missing_variant(
            "flat_value_norm_clipped_updates",
            "no stricter value-norm-clipped flat-value mitigation row exists beyond the already-clipped unmitigated flat arm",
        ),
    ]
    for row in rows:
        row["diagnostic_flat_ce_gain_vs_promoted_sparse"] = _metric(
            controls.get("promoted_contextual_topk2", {}),
            "flat_ce_gain_vs_control",
        )
    return rows


def _measured_variant(
    *,
    variant: str,
    arm: str,
    role: str,
    by_arm: dict[str, dict[str, str]],
    commutator_rows: list[dict[str, str]],
    forgetting_rows: list[dict[str, str]],
    reference_ce: float | None,
    token_ce: float | None,
    dense_ce: float | None,
    low_churn_ce: float | None,
    fixed_ce: float | None,
    random_ce: float | None,
    norm_budget: float | None,
    churn_budget: float | None,
    commutator_budget: float | None,
    required_variant: bool,
) -> dict[str, Any]:
    source = by_arm.get(arm, {})
    ce = _metric(source, "holdout_ce")
    norm = _metric(source, "residual_l2")
    churn = _mean_abs_metric(forgetting_rows, arm, "functional_churn")
    commutator = _mean_metric(commutator_rows, arm, "finite_update_commutator_l2")
    ce_controls_ok = all(
        gain is not None and gain >= -0.005
        for gain in (
            _gain(reference_ce, ce),
            _gain(token_ce, ce),
            _gain(dense_ce, ce),
            _gain(low_churn_ce, ce),
            _gain(fixed_ce, ce),
            _gain(random_ce, ce),
        )
    )
    norm_ok = norm is not None and norm_budget is not None and norm <= norm_budget * 1.10
    churn_ok = churn is not None and churn_budget is not None and churn <= churn_budget * 1.10
    commutator_ok = (
        commutator is not None
        and commutator_budget is not None
        and commutator <= commutator_budget * 1.10
    )
    variant_passes = bool(source) and ce_controls_ok and norm_ok and churn_ok and commutator_ok
    failures = []
    if not source:
        failures.append("missing_measured_arm")
    if source and not ce_controls_ok:
        failures.append("ce_or_control_signal_not_preserved")
    if source and not norm_ok:
        failures.append("residual_norm_budget_failed")
    if source and not churn_ok:
        failures.append("functional_churn_budget_failed")
    if source and not commutator_ok:
        failures.append("finite_update_commutator_budget_failed")
    return {
        "variant": variant,
        "arm": arm,
        "role": role,
        "measured": bool(source),
        "required_variant": required_variant,
        "holdout_ce": ce,
        "ce_gain_vs_promoted_sparse": _gain(reference_ce, ce),
        "ce_gain_vs_token_position": _gain(token_ce, ce),
        "ce_gain_vs_dense_rank_norm": _gain(dense_ce, ce),
        "ce_gain_vs_low_churn_mlp": _gain(low_churn_ce, ce),
        "ce_gain_vs_fixed_support": _gain(fixed_ce, ce),
        "ce_gain_vs_random_support": _gain(random_ce, ce),
        "residual_l2": norm,
        "residual_norm_budget": norm_budget,
        "residual_norm_budget_ok": norm_ok,
        "mean_abs_functional_churn": churn,
        "functional_churn_budget": churn_budget,
        "functional_churn_budget_ok": churn_ok,
        "mean_commutator_l2": commutator,
        "commutator_budget": commutator_budget,
        "commutator_ratio_to_budget": _ratio(commutator, commutator_budget),
        "commutator_budget_ok": commutator_ok,
        "ce_controls_ok": ce_controls_ok,
        "variant_passes": variant_passes,
        "failure_reasons": ";".join(failures),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
    }


def _missing_variant(variant: str, reason: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "arm": "",
        "role": "missing_direct_measurement",
        "measured": False,
        "required_variant": True,
        "variant_passes": False,
        "failure_reasons": reason,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
    }


def _gate_rows(
    variant_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return [_gate("source_artifacts_present", False, "missing required source artifacts")]
    measured = [row for row in variant_rows if row.get("measured") is True and row.get("required_variant") is True]
    passing = [row for row in measured if row.get("variant_passes") is True]
    missing_required = [row for row in variant_rows if row.get("required_variant") is True and row.get("measured") is False]
    anchor = next((row for row in variant_rows if row.get("variant") == "flat_value_commutator_penalty_probe"), {})
    return [
        _gate("at_least_one_measured_mitigation_variant", bool(measured), "no measured mitigation variant rows"),
        _gate("no_required_variants_missing", not missing_required, "required order-averaged or norm-clipped rows are missing"),
        _gate("measured_variant_passes_all_gates", bool(passing), "no measured mitigation clears CE, norm, churn, and commutator gates"),
        _gate(
            "anchor_proxy_commutator_budget_passes",
            anchor.get("commutator_budget_ok") is True,
            "anchor-KL flat-value proxy still exceeds the finite-update commutator budget",
        ),
    ]


def _gate(name: str, passes: bool, failure_reason: str) -> dict[str, Any]:
    return {
        "gate": name,
        "passes": passes,
        "failure_reason": "" if passes else failure_reason,
    }


def _source_failures(
    source_rows: list[dict[str, Any]],
    design: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for row in source_rows:
        if row["source"] == "strategy_review":
            continue
        if not row["present"]:
            failures.append({"source": row["source"], "reason": "missing_required_source"})
    if design and design.get("selected_next_action") != "implement_flat_value_commutator_mitigation_probe_locally":
        failures.append({"source": "design", "reason": "design_did_not_select_probe"})
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        failures.append({"source": "strategy_review", "reason": "ben_notification_required"})
    return failures


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
        "row_count": "",
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


def _metric(row: dict[str, Any], key: str) -> float | None:
    return _float_or_none(row.get(key))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _mean_abs_metric(rows: list[dict[str, str]], arm: str, key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get("arm") == arm]
    present = [abs(value) for value in values if value is not None]
    return mean(present) if present else None


def _gain(reference_ce: float | None, candidate_ce: float | None) -> float | None:
    if reference_ce is None or candidate_ce is None:
        return None
    return reference_ce - candidate_ce


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": (_header_value(text, "notify_ben") or "false").lower() == "true",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; continued with fail-closed local mitigation probing."
    if strategy["notify_ben"] or strategy["strategic_change_level"] == "major":
        return "Strategy review requires Ben notification; the probe fails closed."
    return "Accepted the no-RunPod/fail-closed local-gating recommendation; this probe keeps GPU validation blocked."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "variant_rows.csv", summary["variant_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    notes = [
        "# Same-Router Flat-Value Commutator Mitigation Probe",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        f"- Measured passing variants: `{summary['measured_passing_variant_count']}`",
        f"- Missing required variants: `{summary['missing_required_variant_count']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "GPU validation remains blocked. The measured anchor-KL flat-value proxy does not clear the "
            "finite-update commutator budget, and direct order-averaged or stricter norm-clipped rows are absent."
        ),
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_same_router_flat_value_commutator_mitigation_probe(
        design_path=args.design,
        diagnostic_dir=args.diagnostic_dir,
        synthetic_dir=args.synthetic_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
