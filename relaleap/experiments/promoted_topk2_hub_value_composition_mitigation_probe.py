"""Hub-focused value-composition mitigation probe for promoted top-k-2."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_promoted_topk2_hub_value_composition_mitigation_probe"
)
DEFAULT_LOCALIZATION_REPORT = Path(
    "results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

HUB_VALUE_COMPOSITION_CANDIDATE_FOUND = "hub_value_composition_candidate_found"
HUB_VALUE_COMPOSITION_NOT_ESTABLISHED = "hub_value_composition_not_established"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

_BASELINE_VARIANT = "promoted_contextual_topk2"
_CONTROL_VARIANTS = (
    "rank_matched_contextual_topk1",
    "random_fixed_topk2",
    "norm_matched_dense_active_rank",
)
_HUB_VARIANTS = (
    "hub_value_composition_w010_contextual_topk2",
    "hub_value_composition_w100_contextual_topk2",
)


def run_promoted_topk2_hub_value_composition_mitigation_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    localization_report_path: Path = DEFAULT_LOCALIZATION_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    commutator_reduction_fraction: float = 0.5,
    transfer_retention_fraction: float = 0.8,
    support_usage_retention_fraction: float = 0.8,
    anchor_ce_drift_tolerance: float = 0.05,
    residual_l2_increase_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Run and gate hub-focused value-composition penalty variants."""

    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    localization = _read_json_object(localization_report_path)
    strategy_review = _strategy_review(strategy_review_path)
    microtest = run_retention_churn_microtest(
        config_path,
        out_dir,
        include_hub_value_composition_variants=True,
    )
    variant_rows = [
        row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    ]
    variants = {str(row.get("variant")): row for row in variant_rows}
    hub_rows = [_hub_row(variants, name) for name in _HUB_VARIANTS if name in variants]
    thresholds = {
        "commutator_reduction_fraction": commutator_reduction_fraction,
        "transfer_retention_fraction": transfer_retention_fraction,
        "support_usage_retention_fraction": support_usage_retention_fraction,
        "anchor_ce_drift_tolerance": anchor_ce_drift_tolerance,
        "residual_l2_increase_tolerance": residual_l2_increase_tolerance,
    }
    source_rows = [
        _source_row(
            "pairwise_value_interaction_localization",
            localization_report_path,
            localization,
        ),
        {
            "source": "retention_churn_microtest",
            "path": str(out_dir / "variant_metrics.csv"),
            "present": (out_dir / "variant_metrics.csv").is_file(),
            "status": microtest.get("status"),
            "decision": "",
            "claim_status": f"variant_count={len(variant_rows)}",
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    metrics = _metrics(variants, hub_rows, localization)
    failures = _failures(microtest, variants, localization, metrics)
    qualifying_rows = [
        row
        for row in hub_rows
        if _qualifies(
            row,
            commutator_reduction_fraction=commutator_reduction_fraction,
            transfer_retention_fraction=transfer_retention_fraction,
            support_usage_retention_fraction=support_usage_retention_fraction,
            anchor_ce_drift_tolerance=anchor_ce_drift_tolerance,
            residual_l2_increase_tolerance=residual_l2_increase_tolerance,
        )
    ]

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        rationale = (
            "The hub value-composition mitigation probe cannot be interpreted "
            "because required localization evidence, controls, or numeric gate "
            "fields are missing."
        )
        next_step = "repair hub value-composition mitigation source artifacts"
    elif qualifying_rows:
        status = "pass"
        decision = HUB_VALUE_COMPOSITION_CANDIDATE_FOUND
        best = min(
            qualifying_rows,
            key=lambda row: float(row["commutator_anchor_logit_mse"]),
        )
        selected_next_action = "runpod_hub_value_composition_validation"
        rationale = (
            f"`{best['variant']}` reduced anchor commutator logit MSE by "
            f"`{best['commutator_anchor_logit_mse_reduction_fraction']}` while "
            "preserving transfer improvement, support usage, CE drift, and "
            "residual-stream L2 under the preregistered local gate. It remains "
            "a candidate only; RunPod validation is required before promotion."
        )
        next_step = (
            "validate the qualifying hub value-composition candidate on RunPod "
            "with the same commutator/CE/residual-norm controls"
        )
    else:
        status = "pass"
        decision = HUB_VALUE_COMPOSITION_NOT_ESTABLISHED
        selected_next_action = "value_composition_branch_reassess"
        rationale = (
            "The hub-focused value-composition penalty did not clear the "
            "commutator reduction gate while preserving transfer, support "
            "usage, CE drift, and residual L2. This rejects the narrow hub "
            "penalty as a promoted mitigation under the current local evidence."
        )
        next_step = (
            "reassess value-composition mitigation design before spending GPU "
            "time; keep contextual top-k-2 operational but causal claims blocked"
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "localization_report_path": str(localization_report_path),
        "thresholds": thresholds,
        "control_variants": list(_CONTROL_VARIANTS),
        "hub_value_composition_variants": list(_HUB_VARIANTS),
        "metrics": metrics,
        "hub_value_composition_rows": hub_rows,
        "claim_statuses": {
            "contextual_topk2_router": "operational_default_train_time_support_selection",
            "hub_value_composition_mitigation": (
                "candidate_not_promoted"
                if decision == HUB_VALUE_COMPOSITION_CANDIDATE_FOUND
                else "not_established"
            ),
            "topk2_causal_cooperation": "not_supported",
            "router_policy_mitigation": "closed_not_established",
            "order_averaging": "diagnostic_only_not_promoted",
        },
        "source_rows": source_rows,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "hub_value_composition_rows_csv": str(
                out_dir / "hub_value_composition_rows.csv"
            ),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "variant_metrics_csv": str(out_dir / "variant_metrics.csv"),
            "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
            "per_token_commutator_csv": str(out_dir / "per_token_commutator.csv"),
            "notes_md": str(out_dir / "hub_value_composition_notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", source_rows)
    _write_csv(out_dir / "hub_value_composition_rows.csv", hub_rows)
    _write_notes(out_dir / "hub_value_composition_notes.md", summary)
    return summary


def _hub_row(variants: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    current = variants.get(name, {})
    return {
        "variant": name,
        "baseline_variant": _BASELINE_VARIANT,
        "hub_value_composition_penalty_weight": _float_or_none(
            current.get("hub_value_composition_penalty_weight")
        ),
        "hub_value_composition_column": _int_or_none(
            current.get("hub_value_composition_column")
        ),
        "commutator_anchor_logit_mse": _float_or_none(
            current.get("commutator_anchor_logit_mse")
        ),
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "commutator_anchor_logit_mse_reduction_fraction": _fractional_reduction(
            baseline.get("commutator_anchor_logit_mse"),
            current.get("commutator_anchor_logit_mse"),
        ),
        "commutator_transfer_logit_mse": _float_or_none(
            current.get("commutator_transfer_logit_mse")
        ),
        "transfer_ce_improvement": _float_or_none(current.get("transfer_ce_improvement")),
        "baseline_transfer_ce_improvement": _float_or_none(
            baseline.get("transfer_ce_improvement")
        ),
        "transfer_retention_fraction": _ratio(
            current.get("transfer_ce_improvement"),
            baseline.get("transfer_ce_improvement"),
        ),
        "anchor_used_columns_after_transfer": _float_or_none(
            current.get("anchor_used_columns_after_transfer")
        ),
        "baseline_anchor_used_columns_after_transfer": _float_or_none(
            baseline.get("anchor_used_columns_after_transfer")
        ),
        "support_usage_retention_fraction": _ratio(
            current.get("anchor_used_columns_after_transfer"),
            baseline.get("anchor_used_columns_after_transfer"),
        ),
        "anchor_ce_drift": _float_or_none(current.get("anchor_ce_drift")),
        "baseline_anchor_ce_drift": _float_or_none(baseline.get("anchor_ce_drift")),
        "anchor_support_churn_after_transfer": _float_or_none(
            current.get("anchor_support_churn_after_transfer")
        ),
        "commutator_anchor_support_churn": _float_or_none(
            current.get("commutator_anchor_support_churn")
        ),
        "commutator_anchor_residual_stream_l2": _float_or_none(
            current.get("commutator_anchor_residual_stream_l2")
        ),
        "baseline_commutator_anchor_residual_stream_l2": _float_or_none(
            baseline.get("commutator_anchor_residual_stream_l2")
        ),
        "commutator_anchor_residual_stream_l2_change_fraction": _fractional_change(
            baseline.get("commutator_anchor_residual_stream_l2"),
            current.get("commutator_anchor_residual_stream_l2"),
        ),
    }


def _metrics(
    variants: dict[str, dict[str, Any]],
    hub_rows: list[dict[str, Any]],
    localization: dict[str, Any],
) -> dict[str, Any]:
    baseline = variants.get(_BASELINE_VARIANT, {})
    controls = [variants[name] for name in _CONTROL_VARIANTS if name in variants]
    reductions = [
        float(row["commutator_anchor_logit_mse_reduction_fraction"])
        for row in hub_rows
        if row["commutator_anchor_logit_mse_reduction_fraction"] is not None
    ]
    commutators = [
        float(row["commutator_anchor_logit_mse"])
        for row in hub_rows
        if row["commutator_anchor_logit_mse"] is not None
    ]
    localization_metrics = localization.get("metrics", {})
    return {
        "dominant_column": localization_metrics.get("dominant_column"),
        "dominant_column_abs_synergy_share": _float_or_none(
            localization_metrics.get("dominant_column_abs_synergy_share")
        ),
        "top3_pair_abs_synergy_share": _float_or_none(
            localization_metrics.get("top3_pair_abs_synergy_share")
        ),
        "value_only_fraction_of_full": _float_or_none(
            localization_metrics.get("value_only_fraction_of_full")
        ),
        "baseline_commutator_anchor_logit_mse": _float_or_none(
            baseline.get("commutator_anchor_logit_mse")
        ),
        "baseline_commutator_anchor_residual_stream_l2": _float_or_none(
            baseline.get("commutator_anchor_residual_stream_l2")
        ),
        "baseline_transfer_ce_improvement": _float_or_none(
            baseline.get("transfer_ce_improvement")
        ),
        "baseline_anchor_used_columns_after_transfer": _float_or_none(
            baseline.get("anchor_used_columns_after_transfer")
        ),
        "control_count": len(controls),
        "hub_value_composition_count": len(hub_rows),
        "best_hub_commutator_anchor_logit_mse": min(commutators) if commutators else None,
        "best_hub_reduction_fraction": max(reductions) if reductions else None,
    }


def _qualifies(
    row: dict[str, Any],
    *,
    commutator_reduction_fraction: float,
    transfer_retention_fraction: float,
    support_usage_retention_fraction: float,
    anchor_ce_drift_tolerance: float,
    residual_l2_increase_tolerance: float,
) -> bool:
    ce_drift = _float_or_none(row.get("anchor_ce_drift"))
    residual_l2_change = _float_or_none(
        row.get("commutator_anchor_residual_stream_l2_change_fraction")
    )
    return (
        _at_least(
            row.get("commutator_anchor_logit_mse_reduction_fraction"),
            commutator_reduction_fraction,
        )
        and _at_least(row.get("transfer_retention_fraction"), transfer_retention_fraction)
        and _at_least(
            row.get("support_usage_retention_fraction"),
            support_usage_retention_fraction,
        )
        and ce_drift is not None
        and abs(ce_drift) <= anchor_ce_drift_tolerance
        and residual_l2_change is not None
        and residual_l2_change <= residual_l2_increase_tolerance
    )


def _failures(
    microtest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    localization: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    if localization.get("status") != "pass":
        failures.append(
            {
                "source": "pairwise_value_interaction_localization",
                "field": "status",
                "expected": "pass",
                "actual": localization.get("status"),
            }
        )
    if localization.get("decision") != "pairwise_value_interaction_localized_hub_family":
        failures.append(
            {
                "source": "pairwise_value_interaction_localization",
                "field": "decision",
                "expected": "pairwise_value_interaction_localized_hub_family",
                "actual": localization.get("decision"),
            }
        )
    if microtest.get("status") != "ok":
        failures.append(
            {"source": "retention_churn_microtest", "field": "status", "expected": "ok", "actual": microtest.get("status")}
        )
    for name in (_BASELINE_VARIANT, *_CONTROL_VARIANTS, *_HUB_VARIANTS):
        if name not in variants:
            failures.append(
                {
                    "source": "retention_churn_microtest",
                    "field": "variant",
                    "expected": name,
                    "actual": "missing",
                }
            )
    for name in (_BASELINE_VARIANT, *_HUB_VARIANTS):
        row = variants.get(name, {})
        for field in (
            "commutator_anchor_logit_mse",
            "commutator_anchor_residual_stream_l2",
            "transfer_ce_improvement",
            "anchor_used_columns_after_transfer",
            "anchor_ce_drift",
        ):
            if _float_or_none(row.get(field)) is None:
                failures.append(
                    {
                        "source": "retention_churn_microtest",
                        "field": f"{name}.{field}",
                        "expected": "numeric",
                        "actual": row.get(field),
                    }
                )
    for field in (
        "dominant_column",
        "dominant_column_abs_synergy_share",
        "top3_pair_abs_synergy_share",
        "value_only_fraction_of_full",
    ):
        if metrics.get(field) is None:
            failures.append(
                {
                    "source": "pairwise_value_interaction_localization",
                    "field": field,
                    "expected": "present",
                    "actual": None,
                }
            )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted the conservative recommendation to keep contextual "
            "top-k-2 operational while withholding causal-cooperation claims; "
            "this probe follows the value-composition localization branch "
            "instead of promoting router-policy changes"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _float_or_none(numerator)
    den = _float_or_none(denominator)
    if num is None or den is None or abs(den) <= 1e-12:
        return None
    return num / den


def _fractional_reduction(baseline: Any, current: Any) -> float | None:
    base = _float_or_none(baseline)
    cur = _float_or_none(current)
    if base is None or cur is None or abs(base) <= 1e-12:
        return None
    return (base - cur) / base


def _fractional_change(baseline: Any, current: Any) -> float | None:
    base = _float_or_none(baseline)
    cur = _float_or_none(current)
    if base is None or cur is None or abs(base) <= 1e-12:
        return None
    return (cur - base) / base


def _at_least(value: Any, threshold: float) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and numeric >= threshold


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
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Promoted Top-k-2 Hub Value-Composition Mitigation Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Dominant localized column: `{metrics['dominant_column']}`",
        "- Dominant-column absolute-synergy share: "
        f"`{metrics['dominant_column_abs_synergy_share']}`",
        f"- Best hub reduction fraction: `{metrics['best_hub_reduction_fraction']}`",
        "",
        "## Interpretation",
        "",
        summary["rationale"],
        "",
        "## Claim Policy",
        "",
        "This is a no-promotion mitigation probe. Contextual top-k-2 remains "
        "the operational support-router default; top-k-2 causal-cooperation "
        "claims remain blocked.",
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--localization-report",
        type=Path,
        default=DEFAULT_LOCALIZATION_REPORT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args()
    summary = run_promoted_topk2_hub_value_composition_mitigation_probe(
        config_path=args.config,
        out_dir=args.out,
        localization_report_path=args.localization_report,
        strategy_review_path=args.strategy_review,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "next_step": summary["next_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
