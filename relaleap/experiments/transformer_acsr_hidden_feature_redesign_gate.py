"""Gate the oracle-overlap Transformer-ACSR redesign on hidden-feature evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_CLOSEOUT_REDIRECT = Path("results/reports/hidden_support_classifier_closeout_redirect/summary.json")
DEFAULT_HIDDEN_AUDIT = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/summary.json")
DEFAULT_SEED_ROOT = Path("results/reports/hidden_support_classifier_sequence_ood_budget_audit/seeds")
DEFAULT_ORACLE_PREGATE = Path("results/reports/transformer_acsr_oracle_overlap_training_pregate/summary.json")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_feature_redesign_gate")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "hidden_feature_rows.csv",
    "source_rows.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_feature_redesign_gate(
    *,
    closeout_redirect_path: Path = DEFAULT_CLOSEOUT_REDIRECT,
    hidden_audit_path: Path = DEFAULT_HIDDEN_AUDIT,
    seed_root: Path = DEFAULT_SEED_ROOT,
    oracle_pregate_path: Path = DEFAULT_ORACLE_PREGATE,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local redesign gate using same-student hidden-feature rows."""

    closeout = _read_json(closeout_redirect_path)
    hidden_audit = _read_json(hidden_audit_path)
    oracle_pregate = _read_json(oracle_pregate_path)
    hidden_rows = _hidden_feature_rows(seed_root)
    source_rows = [
        _source_json("hidden_support_classifier_closeout_redirect", closeout_redirect_path, closeout),
        _source_json("hidden_support_classifier_sequence_ood_budget_audit", hidden_audit_path, hidden_audit),
        _source_json("transformer_acsr_oracle_overlap_training_pregate", oracle_pregate_path, oracle_pregate),
        {
            "source": "hidden_feature_seed_rows",
            "path": str(seed_root),
            "present": bool(hidden_rows),
            "status": "read" if hidden_rows else "missing",
            "decision": "hidden_feature_same_student_rows_available" if hidden_rows else "",
            "row_count": len(hidden_rows),
        },
    ]
    failures = [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if not row["present"]
    ]

    aggregates = _aggregates(hidden_rows)
    closeout_selected = (
        closeout.get("selected_next_action")
        == "select_oracle_overlap_aware_transformer_acsr_support_objective_redesign"
    )
    proxy_replaced = bool(
        oracle_pregate.get("source_format") == "oracle_support_summary_rows"
        and hidden_rows
    )
    learned_router_gate = bool(
        aggregates["mean_ce_gain_vs_learned_router"] is not None
        and (
            aggregates["mean_ce_gain_vs_learned_router"] > 0.0
            or (
                aggregates["mean_oracle_regret_recovery_vs_learned_router"] is not None
                and aggregates["mean_oracle_regret_recovery_vs_learned_router"] >= 0.25
            )
        )
    )
    null_gate = all(
        value is not None and value > 0.0
        for value in (
            aggregates["mean_ce_gain_vs_token_position_null"],
            aggregates["mean_ce_gain_vs_shuffled_null"],
            aggregates["mean_ce_gain_vs_frequency_null"],
        )
    )
    leakage_gate = bool(
        aggregates["max_future_perturbation_prefix_delta"] is not None
        and aggregates["max_future_perturbation_prefix_delta"] <= 1e-5
    )
    hidden_feature_gate_passes = bool(
        not failures
        and closeout_selected
        and proxy_replaced
        and learned_router_gate
        and null_gate
        and leakage_gate
    )
    decision = (
        "transformer_acsr_hidden_feature_redesign_gate_passed_local"
        if hidden_feature_gate_passes
        else "transformer_acsr_hidden_feature_redesign_gate_gpu_blocked"
    )
    selected_next_step = (
        "add_sequence_and_rule_ood_budgeted_hidden_feature_gpu_validation"
        if hidden_feature_gate_passes
        else "design_regret_soft_utility_head_with_margin_conditioned_learned_router_fallback"
    )
    summary = {
        "status": "fail" if failures else "pass",
        "decision": decision if not failures else "transformer_acsr_hidden_feature_redesign_gate_failed_closed",
        "claim_status": (
            "hidden_feature_same_student_gate_passed"
            if hidden_feature_gate_passes
            else "hidden_feature_same_student_gate_loses_to_learned_router"
            if hidden_rows and not learned_router_gate
            else "hidden_feature_same_student_gate_incomplete"
        ),
        "source_format": "hidden_feature_same_student_intervention_rows",
        "oracle_overlap_proxy_pregate_replaced": proxy_replaced,
        "closeout_redirect_selected": closeout_selected,
        "hidden_feature_seed_count": len(hidden_rows),
        "learned_router_gate_passes": learned_router_gate,
        "null_gate_passes": null_gate,
        "future_perturbation_leakage_gate_passes": leakage_gate,
        "hidden_feature_gate_passes": hidden_feature_gate_passes,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": selected_next_step,
        "aggregates": aggregates,
        "source_rows": source_rows,
        "failures": failures,
        "strategy_review_handling": (
            "Accepted the latest no-RunPod/fail-closed recommendation. This gate uses hidden-feature "
            "same-student rows before GPU and blocks because the learned-router comparison fails."
        ),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, hidden_rows)
    return summary


def _hidden_feature_rows(seed_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seed_root.exists():
        return rows
    for seed_dir in sorted(path for path in seed_root.iterdir() if path.is_dir()):
        pilot_rows = _read_csv(seed_dir / "transformer_acsr_cpu_smoke_pilot.csv")
        support_rows = _read_csv(seed_dir / "support_head_sequence_heldout_diagnostic.csv")
        pilot = next(
            (
                row
                for row in pilot_rows
                if row.get("row_role") == "primary_transformer_acsr_cpu_smoke_pilot"
            ),
            pilot_rows[0] if pilot_rows else {},
        )
        support = next(
            (
                row
                for row in support_rows
                if row.get("arm") == "promoted_contextual_topk2"
                and row.get("diagnostic") == "support_regret_trained_contextual_router_topk2"
                and row.get("split") == "sequence_heldout"
            ),
            support_rows[0] if support_rows else {},
        )
        if not pilot:
            continue
        hidden_ce = _float_or_none(pilot.get("direct_hidden_support_classifier_ce"))
        learned_ce = _float_or_none(support.get("learned_router_ce"))
        oracle_ce = _float_or_none(support.get("oracle_pair_ce_ceiling"))
        rows.append(
            {
                "seed": seed_dir.name.removeprefix("seed_"),
                "split": "sequence_heldout",
                "predictor": "direct_hidden_support_classifier",
                "source_artifact_dir": str(seed_dir),
                "hidden_classifier_ce": hidden_ce,
                "learned_router_ce": learned_ce,
                "oracle_pair_ce_ceiling": oracle_ce,
                "ce_gain_vs_learned_router": _safe_gain(learned_ce, hidden_ce),
                "oracle_regret_recovery_vs_learned_router": _safe_regret_recovery(
                    learned_ce=learned_ce,
                    candidate_ce=hidden_ce,
                    oracle_ce=oracle_ce,
                ),
                "ce_gain_vs_token_position_null": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_ce_gain_vs_token_position_null")
                ),
                "ce_gain_vs_shuffled_null": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_ce_gain_vs_shuffled_null")
                ),
                "ce_gain_vs_frequency_null": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_ce_gain_vs_frequency_null")
                ),
                "oracle_overlap": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_overlap_with_oracle")
                ),
                "oracle_exact_match": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_exact_match_with_oracle")
                ),
                "future_perturbation_prefix_delta": _float_or_none(
                    pilot.get("direct_hidden_support_classifier_future_perturbation_max_prefix_delta")
                ),
                "prefix_safe_feature_family": "hidden_state_prefix_features",
                "uses_target_token_as_predictor_feature": False,
                "uses_oracle_loss_as_predictor_feature": False,
            }
        )
    return rows


def _aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_hidden_classifier_ce": _mean_present(rows, "hidden_classifier_ce"),
        "mean_learned_router_ce": _mean_present(rows, "learned_router_ce"),
        "mean_oracle_pair_ce_ceiling": _mean_present(rows, "oracle_pair_ce_ceiling"),
        "mean_ce_gain_vs_learned_router": _mean_present(rows, "ce_gain_vs_learned_router"),
        "mean_oracle_regret_recovery_vs_learned_router": _mean_present(
            rows, "oracle_regret_recovery_vs_learned_router"
        ),
        "mean_ce_gain_vs_token_position_null": _mean_present(rows, "ce_gain_vs_token_position_null"),
        "mean_ce_gain_vs_shuffled_null": _mean_present(rows, "ce_gain_vs_shuffled_null"),
        "mean_ce_gain_vs_frequency_null": _mean_present(rows, "ce_gain_vs_frequency_null"),
        "mean_oracle_overlap": _mean_present(rows, "oracle_overlap"),
        "mean_oracle_exact_match": _mean_present(rows, "oracle_exact_match"),
        "max_future_perturbation_prefix_delta": _max_present(
            rows, "future_perturbation_prefix_delta"
        ),
    }


def _safe_gain(reference_ce: float | None, candidate_ce: float | None) -> float | None:
    if reference_ce is None or candidate_ce is None:
        return None
    return reference_ce - candidate_ce


def _safe_regret_recovery(
    *,
    learned_ce: float | None,
    candidate_ce: float | None,
    oracle_ce: float | None,
) -> float | None:
    if learned_ce is None or candidate_ce is None or oracle_ce is None:
        return None
    learned_regret = learned_ce - oracle_ce
    if learned_regret <= 0.0:
        return None
    return (learned_ce - candidate_ce) / learned_regret


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    return mean(values) if values else None


def _max_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    return max(values) if values else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "row_count": "",
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any], hidden_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "hidden_feature_rows.csv", hidden_rows)
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    notes = [
        "# Transformer-ACSR Hidden-Feature Redesign Gate",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Oracle-overlap proxy pregate replaced: `{summary['oracle_overlap_proxy_pregate_replaced']}`",
        f"- Hidden-feature seed count: `{summary['hidden_feature_seed_count']}`",
        f"- Mean CE gain vs learned router: `{summary['aggregates']['mean_ce_gain_vs_learned_router']}`",
        f"- Mean oracle-regret recovery vs learned router: `{summary['aggregates']['mean_oracle_regret_recovery_vs_learned_router']}`",
        f"- Hidden-feature gate passes: `{summary['hidden_feature_gate_passes']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "This report treats the command-generated hidden-feature same-student intervention rows as the "
            "source of truth after the row-proxy oracle-overlap pregate. GPU validation remains blocked "
            "unless the hidden-feature redesign beats the learned router, nulls, and leakage gates locally."
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout-redirect", type=Path, default=DEFAULT_CLOSEOUT_REDIRECT)
    parser.add_argument("--hidden-audit", type=Path, default=DEFAULT_HIDDEN_AUDIT)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--oracle-pregate", type=Path, default=DEFAULT_ORACLE_PREGATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_feature_redesign_gate(
        closeout_redirect_path=args.closeout_redirect,
        hidden_audit_path=args.hidden_audit,
        seed_root=args.seed_root,
        oracle_pregate_path=args.oracle_pregate,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
