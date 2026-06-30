"""Repeat-gate report for hidden-support Transformer-ACSR null evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SEED_REPEAT = Path("results/reports/transformer_acsr_seed_repeat/summary.json")
DEFAULT_HIDDEN_FEATURE_GATE = Path(
    "results/reports/transformer_acsr_hidden_feature_redesign_gate/summary.json"
)
DEFAULT_SEQUENCE_AUDIT = Path(
    "results/reports/hidden_support_classifier_sequence_ood_budget_audit/summary.json"
)
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_null_gate_repeat")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "repeat_rows.csv",
    "source_rows.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_null_gate_repeat(
    *,
    seed_repeat_path: Path = DEFAULT_SEED_REPEAT,
    hidden_feature_gate_path: Path = DEFAULT_HIDDEN_FEATURE_GATE,
    sequence_audit_path: Path = DEFAULT_SEQUENCE_AUDIT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    seed_repeat = _read_json(seed_repeat_path)
    hidden_feature_gate = _read_json(hidden_feature_gate_path)
    sequence_audit = _read_json(sequence_audit_path)
    source_rows = [
        _source_row("transformer_acsr_seed_repeat", seed_repeat_path, seed_repeat),
        _source_row("transformer_acsr_hidden_feature_redesign_gate", hidden_feature_gate_path, hidden_feature_gate),
        _source_row("hidden_support_classifier_sequence_ood_budget_audit", sequence_audit_path, sequence_audit),
    ]
    failures = [
        {"source": row["source"], "path": row["path"], "reason": "required source artifact missing"}
        for row in source_rows
        if row["present"] is not True
    ]

    repeat_rows = _repeat_rows(seed_repeat, hidden_feature_gate, sequence_audit)
    gates = {row["gate"]: row for row in repeat_rows}
    repeat_gate_passes = bool(repeat_rows and all(row["gate_passes"] is True for row in repeat_rows))
    advance_to_gpu = bool(not failures and repeat_gate_passes)
    selected_next_step = (
        "run_runpod_transformer_acsr_hidden_support_validation_with_artifact_checks"
        if advance_to_gpu
        else "design_regret_soft_utility_head_with_margin_conditioned_learned_router_fallback"
    )
    summary = {
        "status": "fail" if failures else "pass",
        "decision": (
            "transformer_acsr_hidden_null_gate_repeat_passed_gpu_ready"
            if advance_to_gpu
            else "transformer_acsr_hidden_null_gate_repeat_gpu_blocked"
            if not failures
            else "transformer_acsr_hidden_null_gate_repeat_failed_closed"
        ),
        "claim_status": (
            "hidden_classifier_repeat_gate_passed"
            if advance_to_gpu
            else "hidden_classifier_repeat_gate_loses_to_learned_router_or_missing_budget_ood"
        ),
        "seed_count": seed_repeat.get("seed_count"),
        "mean_hidden_classifier_ce_gain_vs_learned_router": _first_present(
            hidden_feature_gate.get("aggregates", {}).get("mean_ce_gain_vs_learned_router"),
            sequence_audit.get("mean_hidden_classifier_ce_gain_vs_learned_router"),
            seed_repeat.get("mean_hidden_classifier_ce_gain_vs_learned_router"),
        ),
        "mean_oracle_regret_recovery_vs_learned_router": _first_present(
            hidden_feature_gate.get("aggregates", {}).get(
                "mean_oracle_regret_recovery_vs_learned_router"
            ),
            sequence_audit.get("mean_oracle_regret_recovery_vs_learned_router"),
            seed_repeat.get("mean_hidden_classifier_oracle_regret_recovery_vs_learned_router"),
        ),
        "weak_null_gate_passes": gates.get("weak_null_repeat", {}).get("gate_passes") is True,
        "learned_router_gate_passes": gates.get("learned_router_repeat", {}).get("gate_passes") is True,
        "sequence_heldout_gate_passes": gates.get("sequence_heldout_repeat", {}).get("gate_passes") is True,
        "rule_ood_gate_passes": gates.get("rule_ood_repeat", {}).get("gate_passes") is True,
        "budget_gate_passes": gates.get("budget_repeat", {}).get("gate_passes") is True,
        "leakage_gate_passes": gates.get("future_perturbation_repeat", {}).get("gate_passes") is True,
        "repeat_gate_passes": repeat_gate_passes,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": advance_to_gpu,
        "selected_next_step": selected_next_step,
        "source_rows": source_rows,
        "failures": failures,
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, repeat_rows, source_rows)
    return summary


def _repeat_rows(
    seed_repeat: dict[str, Any],
    hidden_feature_gate: dict[str, Any],
    sequence_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    required = seed_repeat.get("hidden_classifier_gpu_gate_required_fields", {})
    hidden_aggregates = hidden_feature_gate.get("aggregates", {})
    return [
        {
            "gate": "weak_null_repeat",
            "source": "transformer_acsr_seed_repeat",
            "gate_passes": seed_repeat.get("hidden_classifier_null_margin_gate_passes") is True
            and hidden_feature_gate.get("null_gate_passes") is True,
            "source_status": required.get("weak_null_margins", ""),
            "metric": "mean_hidden_classifier_ce_gain_vs_token_position_shuffled_frequency_nulls",
            "metric_value": ";".join(
                str(value)
                for value in (
                    seed_repeat.get("mean_hidden_classifier_ce_gain_vs_token_position_null"),
                    seed_repeat.get("mean_hidden_classifier_ce_gain_vs_shuffled_null"),
                    seed_repeat.get("mean_hidden_classifier_ce_gain_vs_frequency_null"),
                )
            ),
            "failure_reason": "",
        },
        {
            "gate": "learned_router_repeat",
            "source": "transformer_acsr_hidden_feature_redesign_gate",
            "gate_passes": hidden_feature_gate.get("learned_router_gate_passes") is True
            and seed_repeat.get("hidden_classifier_learned_router_gate_passes") is True,
            "source_status": required.get("learned_router_comparison", ""),
            "metric": "mean_ce_gain_vs_learned_router",
            "metric_value": hidden_aggregates.get("mean_ce_gain_vs_learned_router"),
            "failure_reason": "hidden classifier must beat learned router or recover >=25% router-oracle regret",
        },
        {
            "gate": "sequence_heldout_repeat",
            "source": "hidden_support_classifier_sequence_ood_budget_audit",
            "gate_passes": sequence_audit.get("sequence_heldout_gate_passes") is True,
            "source_status": required.get("sequence_heldout", ""),
            "metric": "sequence_heldout_gate_passes",
            "metric_value": sequence_audit.get("sequence_heldout_gate_passes"),
            "failure_reason": "sequence-heldout same-student support intervention gate failed",
        },
        {
            "gate": "rule_ood_repeat",
            "source": "hidden_support_classifier_sequence_ood_budget_audit",
            "gate_passes": sequence_audit.get("rule_combo_heldout_gate_passes") is True,
            "source_status": required.get("rule_ood", ""),
            "metric": "rule_combo_evidence_available",
            "metric_value": sequence_audit.get("rule_combo_evidence_available"),
            "failure_reason": "rule-combo OOD support intervention evidence is absent",
        },
        {
            "gate": "budget_repeat",
            "source": "hidden_support_classifier_sequence_ood_budget_audit",
            "gate_passes": sequence_audit.get("budget_gate_passes") is True
            and seed_repeat.get("hidden_classifier_churn_budget_gate_passes") is True
            and seed_repeat.get("hidden_classifier_commutator_budget_gate_passes") is True,
            "source_status": ",".join(
                str(required.get(name, ""))
                for name in ("churn_budget", "commutator_budget")
            ),
            "metric": "residual_norm_functional_churn_commutator_budget_gates",
            "metric_value": sequence_audit.get("budget_gate_passes"),
            "failure_reason": "exact hidden-classifier norm/churn/commutator budget rows are missing or failed",
        },
        {
            "gate": "future_perturbation_repeat",
            "source": "transformer_acsr_hidden_feature_redesign_gate",
            "gate_passes": hidden_feature_gate.get("future_perturbation_leakage_gate_passes") is True
            and seed_repeat.get("hidden_classifier_leakage_pass_count") == seed_repeat.get("seed_count"),
            "source_status": "pass"
            if hidden_feature_gate.get("future_perturbation_leakage_gate_passes") is True
            else "fail",
            "metric": "max_future_perturbation_prefix_delta",
            "metric_value": hidden_aggregates.get("max_future_perturbation_prefix_delta"),
            "failure_reason": "",
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "missing" if not payload else ""),
        "decision": payload.get("decision", ""),
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


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


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    repeat_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "repeat_rows.csv", repeat_rows)
    _write_csv(out_dir / "source_rows.csv", source_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    notes = [
        "# Transformer-ACSR Hidden Null Gate Repeat",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Weak-null gate passes: `{summary['weak_null_gate_passes']}`",
        f"- Learned-router gate passes: `{summary['learned_router_gate_passes']}`",
        f"- Sequence-heldout gate passes: `{summary['sequence_heldout_gate_passes']}`",
        f"- Rule-OOD gate passes: `{summary['rule_ood_gate_passes']}`",
        f"- Budget gate passes: `{summary['budget_gate_passes']}`",
        f"- Future-perturbation leakage gate passes: `{summary['leakage_gate_passes']}`",
        f"- Mean CE gain vs learned router: `{summary['mean_hidden_classifier_ce_gain_vs_learned_router']}`",
        f"- Mean oracle-regret recovery vs learned router: `{summary['mean_oracle_regret_recovery_vs_learned_router']}`",
        f"- Advance to GPU validation: `{summary['advance_to_gpu_validation']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        (
            "This repeat report treats weak-null wins as insufficient. GPU validation remains blocked unless "
            "the hidden classifier also clears the learned-router, sequence/OOD, budget, and leakage gates."
        ),
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-repeat", type=Path, default=DEFAULT_SEED_REPEAT)
    parser.add_argument("--hidden-feature-gate", type=Path, default=DEFAULT_HIDDEN_FEATURE_GATE)
    parser.add_argument("--sequence-audit", type=Path, default=DEFAULT_SEQUENCE_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_null_gate_repeat(
        seed_repeat_path=args.seed_repeat,
        hidden_feature_gate_path=args.hidden_feature_gate,
        sequence_audit_path=args.sequence_audit,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
