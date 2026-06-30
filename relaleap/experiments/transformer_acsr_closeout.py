"""Close out or redirect the value-aware Transformer-ACSR local branch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("results/reports/transformer_acsr_seed_repeat")
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_closeout")


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


def _load_seed_repeat_summary(source_dir: Path) -> dict[str, Any]:
    summary_path = source_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing Transformer-ACSR seed-repeat summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_transformer_acsr_closeout(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume the seed-repeat report and write a fail-closed closeout artifact."""

    source = _load_seed_repeat_summary(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_all = source.get("completed_seed_count") == source.get("seed_count")
    robust_value_gate = bool(source.get("robust_value_gate_passes"))
    overlap_gate = bool(source.get("oracle_overlap_gate_passes"))
    advance_to_gpu = bool(source.get("advance_to_gpu_validation"))
    value_pass_count = int(source.get("value_aware_gate_pass_count", 0))
    seed_count = int(source.get("seed_count", 0))
    mean_overlap = source.get("mean_value_aware_support_overlap_with_oracle")
    mean_gain = source.get("mean_value_aware_ce_gain_vs_token_position_support")

    failure_reasons: list[str] = []
    if not completed_all:
        failure_reasons.append("seed_repeat_incomplete")
    if not robust_value_gate:
        failure_reasons.append("value_aware_gate_not_robust_across_seeds")
    if not overlap_gate:
        failure_reasons.append("oracle_overlap_below_local_gate")
    if advance_to_gpu:
        closeout_status = "repeat_passed_ready_for_runpod_artifact_checked_validation"
        selected_next = "run_runpod_transformer_acsr_validation_with_artifact_checks"
        branch_closed = False
    else:
        closeout_status = "closed_value_aware_support_router_needs_redesign"
        selected_next = "design_oracle_overlap_aware_transformer_acsr_support_objective"
        branch_closed = True

    rows = [
        {
            "row_role": "primary_value_aware_transformer_acsr_closeout",
            "source_report": str(source_dir),
            "source_decision": source.get("decision", ""),
            "source_seed_count": seed_count,
            "source_completed_seed_count": source.get("completed_seed_count"),
            "source_value_aware_gate_pass_count": value_pass_count,
            "source_leakage_pass_count": source.get("leakage_pass_count"),
            "source_support_intervention_assay_valid_count": source.get(
                "support_intervention_assay_valid_count"
            ),
            "mean_value_aware_ce_gain_vs_token_position_support": mean_gain,
            "mean_value_aware_support_overlap_with_oracle": mean_overlap,
            "robust_value_gate_passes": robust_value_gate,
            "oracle_overlap_gate_passes": overlap_gate,
            "closeout_status": closeout_status,
            "failure_reasons": ";".join(failure_reasons),
            "branch_closed": branch_closed,
            "selected_next_experiment": selected_next,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "interpretation": (
                "Fail-closed closeout for the value-aware Transformer-ACSR support router. "
                "It consumes the adjacent-seed local repeat before deciding whether GPU "
                "validation is scientifically justified."
            ),
        },
        {
            "row_role": "redesign_candidate",
            "source_report": str(source_dir),
            "candidate_path": "oracle_overlap_aware_support_distribution_training",
            "candidate_family": "transformer_acsr",
            "candidate_status": "selected" if branch_closed else "deferred_by_gpu_ready_repeat",
            "required_change": (
                "train support logits/distributions with an explicit oracle-overlap or regret "
                "term while preserving prefix-safe inputs and same-student value intervention"
            ),
            "required_controls": (
                "token_position;shuffled_targets;delayed_targets;mlp_gru;random_support;"
                "causal_topk2;future_perturbation"
            ),
            "selected_next_experiment": selected_next if branch_closed else "",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
        },
        {
            "row_role": "blocked_gpu_validation",
            "source_report": str(source_dir),
            "candidate_path": "runpod_validation",
            "candidate_family": "backend_validation",
            "candidate_status": "blocked" if branch_closed else "ready_after_local_gates",
            "required_change": "local robust value gate and oracle-overlap gate must both pass first",
            "required_controls": "local_artifact_check_after_fetch",
            "selected_next_experiment": "" if branch_closed else selected_next,
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
        },
    ]

    summary = {
        "status": "pass",
        "decision": (
            "transformer_acsr_closeout_gpu_validation_ready"
            if advance_to_gpu
            else "transformer_acsr_closeout_local_redesign_required"
        ),
        "source_report": str(source_dir),
        "source_decision": source.get("decision", ""),
        "seed_count": seed_count,
        "completed_seed_count": source.get("completed_seed_count"),
        "value_aware_gate_pass_count": value_pass_count,
        "mean_value_aware_ce_gain_vs_token_position_support": mean_gain,
        "mean_value_aware_support_overlap_with_oracle": mean_overlap,
        "robust_value_gate_passes": robust_value_gate,
        "oracle_overlap_gate_passes": overlap_gate,
        "closeout_status": closeout_status,
        "failure_reasons": failure_reasons,
        "branch_closed": branch_closed,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": selected_next,
        "artifacts": {
            "closeout_csv": str(out_dir / "transformer_acsr_closeout.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }

    _write_csv(out_dir / "transformer_acsr_closeout.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    notes = [
        "# Transformer-ACSR Closeout",
        "",
        f"- Source: `{source_dir}`",
        f"- Source decision: `{source.get('decision', '')}`",
        f"- Value-aware gate pass count: `{value_pass_count}/{seed_count}`",
        f"- Mean value-aware CE gain vs token/position: `{mean_gain}`",
        f"- Mean value-aware oracle overlap: `{mean_overlap}`",
        f"- Closeout status: `{closeout_status}`",
        f"- Next step: `{selected_next}`",
        "",
        "RunPod/Colab validation remains blocked unless local repeat and oracle-overlap gates pass.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_closeout(source_dir=args.source, out_dir=args.out)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
