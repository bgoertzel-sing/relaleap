"""Select the next branch after sparse-coding basis bottleneck evidence.

This selector consumes the command-generated dense-teacher oracle sparse-coding
feasibility, deployable sparse-coding imitation, and flat-vs-sparse bottleneck
reports. It does not retrain models. Its role is to decide whether the current
orthogonal top-k residual basis should remain a promotable mechanism or be
demoted to a diagnostic before any GPU validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ORACLE_FEASIBILITY = Path("results/reports/dense_teacher_oracle_sparse_coding_feasibility/summary.json")
DEFAULT_IMITATION_PROBE = Path("results/reports/dense_teacher_deployable_sparse_coding_imitation_probe/summary.json")
DEFAULT_BOTTLENECK = Path("results/reports/dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_sparse_coding_basis_branch_selector")

REPAIR_ACTION = "repair_sparse_coding_basis_selector_sources"
DEMOTE_BASIS_ACTION = "demote_current_orthogonal_sparse_coding_basis_to_diagnostic"
ALTERNATIVE_BASIS_ACTION = "design_alternative_residual_basis_pregate"
RETRY_VALUE_HEAD_ACTION = "retry_same_basis_sparse_value_head"
GPU_ACTION = "launch_gpu_validation_for_sparse_coding_basis"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "basis_evidence_rows.csv",
    "branch_rows.csv",
    "gate_rows.csv",
    "notes.md",
)


def run_dense_teacher_sparse_coding_basis_branch_selector(
    *,
    oracle_feasibility_path: Path = DEFAULT_ORACLE_FEASIBILITY,
    imitation_probe_path: Path = DEFAULT_IMITATION_PROBE,
    bottleneck_path: Path = DEFAULT_BOTTLENECK,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a deterministic local sparse-coding basis branch decision."""

    start = time.time()
    oracle = _read_json(oracle_feasibility_path)
    imitation = _read_json(imitation_probe_path)
    bottleneck = _read_json(bottleneck_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("oracle_sparse_coding_feasibility", oracle_feasibility_path, oracle),
        _source_row("deployable_sparse_coding_imitation_probe", imitation_probe_path, imitation),
        _source_row("flat_vs_sparse_value_bottleneck_adjudicator", bottleneck_path, bottleneck),
        {
            "source": "gpt_5_5_pro_strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["verdict"],
            "claim_status": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}",
            "selected_next_step": strategy["recommended_next_action"],
            "training_executed": "",
            "advance_to_gpu_validation": "",
            "promotion_allowed": "",
        },
    ]
    evidence = _evidence(oracle, imitation, bottleneck, strategy)
    basis_evidence_rows = _basis_evidence_rows(evidence)
    gate_rows = _gate_rows(evidence, source_rows)
    failures = _failures(source_rows, gate_rows)
    branch_rows = _branch_rows(evidence, failures)
    selected = [row for row in branch_rows if row["disposition"] == "selected"]

    if failures or len(selected) != 1:
        status = "fail"
        decision = "dense_teacher_sparse_coding_basis_branch_selector_failed_closed"
        claim_status = "sparse_coding_basis_selector_sources_incomplete"
        selected_next_action = REPAIR_ACTION
        selected_next_step = "repair or regenerate sparse-coding basis selector source artifacts"
        rationale = "Required source artifacts are missing or runtime gates did not pass."
    else:
        selected_row = selected[0]
        status = "pass"
        decision = "dense_teacher_sparse_coding_basis_branch_selected"
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
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "backend_policy": "local CPU artifact selector only; Colab and RunPod remain blocked",
        "source_rows": source_rows,
        "basis_evidence_rows": basis_evidence_rows,
        "gate_rows": gate_rows,
        "branch_rows": branch_rows,
        "failures": failures,
        "evidence": evidence,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "direction_shift": _direction_shift(strategy, selected_next_action),
        "deferred_or_rejected_recommendations": [],
        "ben_notification_recommended": strategy["ben_notification_required"],
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    oracle: dict[str, Any],
    imitation: dict[str, Any],
    bottleneck: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    oracle_rows = _list(oracle.get("arm_metrics"))
    imitation_rows = _list(imitation.get("imitation_rows"))
    bottleneck_rows = _list(bottleneck.get("bottleneck_rows"))
    oracle_topk = _arm(oracle_rows, "oracle_topk_orthogonal_sparse_coding")
    oracle_flat = _arm(oracle_rows, "same_router_flat_value_control")
    combo = _arm(imitation_rows, "combo_mlp_router_scalar_imitation")
    conditioned = _arm(imitation_rows, "support_conditioned_combo_sparse_value_head")
    flat = _arm(imitation_rows, "same_router_flat_value_control")
    random_null = _arm(imitation_rows, "random_topk_sparse_coding_null")
    oracle_support_learned_coeff = _arm(imitation_rows, "oracle_support_learned_combo_coeff_sparse_coding")
    learned_support_oracle_coeff = _arm(imitation_rows, "learned_combo_support_oracle_coeff_sparse_coding")
    sparse_vs_flat = _comparison(bottleneck_rows, "best_deployable_sparse_vs_same_router_flat")
    conditioned_vs_combo = _comparison(bottleneck_rows, "naive_support_conditioned_value_head_vs_combo")
    sparse_vs_random = _comparison(bottleneck_rows, "best_deployable_sparse_vs_random_null")
    oracle_support_cross = _comparison(bottleneck_rows, "oracle_support_learned_coeff_vs_oracle_sparse")
    learned_support_cross = _comparison(bottleneck_rows, "learned_support_oracle_coeff_vs_oracle_sparse")
    best_sparse = combo
    if _float(conditioned.get("oracle_gain_retained_fraction"), -1.0) > _float(
        combo.get("oracle_gain_retained_fraction"), -1.0
    ):
        best_sparse = conditioned
    return {
        "oracle_status": oracle.get("status", ""),
        "oracle_claim_status": oracle.get("claim_status", ""),
        "imitation_status": imitation.get("status", ""),
        "imitation_claim_status": imitation.get("claim_status", ""),
        "bottleneck_status": bottleneck.get("status", ""),
        "bottleneck_claim_status": bottleneck.get("claim_status", ""),
        "oracle_sparse_r2": _float(oracle_topk.get("teacher_residual_reconstruction_r2")),
        "oracle_sparse_ce": _float(oracle_topk.get("ce")),
        "oracle_sparse_retention": _coalesce_float(
            oracle_topk.get("oracle_gain_retained_fraction"),
            oracle_topk.get("retention_proxy"),
        ),
        "oracle_flat_r2": _float(oracle_flat.get("teacher_residual_reconstruction_r2")),
        "oracle_flat_ce": _float(oracle_flat.get("ce")),
        "best_deployable_sparse_arm": best_sparse.get("arm", ""),
        "best_deployable_sparse_r2": _float(best_sparse.get("teacher_residual_reconstruction_r2")),
        "best_deployable_sparse_ce": _float(best_sparse.get("ce")),
        "best_deployable_sparse_retention": _float(best_sparse.get("oracle_gain_retained_fraction")),
        "flat_control_r2": _float(flat.get("teacher_residual_reconstruction_r2")),
        "flat_control_ce": _float(flat.get("ce")),
        "flat_control_retention": _float(flat.get("oracle_gain_retained_fraction")),
        "random_null_r2": _float(random_null.get("teacher_residual_reconstruction_r2")),
        "random_null_retention": _float(random_null.get("oracle_gain_retained_fraction")),
        "support_conditioned_r2": _float(conditioned.get("teacher_residual_reconstruction_r2")),
        "support_conditioned_retention": _float(conditioned.get("oracle_gain_retained_fraction")),
        "support_conditioned_coeff_mse": _float(conditioned.get("coefficient_mse_vs_oracle")),
        "combo_coeff_mse": _float(combo.get("coefficient_mse_vs_oracle")),
        "oracle_support_learned_coeff_retention": _float(
            oracle_support_learned_coeff.get("oracle_gain_retained_fraction")
        ),
        "learned_support_oracle_coeff_retention": _float(
            learned_support_oracle_coeff.get("oracle_gain_retained_fraction")
        ),
        "sparse_vs_flat_r2_delta": _float(sparse_vs_flat.get("r2_delta_primary_minus_comparator")),
        "sparse_vs_flat_retention_delta": _float(sparse_vs_flat.get("retention_delta_primary_minus_comparator")),
        "conditioned_vs_combo_r2_delta": _float(conditioned_vs_combo.get("r2_delta_primary_minus_comparator")),
        "conditioned_vs_combo_retention_delta": _float(
            conditioned_vs_combo.get("retention_delta_primary_minus_comparator")
        ),
        "sparse_vs_random_r2_delta": _float(sparse_vs_random.get("r2_delta_primary_minus_comparator")),
        "oracle_support_cross_retention_delta": _float(
            oracle_support_cross.get("retention_delta_primary_minus_comparator")
        ),
        "learned_support_cross_retention_delta": _float(
            learned_support_cross.get("retention_delta_primary_minus_comparator")
        ),
        "strategy_verdict": strategy["verdict"],
        "strategy_recommended_next_action": strategy["recommended_next_action"],
        "ben_notification_required": strategy["ben_notification_required"],
    }


def _basis_evidence_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _basis_row(
            "oracle_basis_feasibility",
            "oracle_topk_orthogonal_sparse_coding",
            evidence["oracle_sparse_r2"],
            evidence["oracle_sparse_ce"],
            evidence["oracle_sparse_retention"],
            "Oracle sparse coding is useful as a diagnostic ceiling.",
        ),
        _basis_row(
            "best_deployable_sparse",
            evidence["best_deployable_sparse_arm"],
            evidence["best_deployable_sparse_r2"],
            evidence["best_deployable_sparse_ce"],
            evidence["best_deployable_sparse_retention"],
            "Best deployable arm under the current orthogonal sparse-coding basis.",
        ),
        _basis_row(
            "same_router_flat_value_control",
            "same_router_flat_value_control",
            evidence["flat_control_r2"],
            evidence["flat_control_ce"],
            evidence["flat_control_retention"],
            "Generic flat value control used as the local blocker.",
        ),
        _basis_row(
            "random_topk_null",
            "random_topk_sparse_coding_null",
            evidence["random_null_r2"],
            "",
            evidence["random_null_retention"],
            "Random support null used to keep the sparse signal from being dismissed as noise.",
        ),
    ]


def _basis_row(kind: str, arm: Any, r2: Any, ce: Any, retention: Any, interpretation: str) -> dict[str, Any]:
    return {
        "evidence_type": kind,
        "arm": arm,
        "teacher_residual_reconstruction_r2": _round(r2),
        "ce": _round(ce),
        "oracle_gain_retained_fraction": _round(retention),
        "interpretation": interpretation,
    }


def _gate_rows(evidence: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_sources_present = not any(row["source"] != "gpt_5_5_pro_strategy_review" and not row["present"] for row in source_rows)
    local_sources_block_gpu = all(
        row["source"] == "gpt_5_5_pro_strategy_review"
        or (row["advance_to_gpu_validation"] is False and row["promotion_allowed"] is False)
        for row in source_rows
    )
    return [
        _gate("required_sources_present", required_sources_present, "required", "oracle, imitation, and bottleneck summaries"),
        _gate(
            "source_reports_passed",
            evidence["oracle_status"] == "pass"
            and evidence["imitation_status"] == "pass"
            and evidence["bottleneck_status"] == "pass",
            "required",
            f"oracle={evidence['oracle_status']}; imitation={evidence['imitation_status']}; bottleneck={evidence['bottleneck_status']}",
        ),
        _gate(
            "source_reports_block_gpu",
            local_sources_block_gpu,
            "required",
            "all consumed local reports have advance_to_gpu_validation=false and promotion_allowed=false",
        ),
        _gate(
            "oracle_sparse_basis_feasible",
            _gt(evidence["oracle_sparse_r2"], 0.75) and _gt(evidence["oracle_sparse_retention"], 0.95),
            "scientific",
            f"oracle_r2={evidence['oracle_sparse_r2']}; oracle_retention={evidence['oracle_sparse_retention']}",
        ),
        _gate(
            "deployable_sparse_signal_beats_null",
            _gt(evidence["sparse_vs_random_r2_delta"], 0.05),
            "scientific",
            f"sparse_vs_random_r2_delta={evidence['sparse_vs_random_r2_delta']}",
        ),
        _gate(
            "deployable_sparse_fails_oracle_retention_gate",
            not _gte(evidence["best_deployable_sparse_retention"], 0.80),
            "scientific",
            f"best_deployable_retention={evidence['best_deployable_sparse_retention']}",
        ),
        _gate(
            "flat_control_blocks_deployable_sparse_r2",
            _lt(evidence["sparse_vs_flat_r2_delta"], 0.0),
            "scientific",
            f"sparse_vs_flat_r2_delta={evidence['sparse_vs_flat_r2_delta']}",
        ),
        _gate(
            "support_conditioned_head_closed",
            _lt(evidence["conditioned_vs_combo_r2_delta"], 0.0)
            and _lt(evidence["conditioned_vs_combo_retention_delta"], 0.0),
            "scientific",
            (
                f"conditioned_vs_combo_r2_delta={evidence['conditioned_vs_combo_r2_delta']}; "
                f"retention_delta={evidence['conditioned_vs_combo_retention_delta']}"
            ),
        ),
        _gate(
            "crossed_support_coefficients_not_single_dominant_blocker",
            _gte(evidence["oracle_support_learned_coeff_retention"], 0.80)
            and _gte(evidence["learned_support_oracle_coeff_retention"], 0.80),
            "scientific",
            (
                f"oracle_support_learned_coeff_retention={evidence['oracle_support_learned_coeff_retention']}; "
                f"learned_support_oracle_coeff_retention={evidence['learned_support_oracle_coeff_retention']}"
            ),
        ),
        _gate(
            "gpu_validation_blocked",
            True,
            "required",
            "requires_gpu_now=false; advance_to_gpu_validation=false; promotion_allowed=false",
        ),
    ]


def _branch_rows(evidence: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if failures:
        return [
            _branch(
                REPAIR_ACTION,
                "selected",
                "required source artifacts are missing or runtime gates did not pass",
                "repair sparse-coding basis branch selector source artifacts",
                "sparse_coding_basis_selector_source_repair_required",
            )
        ]
    flat_blocks = _lt(evidence["sparse_vs_flat_r2_delta"], 0.0)
    retention_blocks = not _gte(evidence["best_deployable_sparse_retention"], 0.80)
    null_signal = _gt(evidence["sparse_vs_random_r2_delta"], 0.05)
    if flat_blocks and retention_blocks and null_signal:
        return [
            _branch(
                DEMOTE_BASIS_ACTION,
                "selected",
                (
                    "the orthogonal top-k sparse-coding basis is a useful oracle diagnostic and beats random nulls, "
                    "but its best deployable arm misses the 0.8 oracle-gain retention gate and loses R2 to the flat value control"
                ),
                (
                    "implement a local alternative residual-basis pregate comparing non-orthogonal learned dictionary, "
                    "CE-gradient-aligned basis, and low-rank/SVD controls before any GPU validation"
                ),
                "orthogonal_sparse_coding_basis_demoted_to_diagnostic_no_gpu",
            ),
            _branch(
                ALTERNATIVE_BASIS_ACTION,
                "queued",
                "basis demotion implies the next scientific step should change basis/value formulation rather than retrain the same head",
                "design a local alternative residual-basis pregate with flat/null controls",
                "alternative_residual_basis_pregate_queued",
            ),
            _branch(
                RETRY_VALUE_HEAD_ACTION,
                "rejected",
                "the support-conditioned same-basis value head was already worse than the combo arm",
                "do not train another same-basis sparse coefficient head without changing the basis or value formulation",
                "same_basis_sparse_value_head_retry_rejected",
            ),
            _branch(
                GPU_ACTION,
                "rejected",
                "local retention and flat-control gates still block promotion evidence",
                "do not run Colab or RunPod validation for this sparse-coding basis",
                "gpu_validation_blocked_by_local_basis_gates",
            ),
        ]
    return [
        _branch(
            ALTERNATIVE_BASIS_ACTION,
            "selected",
            "the current basis is not promotable, but source gates did not support a hard demotion label",
            "write a local alternative residual-basis pregate before backend validation",
            "alternative_residual_basis_pregate_selected_no_gpu",
        ),
        _branch(
            GPU_ACTION,
            "rejected",
            "no selector path permits backend validation directly",
            "keep GPU validation blocked until a local arm clears retention and flat-control gates",
            "gpu_validation_blocked_by_selector",
        ),
    ]


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(payload),
        "status": payload.get("status", "") if payload else "missing",
        "decision": payload.get("decision", "") if payload else "",
        "claim_status": payload.get("claim_status", "") if payload else "",
        "selected_next_step": payload.get("selected_next_step", "") if payload else "",
        "training_executed": payload.get("training_executed", "") if payload else "",
        "advance_to_gpu_validation": payload.get("advance_to_gpu_validation", "") if payload else "",
        "promotion_allowed": payload.get("promotion_allowed", "") if payload else "",
    }


def _failures(source_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        {"source": row["source"], "reason": "missing_required_source", "path": row["path"]}
        for row in source_rows
        if row["source"] != "gpt_5_5_pro_strategy_review" and not row["present"]
    ]
    failures.extend(row for row in gate_rows if row["gate_type"] == "required" and not row["passed"])
    return failures


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    fields: dict[str, Any] = {
        "present": bool(text),
        "strategic_change_level": "",
        "notify_ben": "",
        "recommended_next_action": "",
        "verdict": "",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in fields:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields["notify_ben"]).lower() == "true" or fields["strategic_change_level"] == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No external strategy review was present; followed the status-file sparse-basis selector step."
    return (
        "Accepted the GPT-5.5-Pro recommendation to keep this local and mechanistic before GPU. "
        f"Review header: strategic_change_level={strategy['strategic_change_level']}, notify_ben={strategy['notify_ben']}; "
        "Ben should be notified when requested by the header."
    )


def _direction_shift(strategy: dict[str, Any], selected_next_action: str) -> dict[str, Any]:
    return {
        "strategic_change_level": strategy["strategic_change_level"],
        "ben_should_be_notified": bool(strategy["ben_notification_required"]),
        "selected_next_action": selected_next_action,
        "direction": (
            "demote the current orthogonal sparse-coding basis to diagnostic status and require a local alternative "
            "basis/value-formulation pregate before any GPU validation"
        ),
        "recommendation_disposition": "accepted",
        "deferred_or_rejected_recommendations": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _arm(rows: list[Any], arm: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("arm") == arm:
            return row
    return {}


def _comparison(rows: list[Any], name: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get("comparison") == name:
            return row
    return {}


def _gate(name: str, passed: bool, gate_type: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "gate_type": gate_type, "evidence": evidence}


def _branch(action: str, disposition: str, reason: str, next_step: str, claim_status: str) -> dict[str, str]:
    return {
        "candidate_action": action,
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "claim_status": claim_status,
    }


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coalesce_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _round(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return round(number, 6)


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "basis_evidence_rows.csv", summary["basis_evidence_rows"])
    _write_csv(out_dir / "branch_rows.csv", summary["branch_rows"])
    _write_csv(out_dir / "gate_rows.csv", summary["gate_rows"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _notes(summary: dict[str, Any]) -> str:
    failed = [row["gate"] for row in summary["gate_rows"] if not row["passed"]]
    return "\n".join(
        [
            "# Dense Teacher Sparse-Coding Basis Branch Selector",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Claim status: `{summary['claim_status']}`",
            f"- Selected next action: `{summary['selected_next_action']}`",
            f"- Selected next step: {summary['selected_next_step']}",
            "- GPU validation remains blocked: `requires_gpu_now=false`, `advance_to_gpu_validation=false`, `promotion_allowed=false`.",
            f"- Failed scientific gates: {', '.join(failed) if failed else 'none'}",
            f"- Strategy review handling: {summary['strategy_review_handling']}",
            "",
            "## Rationale",
            "",
            str(summary["rationale"]),
            "",
        ]
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-feasibility", type=Path, default=DEFAULT_ORACLE_FEASIBILITY)
    parser.add_argument("--imitation-probe", type=Path, default=DEFAULT_IMITATION_PROBE)
    parser.add_argument("--bottleneck", type=Path, default=DEFAULT_BOTTLENECK)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_sparse_coding_basis_branch_selector(
        oracle_feasibility_path=args.oracle_feasibility,
        imitation_probe_path=args.imitation_probe,
        bottleneck_path=args.bottleneck,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
