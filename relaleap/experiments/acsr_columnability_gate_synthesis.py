"""Synthesize the local ACSR columnability/discovery gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("results/reports/acsr_common_causal_residual_benchmark_seed2")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_columnability_gate_synthesis_seed2")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_packets.csv",
    "columnability_synthesis.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_columnability_gate_synthesis(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Interpret a completed local sparse/dense columnability benchmark."""

    start = time.time()
    source = _load_source(source_dir)
    review = _strategy_review(strategy_review)
    synthesis_rows = _synthesis_rows(source)
    gate_rows = _gate_rows(source, synthesis_rows, review)
    failures = [
        {"gate": row["criterion"], "reason": row["failure_reason"]}
        for row in gate_rows
        if not row["passed"]
    ]
    status = "pass" if not failures else "fail"
    decision = _decision(status, synthesis_rows)
    claim_status = _claim_status(status, synthesis_rows)
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": _selected_next_step(status, decision, synthesis_rows),
        "source_dir": str(source_dir),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "source_packets": [_source_packet_row(source)],
        "columnability_synthesis": synthesis_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "aggregate_metrics": _aggregate_metrics(synthesis_rows),
        "claim_boundaries": {
            "supported": [
                "the local seed-2 benchmark is now auditable with clean git provenance",
                "teacher-distilled sparse columns weakly separate from shuffled-teacher null",
                "oracle-support distillation improves over target-norm support, marking learned support discovery as a secondary bottleneck candidate",
            ],
            "not_supported": [
                "sparse-support identity as the primary causal substrate",
                "default-router mechanism claims based on CE/support entropy alone",
                "RunPod replication before a stricter local validation target is defined",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_source(source_dir: Path) -> dict[str, Any]:
    summary_path = source_dir / "summary.json"
    arm_path = source_dir / "arm_metrics.csv"
    gate_path = source_dir / "gate_criteria.csv"
    return {
        "source_dir": str(source_dir),
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "arm_metrics_path": str(arm_path),
        "arm_metrics_present": arm_path.is_file(),
        "gate_criteria_path": str(gate_path),
        "gate_criteria_present": gate_path.is_file(),
        "summary": _read_json(summary_path),
        "arm_rows": _read_csv(arm_path),
        "gate_rows": _read_csv(gate_path),
    }


def _source_packet_row(source: dict[str, Any]) -> dict[str, Any]:
    summary = source["summary"]
    return {
        "source_dir": source["source_dir"],
        "summary_present": source["summary_present"],
        "arm_metrics_present": source["arm_metrics_present"],
        "gate_criteria_present": source["gate_criteria_present"],
        "source_status": summary.get("status", ""),
        "source_decision": summary.get("decision", ""),
        "source_claim_status": summary.get("claim_status", ""),
        "source_git_dirty": summary.get("git_dirty", ""),
        "source_git_diff_hash": summary.get("git_diff_hash", ""),
        "arm_count": summary.get("arm_count", ""),
    }


def _synthesis_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    summary = source["summary"]
    interpretation = _as_dict(summary.get("benchmark_interpretation"))
    arms = {str(row.get("arm")): row for row in source["arm_rows"]}
    sparse = arms.get("sparse_contextual_topk2", {})
    dense = arms.get("rank_flop_matched_causal_dense", {})
    current = arms.get("sparse_teacher_distilled_norm_topk2", {})
    target = arms.get("sparse_teacher_distilled_target_norm_topk2", {})
    oracle = arms.get("sparse_teacher_distilled_oracle_support_topk2", {})
    soft = arms.get("sparse_teacher_distilled_soft_temperature_topk2", {})
    shuffled = arms.get("sparse_teacher_distilled_shuffled_teacher_null", {})
    token_position = arms.get("sparse_teacher_distilled_token_position_null", {})
    return [
        {
            "criterion": "dense_teacher_vs_default_sparse",
            "status": "blocked",
            "interpretation": "dense residual remains stronger than default sparse top-k2",
            "sparse_heldout_delta_vs_base_ce": _number(sparse.get("heldout_delta_vs_base_ce")),
            "dense_heldout_delta_vs_base_ce": _number(dense.get("heldout_delta_vs_base_ce")),
            "gap_dense_minus_sparse_ce_delta": _delta_number(dense.get("heldout_delta_vs_base_ce"), sparse.get("heldout_delta_vs_base_ce")),
            "passes_sparse_identity": not bool(interpretation.get("dense_wins_l2_matched")),
        },
        {
            "criterion": "teacher_distill_rescue",
            "status": "blocked",
            "interpretation": "teacher-distilled sparse separates from shuffled teacher but remains far behind default sparse and dense",
            "current_distill_heldout_delta_vs_base_ce": _number(current.get("heldout_delta_vs_base_ce")),
            "current_distill_teacher_mse": _number(current.get("teacher_residual_mse")),
            "shuffled_teacher_mse": _number(shuffled.get("teacher_residual_mse")),
            "mse_margin_vs_shuffled_teacher": interpretation.get("teacher_distilled_mse_margin_vs_shuffled_teacher", ""),
            "gap_vs_default_sparse_ce_delta": interpretation.get("teacher_distilled_gap_vs_default_sparse_ce_delta", ""),
            "gap_vs_dense_ce_delta": interpretation.get("teacher_distilled_gap_vs_l2_matched_dense_ce_delta", ""),
            "passes_sparse_identity": bool(interpretation.get("teacher_distilled_sparse_beats_default_sparse")),
        },
        {
            "criterion": "norm_calibration_rescue",
            "status": "blocked",
            "interpretation": "target-norm scaling improves CE but worsens teacher residual MSE, so norm calibration does not rescue representation quality",
            "target_norm_distill_heldout_delta_vs_base_ce": _number(target.get("heldout_delta_vs_base_ce")),
            "target_norm_distill_teacher_mse": _number(target.get("teacher_residual_mse")),
            "target_norm_mse_margin_vs_current": interpretation.get("target_norm_distill_mse_margin_vs_current", ""),
            "target_norm_gap_vs_default_sparse_ce_delta": interpretation.get("target_norm_distill_gap_vs_default_sparse_ce_delta", ""),
            "passes_sparse_identity": _positive(interpretation.get("target_norm_distill_mse_margin_vs_current")),
        },
        {
            "criterion": "oracle_support_followup",
            "status": "secondary_followup_warranted",
            "interpretation": "oracle support improves hard distillation MSE but CE remains well behind default sparse, so support discovery is a follow-up only after retiring identity-first claims",
            "oracle_distill_heldout_delta_vs_base_ce": _number(oracle.get("heldout_delta_vs_base_ce")),
            "oracle_distill_teacher_mse": _number(oracle.get("teacher_residual_mse")),
            "oracle_mse_margin_vs_target_norm": interpretation.get("oracle_support_distill_mse_margin_vs_target_norm", ""),
            "oracle_gap_vs_default_sparse_ce_delta": interpretation.get("oracle_support_distill_gap_vs_default_sparse_ce_delta", ""),
            "passes_sparse_identity": False,
        },
        {
            "criterion": "soft_sparse_ceiling",
            "status": "blocked",
            "interpretation": "soft all-column sparse mixture is worse than the best hard sparse distill ceiling",
            "soft_distill_heldout_delta_vs_base_ce": _number(soft.get("heldout_delta_vs_base_ce")),
            "soft_distill_teacher_mse": _number(soft.get("teacher_residual_mse")),
            "soft_mse_margin_vs_best_hard_sparse": interpretation.get("soft_topk_distill_mse_margin_vs_best_hard_sparse", ""),
            "soft_gap_vs_default_sparse_ce_delta": interpretation.get("soft_topk_distill_gap_vs_default_sparse_ce_delta", ""),
            "passes_sparse_identity": _positive(interpretation.get("soft_topk_distill_mse_margin_vs_best_hard_sparse")),
        },
        {
            "criterion": "token_position_null_control",
            "status": "passes_control",
            "interpretation": "target-norm distill beats token/position support null, but this control is insufficient to restore sparse-support identity",
            "target_norm_distill_teacher_mse": _number(target.get("teacher_residual_mse")),
            "token_position_teacher_mse": _number(token_position.get("teacher_residual_mse")),
            "target_norm_beats_token_position_null": interpretation.get("target_norm_distill_beats_token_position_null", ""),
            "passes_sparse_identity": False,
        },
    ]


def _gate_rows(
    source: dict[str, Any],
    synthesis_rows: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = source["summary"]
    interpretation = _as_dict(summary.get("benchmark_interpretation"))
    required_arms = {
        "sparse_contextual_topk2",
        "rank_flop_matched_causal_dense",
        "sparse_teacher_distilled_norm_topk2",
        "sparse_teacher_distilled_target_norm_topk2",
        "sparse_teacher_distilled_oracle_support_topk2",
        "sparse_teacher_distilled_soft_temperature_topk2",
        "sparse_teacher_distilled_shuffled_teacher_null",
        "sparse_teacher_distilled_token_position_null",
    }
    observed_arms = {str(row.get("arm")) for row in source["arm_rows"]}
    return [
        _criterion(
            "strategy_review_consumed",
            review.get("status") == "read",
            "latest strategy review is read before synthesis",
            review.get("status", ""),
            "latest strategy review was not available/read",
        ),
        _criterion(
            "source_artifacts_present",
            source["summary_present"] and source["arm_metrics_present"] and source["gate_criteria_present"],
            "source benchmark summary, arm metrics, and gate criteria are present",
            _source_packet_row(source),
            "one or more source benchmark artifacts are missing",
        ),
        _criterion(
            "source_provenance_clean",
            summary.get("git_dirty") is False and not summary.get("git_diff_hash"),
            "source benchmark records clean git provenance",
            {"git_dirty": summary.get("git_dirty"), "git_diff_hash": summary.get("git_diff_hash")},
            "source benchmark was generated from dirty or unknown code provenance",
        ),
        _criterion(
            "required_columnability_arms_present",
            required_arms.issubset(observed_arms),
            "all sparse/dense teacher, oracle, soft, and null arms are present",
            {"missing": sorted(required_arms - observed_arms), "observed_count": len(observed_arms)},
            "source benchmark is missing one or more required columnability arms",
        ),
        _criterion(
            "source_gate_reached_interpretable_failure",
            summary.get("status") == "fail"
            and summary.get("selected_next_step") == "synthesize the local columnability/discovery gate before any RunPod repeat or sparse-support identity claim",
            "source benchmark failed at an interpretable columnability/discovery gate",
            {"status": summary.get("status"), "selected_next_step": summary.get("selected_next_step")},
            "source benchmark did not reach the expected local synthesis gate",
        ),
        _criterion(
            "synthesis_has_retirement_basis",
            bool(interpretation.get("dense_wins_l2_matched"))
            and not bool(interpretation.get("teacher_distilled_sparse_beats_default_sparse"))
            and not _positive(interpretation.get("target_norm_distill_mse_margin_vs_current"))
            and not _positive(interpretation.get("soft_topk_distill_mse_margin_vs_best_hard_sparse")),
            "dense win plus failed teacher/norm/soft sparse rescues provide a retirement basis",
            _aggregate_metrics(synthesis_rows),
            "local evidence is not yet enough to retire sparse-support identity as the primary claim",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _decision(status: str, rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "acsr_columnability_gate_synthesis_failed_closed"
    if any(row["status"] == "secondary_followup_warranted" for row in rows):
        return "retire_sparse_support_identity_primary_claim_keep_support_discovery_followup_optional"
    return "retire_sparse_support_identity_primary_claim"


def _claim_status(status: str, rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "columnability_synthesis_not_interpretable"
    if any(bool(row.get("passes_sparse_identity")) for row in rows):
        return "sparse_support_identity_not_retired"
    return "sparse_support_identity_primary_claim_retired_locally"


def _selected_next_step(status: str, decision: str, rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "repair source benchmark artifacts or provenance before making a sparse-support identity decision"
    if "keep_support_discovery_followup_optional" in decision:
        return "write a bounded support-discovery follow-up design, or proceed to dense/residual interference assays; do not run RunPod yet"
    return "move sparse columns to diagnostic/active-compute comparator role and prioritize dense/residual interference assays before GPU repeats"


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_criterion = {str(row.get("criterion")): row for row in rows}
    dense = by_criterion.get("dense_teacher_vs_default_sparse", {})
    teacher = by_criterion.get("teacher_distill_rescue", {})
    target = by_criterion.get("norm_calibration_rescue", {})
    oracle = by_criterion.get("oracle_support_followup", {})
    soft = by_criterion.get("soft_sparse_ceiling", {})
    return {
        "dense_minus_sparse_ce_delta": dense.get("gap_dense_minus_sparse_ce_delta"),
        "teacher_distill_gap_vs_default_sparse_ce_delta": teacher.get("gap_vs_default_sparse_ce_delta"),
        "teacher_distill_mse_margin_vs_shuffled_teacher": teacher.get("mse_margin_vs_shuffled_teacher"),
        "target_norm_mse_margin_vs_current": target.get("target_norm_mse_margin_vs_current"),
        "oracle_support_mse_margin_vs_target_norm": oracle.get("oracle_mse_margin_vs_target_norm"),
        "soft_topk_mse_margin_vs_best_hard_sparse": soft.get("soft_mse_margin_vs_best_hard_sparse"),
        "retire_sparse_identity_primary_claim": not any(bool(row.get("passes_sparse_identity")) for row in rows),
        "secondary_support_discovery_followup_warranted": any(
            row.get("status") == "secondary_followup_warranted" for row in rows
        ),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "not_found", "recommendation_accepted": False}
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "status": "read",
        "recommendation_accepted": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _direction_shift(review: dict[str, Any]) -> str:
    if review.get("strategic_change_level") == "major" or review.get("notify_ben") == "true":
        return (
            "GPT-5.5-Pro review requested a major or notify-Ben shift away from "
            "sparse-support identity rescue/replication and toward local "
            "columnability-vs-discovery triage. Ben should be notified."
        )
    return "No major strategy-review direction shift recorded for this synthesis."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_packets.csv", summary["source_packets"])
    _write_csv(out_dir / "columnability_synthesis.csv", summary["columnability_synthesis"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["aggregate_metrics"]
    lines = [
        "# ACSR Columnability Gate Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Dense-minus-sparse CE delta gap: `{metrics['dense_minus_sparse_ce_delta']}`",
        f"- Teacher-distill gap vs default sparse CE delta: `{metrics['teacher_distill_gap_vs_default_sparse_ce_delta']}`",
        f"- Oracle-support MSE margin vs target-norm: `{metrics['oracle_support_mse_margin_vs_target_norm']}`",
        "",
        summary["direction_shift"],
        "",
        "This report treats CE as a guardrail and the sparse-support identity claim "
        "as unsupported unless sparse teacher/norm/soft/oracle arms can approach "
        "the common dense residual control with clean provenance.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_number(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_columnability_gate_synthesis(
        source_dir=args.source_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
