"""Gate local ACSR support-discovery evidence after columnability retirement."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SYNTHESIS_DIR = Path("results/reports/acsr_columnability_gate_synthesis_seed2")
DEFAULT_BENCHMARK_DIR = Path("results/reports/acsr_common_causal_residual_benchmark_seed2")
DEFAULT_DESIGN_DIR = Path("results/reports/acsr_support_discovery_followup_design_seed2")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_support_discovery_gate_seed2")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_reports.csv",
    "support_discovery_gate.csv",
    "null_controls.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_support_discovery_gate(
    *,
    synthesis_dir: Path = DEFAULT_SYNTHESIS_DIR,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_oracle_ce_headroom: float = 0.01,
    min_oracle_distill_mse_margin: float = 0.001,
) -> dict[str, Any]:
    """Write a fail-closed local gate for deployable support discovery."""

    start = time.time()
    synthesis = _read_json(synthesis_dir / "summary.json")
    benchmark = _read_json(benchmark_dir / "summary.json")
    design = _read_json(design_dir / "summary.json")
    arms = _read_csv(benchmark_dir / "arm_metrics.csv")
    fingerprints = _read_csv(benchmark_dir / "intervention_fingerprints.csv")
    review = _strategy_review(strategy_review)
    source_rows = _source_rows(synthesis_dir, benchmark_dir, design_dir, synthesis, benchmark, design, arms, fingerprints, review)
    gate_rows = _support_gate_rows(
        synthesis,
        benchmark,
        design,
        arms,
        fingerprints,
        min_oracle_ce_headroom=min_oracle_ce_headroom,
        min_oracle_distill_mse_margin=min_oracle_distill_mse_margin,
    )
    null_rows = _null_control_rows(arms, benchmark)
    criteria = _criteria_rows(synthesis, benchmark, design, arms, fingerprints, review, gate_rows, null_rows)
    hard_failures = [row for row in criteria if not row["passed"] and row["severity"] == "hard"]
    blockers = [row for row in criteria if not row["passed"] and row["severity"] == "claim_blocker"]
    status = "pass" if not hard_failures else "fail"
    deployable_positive = status == "pass" and not blockers and all(bool(row.get("passed")) for row in gate_rows)
    summary = {
        "status": status,
        "decision": (
            "support_discovery_gate_positive_ready_for_local_repeat"
            if deployable_positive
            else (
                "support_discovery_gate_blocks_deployable_claim_pending_learned_head"
                if status == "pass"
                else "support_discovery_gate_failed_closed"
            )
        ),
        "claim_status": (
            "deployable_support_discovery_local_gate_positive_not_gpu_validated"
            if deployable_positive
            else (
                "deployable_support_discovery_not_established_sparse_identity_retired"
                if status == "pass"
                else "support_discovery_gate_not_interpretable"
            )
        ),
        "selected_next_step": _selected_next_step(status, deployable_positive),
        "next_command": (
            "./.venv-conda/bin/python -m relaleap.experiments.acsr_support_discovery_gate "
            "--out results/reports/acsr_support_discovery_gate_seed2"
        ),
        "source_reports": source_rows,
        "support_discovery_gate": gate_rows,
        "null_controls": null_rows,
        "gate_criteria": criteria,
        "failures": hard_failures,
        "claim_blockers": blockers,
        "aggregate_metrics": _aggregate_metrics(arms, synthesis),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "claim_boundaries": {
            "supported": [
                "sparse-support identity remains retired locally",
                "oracle support and teacher-oracle distillation provide only secondary support-discovery headroom",
                "RunPod remains deferred until a positive local deployable gate exists",
            ],
            "not_supported": [
                "deployable learned support discovery",
                "same-student fixed-support causal mechanism",
                "dual-student cross-forcing mechanism",
                "default-router promotion or GPU validation target",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    synthesis_dir: Path,
    benchmark_dir: Path,
    design_dir: Path,
    synthesis: dict[str, Any],
    benchmark: dict[str, Any],
    design: dict[str, Any],
    arms: list[dict[str, Any]],
    fingerprints: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _source("columnability_synthesis", synthesis_dir / "summary.json", synthesis),
        _source("common_causal_residual_benchmark", benchmark_dir / "summary.json", benchmark),
        _source("support_discovery_followup_design", design_dir / "summary.json", design),
        {
            "source": "benchmark_arm_metrics",
            "path": str(benchmark_dir / "arm_metrics.csv"),
            "present": bool(arms),
            "status": "present" if arms else "missing",
            "decision": f"rows={len(arms)}",
            "claim_status": "support_oracle_null_dense_arms",
            "git_commit": "",
        },
        {
            "source": "benchmark_intervention_fingerprints",
            "path": str(benchmark_dir / "intervention_fingerprints.csv"),
            "present": bool(fingerprints),
            "status": "present" if fingerprints else "missing",
            "decision": f"rows={len(fingerprints)}",
            "claim_status": "support_overlap_rows",
            "git_commit": "",
        },
        {
            "source": "strategy_review",
            "path": review.get("path", ""),
            "present": review.get("status") == "read",
            "status": review.get("status", "missing"),
            "decision": review.get("recommended_next_action", ""),
            "claim_status": (
                f"strategic_change_level={review.get('strategic_change_level', '')}; "
                f"notify_ben={review.get('notify_ben', '')}"
            ),
            "git_commit": "",
        },
    ]


def _source(name: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": name,
        "path": str(path),
        "present": bool(summary),
        "status": summary.get("status", "missing"),
        "decision": summary.get("decision", ""),
        "claim_status": summary.get("claim_status", ""),
        "git_commit": summary.get("git_commit", ""),
    }


def _support_gate_rows(
    synthesis: dict[str, Any],
    benchmark: dict[str, Any],
    design: dict[str, Any],
    arms: list[dict[str, Any]],
    fingerprints: list[dict[str, Any]],
    *,
    min_oracle_ce_headroom: float,
    min_oracle_distill_mse_margin: float,
) -> list[dict[str, Any]]:
    metrics = _aggregate_metrics(arms, synthesis)
    interpretation = _as_dict(benchmark.get("benchmark_interpretation"))
    return [
        {
            "component": "identity_retirement_guardrail",
            "metric": "retire_sparse_identity_primary_claim",
            "observed": metrics.get("retire_sparse_identity_primary_claim", ""),
            "threshold": "must be true",
            "passed": metrics.get("retire_sparse_identity_primary_claim") is True,
            "interpretation": "support discovery remains secondary; sparse identity is not revived",
        },
        {
            "component": "oracle_ce_headroom",
            "metric": "sparse_oracle_minus_sparse_default_heldout_ce_delta",
            "observed": metrics.get("sparse_oracle_minus_sparse_default_heldout_ce_delta"),
            "threshold": f"<= {-abs(min_oracle_ce_headroom)}",
            "passed": _number(metrics.get("sparse_oracle_minus_sparse_default_heldout_ce_delta")) is not None
            and float(metrics["sparse_oracle_minus_sparse_default_heldout_ce_delta"]) <= -abs(min_oracle_ce_headroom),
            "interpretation": "oracle support must create enough CE headroom to justify support discovery as a deployable target",
        },
        {
            "component": "teacher_oracle_distill_margin",
            "metric": "oracle_support_mse_margin_vs_target_norm",
            "observed": metrics.get("oracle_support_mse_margin_vs_target_norm"),
            "threshold": f">= {min_oracle_distill_mse_margin}",
            "passed": _number(metrics.get("oracle_support_mse_margin_vs_target_norm")) is not None
            and float(metrics["oracle_support_mse_margin_vs_target_norm"]) >= min_oracle_distill_mse_margin,
            "interpretation": "oracle support should improve teacher-residual representation quality before learned discovery is meaningful",
        },
        {
            "component": "token_position_null_margin",
            "metric": "target_norm_distill_beats_token_position_null",
            "observed": interpretation.get("target_norm_distill_beats_token_position_null", ""),
            "threshold": "must be true",
            "passed": bool(interpretation.get("target_norm_distill_beats_token_position_null")),
            "interpretation": "support discovery must beat token/position-only structure",
        },
        {
            "component": "shuffled_feature_support_head_null",
            "metric": "learned_support_head_vs_shuffled_causal_feature_null",
            "observed": "not_present",
            "threshold": "must be present and beaten for deployable claim",
            "passed": False,
            "interpretation": "current source packet has dense shuffled-feature nulls, but not a learned support-head shuffled-feature control",
        },
        {
            "component": "same_student_fixed_support_forcing",
            "metric": "same_student_support_forcing",
            "observed": "not_present",
            "threshold": "must be present for mechanism claim",
            "passed": False,
            "interpretation": "current source packet does not yet force learned, oracle, and null supports through identical student values",
        },
        {
            "component": "intervention_fingerprint_basis",
            "metric": "fingerprint_rows",
            "observed": len(fingerprints),
            "threshold": "> 0",
            "passed": bool(fingerprints),
            "interpretation": "support overlap fingerprints are available as a weak source basis, not a causal mechanism proof",
        },
        {
            "component": "followup_design_basis",
            "metric": "design_status",
            "observed": design.get("status", ""),
            "threshold": "pass",
            "passed": design.get("status") == "pass",
            "interpretation": "the prior design artifact must authorize a local-only gate before GPU work",
        },
    ]


def _null_control_rows(arms: list[dict[str, Any]], benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    sparse = _arm(arms, "sparse_contextual_topk2")
    oracle = _arm(arms, "sparse_oracle_support")
    token_position = _arm(arms, "sparse_token_position_null")
    teacher_token_position = _arm(arms, "sparse_teacher_distilled_token_position_null")
    shuffled_teacher = _arm(arms, "sparse_teacher_distilled_shuffled_teacher_null")
    shuffled_support = _arm(arms, "sparse_shuffled_support_marginals")
    frequency = _arm(arms, "sparse_frequency_matched_random")
    dense_shuffled = _arm(arms, "rank_flop_matched_shuffled_causal_feature_dense_null")
    return [
        _null_row("oracle_support", oracle, sparse, "nondeployable oracle ceiling"),
        _null_row("token_position_support_null", token_position, sparse, "token/position-only support control"),
        _null_row("teacher_token_position_distill_null", teacher_token_position, _arm(arms, "sparse_teacher_distilled_target_norm_topk2"), "teacher distill token/position-only control"),
        _null_row("shuffled_teacher_distill_null", shuffled_teacher, _arm(arms, "sparse_teacher_distilled_norm_topk2"), "shuffled teacher residual control"),
        _null_row("shuffled_support_marginals_null", shuffled_support, sparse, "support marginal shuffle control"),
        _null_row("frequency_matched_random_support_null", frequency, sparse, "frequency-matched random support control"),
        _null_row("dense_shuffled_causal_feature_null", dense_shuffled, _arm(arms, "rank_flop_matched_causal_dense"), "dense residual shuffled causal-feature control"),
        {
            "control": "learned_support_head_shuffled_feature_null",
            "present": False,
            "heldout_delta_vs_base_ce": "",
            "gap_vs_reference_heldout_ce_delta": "",
            "interpretation": "missing required support-head null for deployable support-discovery claim",
        },
        {
            "control": "benchmark_interpretation",
            "present": bool(benchmark.get("benchmark_interpretation")),
            "heldout_delta_vs_base_ce": "",
            "gap_vs_reference_heldout_ce_delta": "",
            "interpretation": json.dumps(_as_dict(benchmark.get("benchmark_interpretation")), sort_keys=True),
        },
    ]


def _null_row(control: str, row: dict[str, Any], reference: dict[str, Any], interpretation: str) -> dict[str, Any]:
    return {
        "control": control,
        "present": bool(row),
        "heldout_delta_vs_base_ce": _number(row.get("heldout_delta_vs_base_ce")),
        "gap_vs_reference_heldout_ce_delta": _delta_number(
            row.get("heldout_delta_vs_base_ce"),
            reference.get("heldout_delta_vs_base_ce"),
        ),
        "interpretation": interpretation,
    }


def _criteria_rows(
    synthesis: dict[str, Any],
    benchmark: dict[str, Any],
    design: dict[str, Any],
    arms: list[dict[str, Any]],
    fingerprints: list[dict[str, Any]],
    review: dict[str, Any],
    gate_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observed_arms = {str(row.get("arm")) for row in arms}
    required_arms = {
        "sparse_contextual_topk2",
        "sparse_oracle_support",
        "sparse_token_position_null",
        "sparse_shuffled_support_marginals",
        "sparse_frequency_matched_random",
        "sparse_teacher_distilled_target_norm_topk2",
        "sparse_teacher_distilled_token_position_null",
        "sparse_teacher_distilled_shuffled_teacher_null",
    }
    return [
        _criterion("strategy_review_consumed", review.get("status") == "read", "hard", "latest strategy review is read", review.get("status", ""), "strategy review was not read"),
        _criterion("source_artifacts_present", bool(synthesis and benchmark and design and arms), "hard", "synthesis, benchmark, design, and arm metrics exist", {"synthesis": bool(synthesis), "benchmark": bool(benchmark), "design": bool(design), "arms": len(arms)}, "one or more source artifacts are missing"),
        _criterion("source_provenance_clean", benchmark.get("git_dirty") is False and not benchmark.get("git_diff_hash"), "hard", "benchmark source records clean git provenance", {"git_dirty": benchmark.get("git_dirty"), "git_diff_hash": benchmark.get("git_diff_hash")}, "benchmark provenance is dirty or unknown"),
        _criterion("identity_claim_retired", synthesis.get("claim_status") == "sparse_support_identity_primary_claim_retired_locally", "hard", "synthesis has retired sparse-support identity", synthesis.get("claim_status", ""), "sparse-support identity was not retired before this gate"),
        _criterion("followup_design_passed", design.get("status") == "pass", "hard", "support-discovery follow-up design passed", design.get("status", ""), "support-discovery design did not pass"),
        _criterion("required_source_arms_present", required_arms.issubset(observed_arms), "hard", "oracle, sparse, teacher, and null arms are present", {"missing": sorted(required_arms - observed_arms), "observed_count": len(observed_arms)}, "required source arms are missing"),
        _criterion("fingerprints_present", bool(fingerprints), "hard", "support intervention fingerprints exist", f"rows={len(fingerprints)}", "fingerprint rows are missing"),
        _criterion("oracle_support_headroom_positive", _gate_passed(gate_rows, "oracle_ce_headroom"), "claim_blocker", "oracle support has nontrivial CE headroom", _gate_observed(gate_rows, "oracle_ce_headroom"), "oracle CE headroom is too small for a deployable support-discovery target"),
        _criterion("teacher_oracle_distill_margin_positive", _gate_passed(gate_rows, "teacher_oracle_distill_margin"), "claim_blocker", "oracle support improves teacher distill MSE", _gate_observed(gate_rows, "teacher_oracle_distill_margin"), "oracle support does not improve teacher residual representation enough"),
        _criterion("token_position_null_beaten", _gate_passed(gate_rows, "token_position_null_margin"), "claim_blocker", "source beats token/position null", _gate_observed(gate_rows, "token_position_null_margin"), "token/position null is not beaten"),
        _criterion("support_head_shuffled_feature_null_present", _null_present(null_rows, "learned_support_head_shuffled_feature_null") and _gate_passed(gate_rows, "shuffled_feature_support_head_null"), "claim_blocker", "learned support head beats shuffled-feature null", "missing", "learned support-head shuffled-feature null is absent"),
        _criterion("same_student_forcing_present", _gate_passed(gate_rows, "same_student_fixed_support_forcing"), "claim_blocker", "same-student fixed-support forcing exists", "missing", "same-student support forcing is absent"),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _aggregate_metrics(arms: list[dict[str, Any]], synthesis: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(synthesis.get("aggregate_metrics")).copy()
    sparse = _arm(arms, "sparse_contextual_topk2")
    oracle = _arm(arms, "sparse_oracle_support")
    if sparse and oracle:
        metrics["sparse_oracle_minus_sparse_default_heldout_ce_delta"] = _delta_number(
            oracle.get("heldout_delta_vs_base_ce"),
            sparse.get("heldout_delta_vs_base_ce"),
        )
    return metrics


def _selected_next_step(status: str, deployable_positive: bool) -> str:
    if status != "pass":
        return "repair source artifacts or provenance before interpreting support-discovery evidence"
    if deployable_positive:
        return "repeat the positive deployable support-discovery gate locally before considering RunPod validation"
    return (
        "train a local deployable support head with oracle-support labels, shuffled-causal-feature "
        "and token/position nulls, plus same-student fixed-support forcing; do not run RunPod yet"
    )


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
            "GPT-5.5-Pro review requested a major or notify-Ben shift. This gate accepts it: "
            "sparse-support identity stays retired, support discovery remains secondary, and Ben "
            "should be notified before treating support discovery as a primary claim."
        )
    return "No major strategy-review direction shift recorded for this gate."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_reports.csv", summary["source_reports"])
    _write_csv(out_dir / "support_discovery_gate.csv", summary["support_discovery_gate"])
    _write_csv(out_dir / "null_controls.csv", summary["null_controls"])
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
        "# ACSR Support-Discovery Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Oracle CE headroom vs learned sparse: `{metrics.get('sparse_oracle_minus_sparse_default_heldout_ce_delta', '')}`",
        f"- Oracle MSE margin vs target norm: `{metrics.get('oracle_support_mse_margin_vs_target_norm', '')}`",
        "",
        summary["direction_shift"],
        "",
        "This command is local-only. It keeps RunPod deferred because current evidence lacks "
        "a learned deployable support head, shuffled-causal-feature support-head null, and "
        "same-student fixed-support forcing.",
    ]
    if summary["claim_blockers"]:
        lines.extend(["", "## Claim Blockers"])
        for blocker in summary["claim_blockers"]:
            lines.append(f"- `{blocker['criterion']}`: {blocker['failure_reason']}")
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate_passed(rows: list[dict[str, Any]], component: str) -> bool:
    row = _gate_row(rows, component)
    return bool(row.get("passed"))


def _gate_observed(rows: list[dict[str, Any]], component: str) -> Any:
    return _gate_row(rows, component).get("observed", "")


def _gate_row(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    for row in rows:
        if row.get("component") == component:
            return row
    return {}


def _null_present(rows: list[dict[str, Any]], control: str) -> bool:
    for row in rows:
        if row.get("control") == control:
            return bool(row.get("present"))
    return False


def _arm(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == name:
            return row
    return {}


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
    parser.add_argument("--synthesis-dir", type=Path, default=DEFAULT_SYNTHESIS_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-oracle-ce-headroom", type=float, default=0.01)
    parser.add_argument("--min-oracle-distill-mse-margin", type=float, default=0.001)
    args = parser.parse_args()
    summary = run_acsr_support_discovery_gate(
        synthesis_dir=args.synthesis_dir,
        benchmark_dir=args.benchmark_dir,
        design_dir=args.design_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
        min_oracle_ce_headroom=args.min_oracle_ce_headroom,
        min_oracle_distill_mse_margin=args.min_oracle_distill_mse_margin,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
