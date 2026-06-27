"""Design the post-columnability ACSR support-discovery follow-up."""

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
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_support_discovery_followup_design_seed2")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_reports.csv",
    "support_discovery_design.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_support_discovery_followup_design(
    *,
    synthesis_dir: Path = DEFAULT_SYNTHESIS_DIR,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a bounded local support-discovery design after retiring identity claims."""

    start = time.time()
    synthesis_summary = _read_json(synthesis_dir / "summary.json")
    benchmark_summary = _read_json(benchmark_dir / "summary.json")
    benchmark_arms = _read_csv(benchmark_dir / "arm_metrics.csv")
    benchmark_fingerprints = _read_csv(benchmark_dir / "intervention_fingerprints.csv")
    review = _strategy_review(strategy_review)
    source_rows = _source_rows(
        synthesis_dir,
        benchmark_dir,
        synthesis_summary,
        benchmark_summary,
        benchmark_arms,
        benchmark_fingerprints,
        review,
    )
    design_rows = _design_rows(synthesis_summary, benchmark_arms, benchmark_fingerprints)
    gate_rows = _gate_rows(
        synthesis_summary,
        benchmark_summary,
        benchmark_arms,
        benchmark_fingerprints,
        review,
        design_rows,
    )
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "support_discovery_followup_design_recorded_identity_claim_stays_retired"
            if status == "pass"
            else "support_discovery_followup_design_failed_closed"
        ),
        "claim_status": (
            "design_only_support_discovery_not_established_sparse_identity_retired"
            if status == "pass"
            else "support_discovery_design_not_interpretable"
        ),
        "selected_next_step": _selected_next_step(status),
        "next_command": (
            "./.venv-conda/bin/python -m relaleap.experiments.acsr_support_discovery_gate"
        ),
        "source_reports": source_rows,
        "support_discovery_design": design_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "aggregate_metrics": _aggregate_metrics(synthesis_summary, benchmark_arms),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "claim_boundaries": {
            "supported": [
                "sparse-support identity remains retired as the primary claim",
                "oracle support provides limited but real local headroom for a support-discovery follow-up",
                "the next step can be local and command-driven without RunPod replication",
            ],
            "not_supported": [
                "deployable support discovery",
                "causal sparse-support substrate claims",
                "default-router promotion",
                "GPU validation before a local gate produces a positive target",
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
    synthesis_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
    benchmark_arms: list[dict[str, Any]],
    benchmark_fingerprints: list[dict[str, Any]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "columnability_synthesis",
            "path": str(synthesis_dir / "summary.json"),
            "present": bool(synthesis_summary),
            "status": synthesis_summary.get("status", "missing"),
            "decision": synthesis_summary.get("decision", ""),
            "claim_status": synthesis_summary.get("claim_status", ""),
            "git_commit": synthesis_summary.get("git_commit", ""),
        },
        {
            "source": "common_causal_residual_benchmark",
            "path": str(benchmark_dir / "summary.json"),
            "present": bool(benchmark_summary),
            "status": benchmark_summary.get("status", "missing"),
            "decision": benchmark_summary.get("decision", ""),
            "claim_status": benchmark_summary.get("claim_status", ""),
            "git_commit": benchmark_summary.get("git_commit", ""),
        },
        {
            "source": "benchmark_arm_metrics",
            "path": str(benchmark_dir / "arm_metrics.csv"),
            "present": bool(benchmark_arms),
            "status": "present" if benchmark_arms else "missing",
            "decision": f"rows={len(benchmark_arms)}",
            "claim_status": "oracle_dense_sparse_teacher_arms",
            "git_commit": "",
        },
        {
            "source": "benchmark_intervention_fingerprints",
            "path": str(benchmark_dir / "intervention_fingerprints.csv"),
            "present": bool(benchmark_fingerprints),
            "status": "present" if benchmark_fingerprints else "missing",
            "decision": f"rows={len(benchmark_fingerprints)}",
            "claim_status": "support_overlap_and_functional_fingerprint_rows",
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


def _design_rows(
    synthesis_summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = _aggregate_metrics(synthesis_summary, arm_rows)
    sparse = _arm(arm_rows, "sparse_contextual_topk2")
    oracle = _arm(arm_rows, "sparse_oracle_support")
    oracle_distill = _arm(arm_rows, "sparse_teacher_distilled_oracle_support_topk2")
    target_distill = _arm(arm_rows, "sparse_teacher_distilled_target_norm_topk2")
    token_null = _arm(arm_rows, "sparse_teacher_distilled_token_position_null")
    return [
        {
            "component": "identity_retirement_guardrail",
            "available_now": True,
            "primary_metric": "retire_sparse_identity_primary_claim",
            "current_value": metrics.get("retire_sparse_identity_primary_claim", ""),
            "required_for_followup": "support discovery may proceed only as a secondary mechanism probe",
            "design_requirement": "every follow-up report must carry sparse_support_identity_primary_claim_retired_locally forward",
        },
        {
            "component": "oracle_support_headroom",
            "available_now": bool(oracle),
            "primary_metric": "sparse_oracle_minus_sparse_default_heldout_ce_delta",
            "current_value": _delta_number(
                oracle.get("heldout_delta_vs_base_ce"),
                sparse.get("heldout_delta_vs_base_ce"),
            ),
            "required_for_followup": "oracle support should beat learned support before router discovery is worth testing",
            "design_requirement": "train a causal deployable support head only if oracle headroom is positive on the local packet",
        },
        {
            "component": "teacher_oracle_distill_margin",
            "available_now": bool(oracle_distill and target_distill),
            "primary_metric": "oracle_support_mse_margin_vs_target_norm",
            "current_value": metrics.get("oracle_support_mse_margin_vs_target_norm", ""),
            "required_for_followup": "oracle support should improve teacher-residual representation quality",
            "design_requirement": "report teacher MSE, cosine, norm ratio, CE, damage, and active-compute for learned, oracle, random, and null supports",
        },
        {
            "component": "token_position_null_guardrail",
            "available_now": bool(token_null),
            "primary_metric": "target_norm_distill_beats_token_position_null",
            "current_value": _benchmark_interpretation(synthesis_summary).get(
                "target_norm_distill_beats_token_position_null",
                "",
            ),
            "required_for_followup": "deployable supports must beat token/position-only nulls, not just shuffled teacher",
            "design_requirement": "include token/position-only and shuffled-causal-feature support heads in the same run",
        },
        {
            "component": "functional_intervention_purity",
            "available_now": bool(fingerprint_rows),
            "primary_metric": "support_overlap_and_intervention_fingerprint_rows",
            "current_value": f"rows={len(fingerprint_rows)}",
            "required_for_followup": "support-policy gains must be separated from residual-value co-adaptation",
            "design_requirement": "add same-student fixed-support forcing and dual-student cross-forcing before mechanism claims",
        },
        {
            "component": "local_then_gpu_gate",
            "available_now": True,
            "primary_metric": "runpod_deferred_until_positive_local_gate",
            "current_value": "deferred",
            "required_for_followup": "GPU validation requires a local deployable support-discovery target",
            "design_requirement": "do not run RunPod until the local gate beats token/position and shuffled-feature nulls under clean provenance",
        },
    ]


def _gate_rows(
    synthesis_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
    fingerprint_rows: list[dict[str, Any]],
    review: dict[str, Any],
    design_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = _aggregate_metrics(synthesis_summary, arm_rows)
    observed_arms = {str(row.get("arm")) for row in arm_rows}
    required_arms = {
        "sparse_contextual_topk2",
        "sparse_oracle_support",
        "sparse_teacher_distilled_target_norm_topk2",
        "sparse_teacher_distilled_oracle_support_topk2",
        "sparse_teacher_distilled_token_position_null",
    }
    return [
        _criterion(
            "strategy_review_consumed",
            review.get("status") == "read",
            "latest GPT-5.5-Pro strategy review is read",
            review.get("status", ""),
            "strategy review was not available/read",
        ),
        _criterion(
            "identity_claim_retired_before_followup",
            synthesis_summary.get("claim_status") == "sparse_support_identity_primary_claim_retired_locally"
            and bool(metrics.get("retire_sparse_identity_primary_claim")),
            "support-discovery follow-up must not revive sparse-support identity",
            {
                "claim_status": synthesis_summary.get("claim_status", ""),
                "retire_sparse_identity_primary_claim": metrics.get("retire_sparse_identity_primary_claim"),
            },
            "sparse-support identity has not been cleanly retired",
        ),
        _criterion(
            "benchmark_provenance_clean",
            benchmark_summary.get("git_dirty") is False and not benchmark_summary.get("git_diff_hash"),
            "source benchmark must record clean git provenance",
            {
                "git_dirty": benchmark_summary.get("git_dirty", ""),
                "git_diff_hash": benchmark_summary.get("git_diff_hash", ""),
            },
            "source benchmark provenance is dirty or unknown",
        ),
        _criterion(
            "required_support_discovery_arms_present",
            required_arms.issubset(observed_arms),
            "benchmark must include default, oracle, teacher-oracle, target-norm, and token-position arms",
            {"missing": sorted(required_arms - observed_arms), "observed_count": len(observed_arms)},
            "required support-discovery source arms are missing",
        ),
        _criterion(
            "oracle_headroom_is_secondary_not_identity_rescue",
            _positive(metrics.get("oracle_support_mse_margin_vs_target_norm"))
            and bool(metrics.get("secondary_support_discovery_followup_warranted"))
            and _number(metrics.get("teacher_distill_gap_vs_default_sparse_ce_delta")) is not None
            and _number(metrics.get("teacher_distill_gap_vs_default_sparse_ce_delta")) > 0.0,
            "oracle support headroom may justify follow-up only while teacher distill still fails the identity rescue",
            {
                "oracle_support_mse_margin_vs_target_norm": metrics.get("oracle_support_mse_margin_vs_target_norm"),
                "teacher_distill_gap_vs_default_sparse_ce_delta": metrics.get("teacher_distill_gap_vs_default_sparse_ce_delta"),
                "secondary_support_discovery_followup_warranted": metrics.get("secondary_support_discovery_followup_warranted"),
            },
            "oracle support does not create an interpretable secondary follow-up",
        ),
        _criterion(
            "intervention_fingerprints_available",
            bool(fingerprint_rows),
            "support overlap/intervention fingerprint rows are available for follow-up design",
            f"rows={len(fingerprint_rows)}",
            "intervention fingerprint rows are missing",
        ),
        _criterion(
            "design_components_available",
            all(bool(row.get("available_now")) for row in design_rows),
            "all design components have a current source basis",
            [row.get("component") for row in design_rows if not row.get("available_now")],
            "one or more design components lack a source basis",
        ),
    ]


def _aggregate_metrics(
    synthesis_summary: dict[str, Any],
    arm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _as_dict(synthesis_summary.get("aggregate_metrics")).copy()
    sparse = _arm(arm_rows, "sparse_contextual_topk2")
    oracle = _arm(arm_rows, "sparse_oracle_support")
    if sparse and oracle:
        metrics["sparse_oracle_minus_sparse_default_heldout_ce_delta"] = _delta_number(
            oracle.get("heldout_delta_vs_base_ce"),
            sparse.get("heldout_delta_vs_base_ce"),
        )
    return metrics


def _selected_next_step(status: str) -> str:
    if status != "pass":
        return "repair the columnability synthesis or benchmark provenance before designing support-discovery follow-up"
    return (
        "implement a local opt-in support-discovery gate with oracle-support labels, "
        "token/position and shuffled-feature nulls, same-student fixed-support forcing, "
        "and no RunPod until the local gate is positive"
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
            "GPT-5.5-Pro review requested a major or notify-Ben shift. This design "
            "accepts it: sparse-support identity stays retired, support discovery is "
            "secondary, and Ben should be notified before treating this as a new primary claim."
        )
    return "No major strategy-review direction shift recorded for this design."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_reports.csv", summary["source_reports"])
    _write_csv(out_dir / "support_discovery_design.csv", summary["support_discovery_design"])
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
        "# ACSR Support-Discovery Follow-Up Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Oracle MSE margin vs target norm: `{metrics.get('oracle_support_mse_margin_vs_target_norm', '')}`",
        f"- Teacher-distill CE gap vs default sparse: `{metrics.get('teacher_distill_gap_vs_default_sparse_ce_delta', '')}`",
        f"- Sparse oracle CE-delta gap vs learned sparse: `{metrics.get('sparse_oracle_minus_sparse_default_heldout_ce_delta', '')}`",
        "",
        summary["direction_shift"],
        "",
        "This is a design-only artifact. It keeps the sparse-support identity claim retired "
        "and scopes the next experiment to deployable support discovery with null and "
        "same-student intervention controls.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _benchmark_interpretation(summary: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(summary.get("benchmark_interpretation"))


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
    parser.add_argument("--synthesis-dir", type=Path, default=DEFAULT_SYNTHESIS_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_support_discovery_followup_design(
        synthesis_dir=args.synthesis_dir,
        benchmark_dir=args.benchmark_dir,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
