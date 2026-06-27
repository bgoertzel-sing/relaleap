"""Design a bounded margin-aware cross-value transfer objective for ACSR."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SYNTHESIS = Path("results/reports/acsr_dual_student_cross_forcing_synthesis/summary.json")
DEFAULT_VALUE_SYNTHESIS = Path(
    "results/reports/acsr_dual_student_cross_forcing_synthesis/"
    "value_student_support_synthesis.csv"
)
DEFAULT_STRATIFIED_SYNTHESIS = Path(
    "results/reports/acsr_dual_student_cross_forcing_synthesis/"
    "stratified_transfer_synthesis.csv"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_margin_aware_transfer_objective_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "evidence_metrics.csv",
    "objective_terms.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_margin_aware_transfer_objective_design(
    *,
    synthesis: Path = DEFAULT_SYNTHESIS,
    value_synthesis: Path = DEFAULT_VALUE_SYNTHESIS,
    stratified_synthesis: Path = DEFAULT_STRATIFIED_SYNTHESIS,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Convert completed cross-forcing evidence into a fail-closed objective spec."""

    start = time.time()
    synthesis_summary = _read_json_object(synthesis)
    value_rows = _read_csv_rows(value_synthesis)
    stratified_rows = _read_csv_rows(stratified_synthesis)
    review = _strategy_review(strategy_review)
    evidence_rows = _evidence_rows(
        synthesis_summary=synthesis_summary,
        synthesis=synthesis,
        value_synthesis=value_synthesis,
        value_rows=value_rows,
        stratified_synthesis=stratified_synthesis,
        stratified_rows=stratified_rows,
        review=review,
    )
    objective_rows = _objective_rows(synthesis_summary, stratified_rows)
    gate_rows = _gate_rows(synthesis_summary, value_rows, stratified_rows, review)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "acsr_margin_aware_transfer_objective_design_recorded"
            if status == "pass"
            else "acsr_margin_aware_transfer_objective_design_failed_closed"
        ),
        "claim_status": "objective_design_only_not_promoted",
        "selected_next_step": (
            "implement a local low-step transfer-objective probe with these exact "
            "gate weights and compare against the current direct causal MLP router"
            if status == "pass"
            else "repair dual-student synthesis evidence before designing a transfer objective"
        ),
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "evidence_metrics": evidence_rows,
        "objective_terms": objective_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_boundaries": {
            "supported": [
                "a bounded candidate objective can be specified from local cross-value transfer evidence",
                "the objective should focus on high-regret and support-disagreement contexts",
                "low-margin partner rows should be downweighted because the current evidence is neutral there",
            ],
            "not_supported": [
                "default ACSR promotion",
                "ACSR-as-anticipation",
                "GPU repeat before a local executable objective probe",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence_rows(
    *,
    synthesis_summary: dict[str, Any],
    synthesis: Path,
    value_synthesis: Path,
    value_rows: list[dict[str, str]],
    stratified_synthesis: Path,
    stratified_rows: list[dict[str, str]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = synthesis_summary.get("aggregate_metrics", {})
    return [
        {
            "source": "dual_student_synthesis_summary",
            "path": str(synthesis),
            "present": bool(synthesis_summary),
            "status": synthesis_summary.get("status", "missing"),
            "claim_status": synthesis_summary.get("claim_status", ""),
            "metric": "mean_partner_delta_vs_token_position_null",
            "value": metrics.get("mean_partner_delta_vs_token_position_null", ""),
        },
        {
            "source": "value_student_support_synthesis",
            "path": str(value_synthesis),
            "present": value_synthesis.is_file(),
            "status": "present" if value_rows else "missing",
            "claim_status": "all_token_value_paths",
            "metric": "row_count",
            "value": len(value_rows),
        },
        {
            "source": "stratified_transfer_synthesis",
            "path": str(stratified_synthesis),
            "present": stratified_synthesis.is_file(),
            "status": "present" if stratified_rows else "missing",
            "claim_status": "regret_disagreement_margin_strata",
            "metric": "row_count",
            "value": len(stratified_rows),
        },
        {
            "source": "strategy_review",
            "path": review["path"],
            "present": review["present"],
            "status": "read" if review["present"] else "missing",
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']};"
                f"notify_ben={review['notify_ben']}"
            ),
            "metric": "recommended_next_action",
            "value": review["recommended_next_action"],
        },
    ]


def _objective_rows(
    synthesis_summary: dict[str, Any],
    stratified_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metrics = synthesis_summary.get("aggregate_metrics", {})
    high_regret_delta = _number(metrics.get("mean_high_regret_partner_delta_vs_token_position_null"))
    disagreement_delta = _number(metrics.get("mean_disagreement_partner_delta_vs_token_position_null"))
    low_margin_delta = _number(metrics.get("mean_low_margin_partner_delta_vs_token_position_null"))
    high_regret_weight = 1.0 if high_regret_delta is not None and high_regret_delta < 0.0 else 0.0
    disagreement_weight = (
        0.5 if disagreement_delta is not None and disagreement_delta < 0.0 else 0.0
    )
    low_margin_weight = 0.0 if low_margin_delta is None or low_margin_delta >= -0.01 else 0.25
    available_strata = sorted(
        {
            f"{row.get('stratum_type')}:{row.get('stratum_value')}"
            for row in stratified_rows
            if row.get("status", "available") == "available"
        }
    )
    return [
        {
            "term": "cross_value_partner_support_ce",
            "role": "primary auxiliary loss",
            "weight": 1.0,
            "token_filter": "all_tokens",
            "objective": (
                "minimize CE when the current router's selected support is evaluated "
                "through the independently trained partner value bank"
            ),
            "evidence_basis": "partner support beats token-position, shuffled, and random nulls in all value paths",
            "implementation_guardrail": "report own-value CE separately; do not promote if own-value CE worsens beyond guardrail",
        },
        {
            "term": "high_regret_cross_value_focus",
            "role": "stratified upweight",
            "weight": high_regret_weight,
            "token_filter": "top_quartile_token_position_null_regret",
            "objective": "upweight tokens where token-position null has high oracle regret",
            "evidence_basis": f"mean high-regret partner delta vs token-position null = {high_regret_delta}",
            "implementation_guardrail": "disable when high-regret transfer is not better than token-position null",
        },
        {
            "term": "support_disagreement_focus",
            "role": "stratified upweight",
            "weight": disagreement_weight,
            "token_filter": "partner_vs_own OR partner_vs_token_position_null",
            "objective": "upweight contexts where support identity differs from own or token-position support",
            "evidence_basis": f"mean disagreement partner delta vs token-position null = {disagreement_delta}",
            "implementation_guardrail": "keep as auxiliary only because support identity alone is not causal evidence",
        },
        {
            "term": "low_margin_suppression",
            "role": "margin gate",
            "weight": low_margin_weight,
            "token_filter": "low_margin",
            "objective": "suppress transfer supervision on low-margin top-k choices unless they show real gain",
            "evidence_basis": f"mean low-margin partner delta vs token-position null = {low_margin_delta}",
            "implementation_guardrail": "default weight stays zero under neutral or worse low-margin evidence",
        },
        {
            "term": "residual_norm_normalized_report",
            "role": "required metric not optimized directly",
            "weight": 0.0,
            "token_filter": "all objective rows",
            "objective": "record CE delta per residual-update L2 so larger residual norms cannot explain the result",
            "evidence_basis": (
                "current synthesis includes mean_partner_delta_vs_token_position_null_per_residual_l2"
            ),
            "implementation_guardrail": "fail closed if residual-norm-normalized metrics are missing",
        },
    ]


def _gate_rows(
    synthesis_summary: dict[str, Any],
    value_rows: list[dict[str, str]],
    stratified_rows: list[dict[str, str]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = synthesis_summary.get("aggregate_metrics", {})
    required_strata = {
        ("oracle_regret", "top_quartile_token_position_null_regret"),
        ("support_disagreement", "partner_vs_own"),
        ("support_disagreement", "partner_vs_token_position_null"),
        ("partner_support_margin_bin", "high_margin"),
    }
    observed_strata = {
        (row.get("stratum_type"), row.get("stratum_value"))
        for row in stratified_rows
        if row.get("status", "available") == "available"
    }
    return [
        _criterion(
            "strategy_review_consumed",
            review["present"],
            "latest GPT-5.5-Pro review is read",
            review["recommended_next_action"],
            "strategy review missing",
        ),
        _criterion(
            "cross_value_transfer_supported_not_promoted",
            synthesis_summary.get("claim_status") == "cross_value_support_transfer_supported_not_promoted",
            "dual-student synthesis supports transfer but does not promote",
            synthesis_summary.get("claim_status", "missing"),
            "dual-student synthesis is absent or not in the supported-not-promoted state",
        ),
        _criterion(
            "all_null_ladder_beaten",
            metrics.get("all_partner_beats_required_nulls") is True,
            "partner support beats token-position, shuffled, and random nulls",
            metrics.get("all_partner_beats_required_nulls"),
            "partner support does not beat every required null",
        ),
        _criterion(
            "value_paths_available",
            len(value_rows) >= 4,
            "both value students are available across two source packets",
            len(value_rows),
            "too few value-student synthesis rows",
        ),
        _criterion(
            "regret_disagreement_margin_strata_available",
            required_strata.issubset(observed_strata),
            "high-regret, disagreement, and margin strata are available",
            sorted(observed_strata),
            "required transfer strata are missing",
        ),
        _criterion(
            "residual_norm_control_available",
            metrics.get("residual_norm_control_available") is True,
            "residual-norm-normalized transfer metrics are present",
            metrics.get("residual_norm_control_available"),
            "residual-norm control is missing",
        ),
        _criterion(
            "low_margin_not_promoted",
            (_number(metrics.get("mean_low_margin_partner_delta_vs_token_position_null")) or 0.0) >= -0.01,
            "low-margin transfer is neutral or too small, so low-margin rows stay downweighted",
            metrics.get("mean_low_margin_partner_delta_vs_token_position_null"),
            "low-margin transfer became strongly supportive and needs a new objective design",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": "",
            "notify_ben": "",
            "recommended_next_action": "",
            "verdict": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _direction_shift(review: dict[str, Any]) -> str:
    if not review["present"]:
        return "No strategy review was available; objective design fails closed."
    if review["strategic_change_level"] == "major" or review["notify_ben"] == "true":
        return (
            "GPT-5.5-Pro review requested a major or notify-Ben direction shift: "
            f"{review['recommended_next_action']} Ben should be notified: "
            f"{review['notify_ben']}."
        )
    return (
        "Accepted GPT-5.5-Pro local recommendation where still applicable; "
        "no major direction shift or Ben notification requested."
    )


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "evidence_metrics.csv", summary["evidence_metrics"])
    _write_csv(out_dir / "objective_terms.csv", summary["objective_terms"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Margin-Aware Transfer Objective Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        "",
        summary["direction_shift"],
        "",
        "## Objective Terms",
    ]
    for row in summary["objective_terms"]:
        lines.append(
            f"- `{row['term']}` weight `{row['weight']}` on `{row['token_filter']}`: "
            f"{row['objective']}"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            summary["selected_next_step"],
            "",
            "This is a local design artifact only; it does not select RunPod or Colab.",
        ]
    )
    if summary["failures"]:
        lines.extend(["", "## Fail-Closed Reasons"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--value-synthesis", type=Path, default=DEFAULT_VALUE_SYNTHESIS)
    parser.add_argument("--stratified-synthesis", type=Path, default=DEFAULT_STRATIFIED_SYNTHESIS)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_margin_aware_transfer_objective_design(
        synthesis=args.synthesis,
        value_synthesis=args.value_synthesis,
        stratified_synthesis=args.stratified_synthesis,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
