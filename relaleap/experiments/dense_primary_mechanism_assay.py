"""Select the dense/MLP primary mechanism assay after sparse ACSR demotion."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GATE_DIR = Path("results/reports/acsr_sparse_dense_mechanism_gate")
DEFAULT_STRATIFIED_DECISION_DIR = Path("results/reports/acsr_mechanism_stratified_decision")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_OUT_DIR = Path("results/reports/dense_primary_mechanism_assay")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "candidate_scorecard.csv",
    "per_token_strata.csv",
    "decision_criteria.csv",
    "notes.md",
)

CANDIDATE_ARMS = (
    "dense_rank16_best_norm",
    "dense_rank24_best_norm",
    "parameter_matched_causal_mlp_control",
)

LOWER_IS_BETTER = (
    "ce_loss",
    "anchor_kl_or_logit_mse",
    "functional_churn",
    "retention_or_forgetting",
)
HIGHER_IS_BETTER = ("intervention_fingerprint_purity",)


def run_dense_primary_mechanism_assay(
    *,
    gate_dir: Path = DEFAULT_GATE_DIR,
    stratified_decision_dir: Path = DEFAULT_STRATIFIED_DECISION_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_per_token_rows_per_arm: int = 16,
    min_split_count_per_arm: int = 2,
) -> dict[str, Any]:
    """Rank local dense/MLP controls and name the primary non-sparse assay."""

    start = time.time()
    gate_summary = _read_json(gate_dir / "summary.json")
    stratified_summary = _read_json(stratified_decision_dir / "summary.json")
    mechanism_rows = _read_csv(gate_dir / "mechanism_metrics.csv")
    per_token_rows = _read_csv(dense_observables_dir / "per_token_observables.csv")

    scorecard = _candidate_scorecard(mechanism_rows)
    strata_rows = _per_token_strata(per_token_rows)
    coverage = _coverage(per_token_rows)
    primary = _select_primary(scorecard)
    criteria = _criteria(
        gate_summary=gate_summary,
        stratified_summary=stratified_summary,
        scorecard=scorecard,
        coverage=coverage,
        primary=primary,
        min_per_token_rows_per_arm=min_per_token_rows_per_arm,
        min_split_count_per_arm=min_split_count_per_arm,
    )
    failures = [row for row in criteria if not row["passed"]]

    if failures:
        status = "fail"
        decision = "dense_primary_mechanism_assay_failed_closed"
        claim_status = "dense_primary_mechanism_not_yet_selectable"
        selected_next_step = "repair missing local dense/control per-token observables before GPU validation"
        primary_arm = ""
    else:
        status = "pass"
        primary_arm = primary.get("arm", "")
        decision = "dense_primary_mechanism_assay_selected"
        claim_status = "dense_or_mlp_control_selected_as_primary_mechanism_assay"
        selected_next_step = (
            f"use `{primary_arm}` as the primary local residual-mechanism assay and keep sparse "
            "ACSR columns as diagnostic/intervention tooling"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "primary_arm": primary_arm,
        "primary_family": primary.get("family", "") if primary_arm else "",
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "sparse_columns_role": "diagnostic_intervention_tooling",
        "source_dirs": {
            "gate": str(gate_dir),
            "stratified_decision": str(stratified_decision_dir),
            "dense_observables": str(dense_observables_dir),
        },
        "gate_decision": gate_summary.get("decision"),
        "stratified_decision": stratified_summary.get("decision"),
        "candidate_scorecard": scorecard,
        "per_token_coverage": coverage,
        "criteria": criteria,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, scorecard, strata_rows, criteria)
    return summary


def _candidate_scorecard(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_arm = {row.get("arm", ""): row for row in rows}
    candidates: list[dict[str, Any]] = []
    for arm in CANDIDATE_ARMS:
        source = by_arm.get(arm, {})
        candidates.append(
            {
                "arm": arm,
                "family": source.get("family", ""),
                "ce_loss": _float_or_blank(source.get("ce_loss")),
                "residual_l2": _float_or_blank(source.get("residual_l2")),
                "active_rank_or_topk": _float_or_blank(source.get("active_rank_or_topk")),
                "active_params": _float_or_blank(source.get("active_params")),
                "anchor_kl_or_logit_mse": _float_or_blank(source.get("anchor_kl_or_logit_mse")),
                "functional_churn": _float_or_blank(source.get("functional_churn")),
                "retention_or_forgetting": _float_or_blank(source.get("retention_or_forgetting")),
                "intervention_fingerprint_purity": _float_or_blank(
                    source.get("intervention_fingerprint_purity")
                ),
                "mechanism_fields_present": _bool(source.get("mechanism_fields_present")),
                "missing_mechanism_fields": source.get("missing_mechanism_fields", ""),
            }
        )
    _add_rank_scores(candidates)
    return candidates


def _add_rank_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        score = 0
        comparisons = 0
        for other in rows:
            if other is row:
                continue
            for field in LOWER_IS_BETTER:
                value = _float(row.get(field))
                other_value = _float(other.get(field))
                if value is None or other_value is None:
                    continue
                comparisons += 1
                if value < other_value:
                    score += 1
            for field in HIGHER_IS_BETTER:
                value = _float(row.get(field))
                other_value = _float(other.get(field))
                if value is None or other_value is None:
                    continue
                comparisons += 1
                if value > other_value:
                    score += 1
        row["pairwise_metric_wins"] = score
        row["pairwise_metric_comparisons"] = comparisons


def _per_token_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in CANDIDATE_ARMS:
            continue
        position = int(_float(row.get("position_index")) or 0)
        grouped[(arm, "split", row.get("split", ""))].append(row)
        grouped[(arm, "position_parity", "even" if position % 2 == 0 else "odd")].append(row)
        grouped[(arm, "base_ce_bin", _bin(_float(row.get("base_ce_loss"))))].append(row)
        grouped[(arm, "residual_l2_bin", _bin(_float(row.get("residual_update_l2"))))].append(row)

    strata: list[dict[str, Any]] = []
    for (arm, stratum_type, stratum), items in sorted(grouped.items()):
        strata.append(
            {
                "arm": arm,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "row_count": len(items),
                "ce_loss": _mean(items, "ce_loss"),
                "delta_vs_base_ce": _mean(items, "delta_vs_base_ce"),
                "residual_update_l2": _mean(items, "residual_update_l2"),
                "logit_mse_vs_base": _mean(items, "logit_mse_vs_base"),
                "prediction_changed_vs_base": _mean_bool(items, "prediction_changed_vs_base"),
            }
        )
    return strata


def _coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for arm in CANDIDATE_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        coverage[arm] = {
            "per_token_rows": len(arm_rows),
            "splits": sorted({row.get("split", "") for row in arm_rows if row.get("split", "")}),
            "has_anchor": any(row.get("split") == "anchor" for row in arm_rows),
            "has_heldout": any(row.get("split") == "heldout" for row in arm_rows),
        }
    return coverage


def _select_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row.get("mechanism_fields_present")]
    if not present:
        return {}
    return sorted(
        present,
        key=lambda row: (
            -int(row.get("pairwise_metric_wins") or 0),
            _float(row.get("ce_loss")) if _float(row.get("ce_loss")) is not None else float("inf"),
            _float(row.get("active_params")) if _float(row.get("active_params")) is not None else float("inf"),
        ),
    )[0]


def _criteria(
    *,
    gate_summary: dict[str, Any],
    stratified_summary: dict[str, Any],
    scorecard: list[dict[str, Any]],
    coverage: dict[str, Any],
    primary: dict[str, Any],
    min_per_token_rows_per_arm: int,
    min_split_count_per_arm: int,
) -> list[dict[str, Any]]:
    complete_candidates = [
        row for row in scorecard if row.get("mechanism_fields_present") and not row.get("missing_mechanism_fields")
    ]
    coverage_ok = {
        arm: (
            data["per_token_rows"] >= min_per_token_rows_per_arm
            and len(data["splits"]) >= min_split_count_per_arm
            and data["has_anchor"]
            and data["has_heldout"]
        )
        for arm, data in coverage.items()
    }
    return [
        _criterion(
            "sparse_gate_blocked_or_demoted",
            gate_summary.get("decision") == "acsr_sparse_dense_mechanism_gate_blocked"
            and stratified_summary.get("decision") == "demote_acsr_sparse_columns_to_diagnostics",
            "sparse ACSR must already be blocked and demoted before selecting a dense primary assay",
            {
                "gate_decision": gate_summary.get("decision"),
                "stratified_decision": stratified_summary.get("decision"),
            },
            "sparse ACSR has not been explicitly blocked and demoted",
        ),
        _criterion(
            "all_candidate_mechanism_fields_present",
            len(complete_candidates) == len(CANDIDATE_ARMS),
            "every dense/MLP candidate has aggregate mechanism fields",
            {
                row["arm"]: {
                    "mechanism_fields_present": row["mechanism_fields_present"],
                    "missing": row["missing_mechanism_fields"],
                }
                for row in scorecard
            },
            "one or more dense/MLP candidates lacks aggregate mechanism fields",
        ),
        _criterion(
            "per_token_coverage_available",
            all(coverage_ok.values()),
            f"each candidate has >= {min_per_token_rows_per_arm} per-token rows and anchor/heldout splits",
            coverage,
            "one or more dense/MLP candidates lacks per-token anchor/heldout coverage",
        ),
        _criterion(
            "primary_candidate_selected",
            bool(primary.get("arm")),
            "scorecard selects one primary dense/MLP candidate",
            primary.get("arm", ""),
            "no complete dense/MLP primary candidate could be selected",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: Any,
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    scorecard: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "candidate_scorecard.csv", scorecard)
    _write_csv(out_dir / "per_token_strata.csv", strata_rows)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# Dense Primary Mechanism Assay",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Primary arm: `{summary['primary_arm']}`",
        f"- Sparse columns role: `{summary['sparse_columns_role']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This local report is the handoff after sparse ACSR demotion. It selects a dense rank "
        "or parameter-matched causal-MLP residual arm as the primary mechanism assay only when "
        "aggregate mechanism fields and per-token anchor/heldout strata are present.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Blockers"])
        lines.extend(
            f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"]
        )
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 2.5:
        return "low"
    if value < 4.0:
        return "mid"
    return "high"


def _mean(rows: list[dict[str, str]], field: str) -> Any:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(values) / len(values)


def _mean_bool(rows: list[dict[str, str]], field: str) -> Any:
    values = [_bool_or_none(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_blank(value: Any) -> Any:
    parsed = _float(value)
    return "" if parsed is None else parsed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument("--stratified-decision-dir", type=Path, default=DEFAULT_STRATIFIED_DECISION_DIR)
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-per-token-rows-per-arm", type=int, default=16)
    parser.add_argument("--min-split-count-per-arm", type=int, default=2)
    args = parser.parse_args()
    summary = run_dense_primary_mechanism_assay(
        gate_dir=args.gate_dir,
        stratified_decision_dir=args.stratified_decision_dir,
        dense_observables_dir=args.dense_observables_dir,
        out_dir=args.out,
        min_per_token_rows_per_arm=args.min_per_token_rows_per_arm,
        min_split_count_per_arm=args.min_split_count_per_arm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
