"""Local mechanism-stratified ACSR sparse-vs-control decision report."""

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
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_OUT_DIR = Path("results/reports/acsr_mechanism_stratified_decision")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "strata_metrics.csv",
    "decision_criteria.csv",
    "notes.md",
)

SPARSE_ARM = "acsr_mlp_predicted_future"
CONTROL_ARMS = (
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


def run_acsr_mechanism_stratified_decision(
    *,
    gate_dir: Path = DEFAULT_GATE_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_strata_per_control: int = 4,
) -> dict[str, Any]:
    """Stratify available mechanism rows and decide whether sparse ACSR continues."""

    start = time.time()
    gate_summary = _read_json(gate_dir / "summary.json")
    mechanism_rows = _read_csv(gate_dir / "mechanism_metrics.csv")
    per_token_rows = _read_csv(dense_observables_dir / "per_token_observables.csv")

    aggregate_rows = _aggregate_strata(mechanism_rows)
    per_token_strata = _per_token_strata(per_token_rows)
    strata_rows = aggregate_rows + per_token_strata
    criteria = _criteria(
        gate_summary=gate_summary,
        mechanism_rows=mechanism_rows,
        strata_rows=strata_rows,
        min_strata_per_control=min_strata_per_control,
    )
    failures = [row for row in criteria if not row["passed"]]
    demotion_reasons = [row["failure_reason"] for row in failures if row["failure_reason"]]
    if not failures:
        decision = "continue_acsr_sparse_columns_with_stratified_local_followup"
        claim_status = "sparse_acsr_has_stratified_signal_worth_local_followup"
        next_step = "extract sparse ACSR per-token observables on a held-out local packet"
        continue_sparse = True
    else:
        decision = "demote_acsr_sparse_columns_to_diagnostics"
        claim_status = "sparse_acsr_not_supported_as_primary_mechanism"
        next_step = "treat ACSR sparse columns as diagnostic tooling and make dense/causal-MLP controls the primary mechanism assay"
        continue_sparse = False

    summary = {
        "status": "pass" if gate_summary else "fail",
        "decision": decision,
        "claim_status": claim_status,
        "continue_sparse_columns": continue_sparse,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "source_dirs": {
            "gate": str(gate_dir),
            "dense_observables": str(dense_observables_dir),
        },
        "gate_decision": gate_summary.get("decision"),
        "gate_claim_status": gate_summary.get("claim_status"),
        "aggregate_sparse_scorecard": _aggregate_scorecard(mechanism_rows),
        "strata_row_count": len(strata_rows),
        "per_token_strata_row_count": len(per_token_strata),
        "criteria": criteria,
        "failures": failures,
        "demotion_reasons": demotion_reasons,
        "selected_next_step": next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, strata_rows, criteria)
    return summary


def _aggregate_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_arm = {row.get("arm", ""): row for row in rows}
    sparse = by_arm.get(SPARSE_ARM, {})
    out: list[dict[str, Any]] = []
    for arm in (SPARSE_ARM, *CONTROL_ARMS):
        row = by_arm.get(arm, {})
        out.append(
            {
                "stratum_type": "aggregate",
                "stratum": "all_tokens",
                "arm": arm,
                "family": row.get("family", ""),
                "row_count": 1 if row else 0,
                "ce_loss": _float_or_blank(row.get("ce_loss")),
                "anchor_kl_or_logit_mse": _float_or_blank(row.get("anchor_kl_or_logit_mse")),
                "functional_churn": _float_or_blank(row.get("functional_churn")),
                "retention_or_forgetting": _float_or_blank(row.get("retention_or_forgetting")),
                "intervention_fingerprint_purity": _float_or_blank(row.get("intervention_fingerprint_purity")),
                "sparse_comparable": bool(row) and bool(sparse),
                "note": "aggregate gate metric",
            }
        )
    return out


def _per_token_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in CONTROL_ARMS:
            continue
        split = row.get("split", "")
        position = int(_float(row.get("position_index")) or 0)
        grouped[(arm, "split", split)].append(row)
        grouped[(arm, "position_parity", "even" if position % 2 == 0 else "odd")].append(row)
        grouped[(arm, "base_ce_bin", _bin(_float(row.get("base_ce_loss"))))].append(row)
        grouped[(arm, "residual_l2_bin", _bin(_float(row.get("residual_update_l2"))))].append(row)

    out: list[dict[str, Any]] = []
    for (arm, stratum_type, stratum), items in sorted(grouped.items()):
        out.append(
            {
                "stratum_type": stratum_type,
                "stratum": stratum,
                "arm": arm,
                "family": "control_per_token",
                "row_count": len(items),
                "ce_loss": _mean(items, "ce_loss"),
                "anchor_kl_or_logit_mse": _mean(items, "logit_mse_vs_base"),
                "functional_churn": _mean_bool(items, "prediction_changed_vs_base"),
                "retention_or_forgetting": _mean(items, "delta_vs_base_ce"),
                "intervention_fingerprint_purity": "",
                "sparse_comparable": False,
                "note": "control-only per-token stratum; sparse per-token rows are not available in current ACSR packet",
            }
        )
    return out


def _criteria(
    *,
    gate_summary: dict[str, Any],
    mechanism_rows: list[dict[str, str]],
    strata_rows: list[dict[str, Any]],
    min_strata_per_control: int,
) -> list[dict[str, Any]]:
    scorecard = _aggregate_scorecard(mechanism_rows)
    control_strata_counts = {
        arm: sum(
            1
            for row in strata_rows
            if row.get("arm") == arm and row.get("stratum_type") != "aggregate"
        )
        for arm in CONTROL_ARMS
    }
    sparse_strata = [
        row
        for row in strata_rows
        if row.get("arm") == SPARSE_ARM and row.get("stratum_type") != "aggregate"
    ]
    aggregate_sparse_wins = scorecard["wins"] == scorecard["total_required_comparisons"]
    gate_blocked = gate_summary.get("decision") == "acsr_sparse_dense_mechanism_gate_blocked"
    return [
        _criterion(
            "prior_sparse_dense_gate_blocks",
            not gate_blocked,
            "previous aggregate sparse-dense gate must not be blocked",
            gate_summary.get("decision"),
            "aggregate sparse-dense gate already blocks sparse as primary mechanism",
        ),
        _criterion(
            "aggregate_sparse_wins_all_required_fields",
            aggregate_sparse_wins,
            "sparse must beat dense rank16/24 and parameter-matched control on all required fields",
            scorecard,
            "sparse does not win all required aggregate mechanism comparisons",
        ),
        _criterion(
            "control_strata_available",
            all(count >= min_strata_per_control for count in control_strata_counts.values()),
            f"each control must expose at least {min_strata_per_control} mechanism strata",
            control_strata_counts,
            "one or more controls lack enough local mechanism strata",
        ),
        _criterion(
            "sparse_per_token_strata_available",
            bool(sparse_strata),
            "sparse ACSR must expose per-token strata to test mechanism-specific separation",
            len(sparse_strata),
            "current ACSR packet has no sparse per-token observables for stratified comparison",
        ),
    ]


def _aggregate_scorecard(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_arm = {row.get("arm", ""): row for row in rows}
    sparse = by_arm.get(SPARSE_ARM, {})
    comparisons: list[dict[str, Any]] = []
    for control in CONTROL_ARMS:
        control_row = by_arm.get(control, {})
        for field in LOWER_IS_BETTER:
            sparse_value = _float(sparse.get(field))
            control_value = _float(control_row.get(field))
            won = sparse_value is not None and control_value is not None and sparse_value < control_value
            comparisons.append(
                {
                    "control": control,
                    "field": field,
                    "direction": "lower_is_better",
                    "sparse": sparse_value,
                    "control_value": control_value,
                    "sparse_wins": won,
                }
            )
        for field in HIGHER_IS_BETTER:
            sparse_value = _float(sparse.get(field))
            control_value = _float(control_row.get(field))
            won = sparse_value is not None and control_value is not None and sparse_value > control_value
            comparisons.append(
                {
                    "control": control,
                    "field": field,
                    "direction": "higher_is_better",
                    "sparse": sparse_value,
                    "control_value": control_value,
                    "sparse_wins": won,
                }
            )
    wins = sum(1 for row in comparisons if row["sparse_wins"])
    return {
        "wins": wins,
        "losses_or_ties": len(comparisons) - wins,
        "total_required_comparisons": len(comparisons),
        "comparisons": comparisons,
    }


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
    strata_rows: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "strata_metrics.csv", strata_rows)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# ACSR Mechanism-Stratified Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Continue sparse columns: `{summary['continue_sparse_columns']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This local follow-up consumes the sparse-dense mechanism gate and available per-token "
        "control observables. It fails closed when sparse ACSR lacks comparable per-token "
        "strata or loses/ties aggregate mechanism fields.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Decision Blockers"])
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
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-strata-per-control", type=int, default=4)
    args = parser.parse_args()
    summary = run_acsr_mechanism_stratified_decision(
        gate_dir=args.gate_dir,
        dense_observables_dir=args.dense_observables_dir,
        out_dir=args.out,
        min_strata_per_control=args.min_strata_per_control,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
