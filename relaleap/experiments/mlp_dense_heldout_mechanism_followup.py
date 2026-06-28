"""Heldout mechanism follow-up for the selected dense/MLP primary assay."""

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


DEFAULT_PRIMARY_ASSAY_DIR = Path("results/reports/dense_primary_mechanism_assay")
DEFAULT_DENSE_OBSERVABLES_DIR = Path("results/reports/acsr_dense_mechanism_observables")
DEFAULT_OUT_DIR = Path("results/reports/mlp_dense_heldout_mechanism_followup")

PRIMARY_ARM = "parameter_matched_causal_mlp_control"
DENSE_ARMS = ("dense_rank16_best_norm", "dense_rank24_best_norm")
CANDIDATE_ARMS = DENSE_ARMS + (PRIMARY_ARM,)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "heldout_strata.csv",
    "mechanism_comparison.csv",
    "decision_criteria.csv",
    "notes.md",
)

LOWER_IS_BETTER = (
    "heldout_ce_loss",
    "heldout_logit_mse_vs_base",
    "heldout_prediction_changed_vs_base",
    "functional_churn",
    "retention_or_forgetting",
)
HIGHER_IS_BETTER = ("intervention_fingerprint_purity",)


def run_mlp_dense_heldout_mechanism_followup(
    *,
    primary_assay_dir: Path = DEFAULT_PRIMARY_ASSAY_DIR,
    dense_observables_dir: Path = DEFAULT_DENSE_OBSERVABLES_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_heldout_rows_per_arm: int = 16,
) -> dict[str, Any]:
    """Compare the selected causal MLP control with dense rank-16/24 on heldout mechanism fields."""

    start = time.time()
    primary_summary = _read_json(primary_assay_dir / "summary.json")
    scorecard_rows = _read_csv(primary_assay_dir / "candidate_scorecard.csv")
    per_token_rows = _read_csv(dense_observables_dir / "per_token_observables.csv")

    comparisons = _mechanism_comparison(scorecard_rows, per_token_rows)
    strata = _heldout_strata(per_token_rows)
    coverage = _coverage(per_token_rows)
    primary_row = next((row for row in comparisons if row["arm"] == PRIMARY_ARM), {})
    criteria = _criteria(
        primary_summary=primary_summary,
        comparisons=comparisons,
        coverage=coverage,
        primary_row=primary_row,
        min_heldout_rows_per_arm=min_heldout_rows_per_arm,
    )
    failures = [row for row in criteria if not row["passed"]]

    if failures:
        status = "fail"
        decision = "mlp_dense_heldout_mechanism_followup_failed_closed"
        claim_status = "heldout_mechanism_followup_not_decisive"
        selected_next_step = "repair missing heldout dense/MLP observables before GPU validation"
    else:
        status = "pass"
        primary_wins = int(primary_row.get("pairwise_metric_wins") or 0)
        primary_comparisons = int(primary_row.get("pairwise_metric_comparisons") or 0)
        churn_tradeoff = any(
            _float(primary_row.get("functional_churn")) is not None
            and _float(other.get("functional_churn")) is not None
            and _float(primary_row.get("functional_churn")) > _float(other.get("functional_churn"))
            for other in comparisons
            if other.get("arm") in DENSE_ARMS
        )
        if churn_tradeoff:
            decision = "mlp_primary_with_functional_churn_tradeoff"
            claim_status = "mlp_control_leads_ce_retention_fingerprint_but_dense_has_lower_churn"
            selected_next_step = (
                "add one local churn-targeted intervention fingerprint check for the MLP control "
                "before any GPU validation"
            )
        elif primary_wins >= max(1, primary_comparisons // 2):
            decision = "mlp_primary_heldout_mechanism_supported"
            claim_status = "mlp_control_remains_primary_on_heldout_mechanism_fields"
            selected_next_step = "package the MLP dense-primary assay for the next GPU repeat only if Ben requests it"
        else:
            decision = "mlp_primary_heldout_mechanism_blocked_by_dense"
            claim_status = "dense_rank_control_beats_mlp_on_heldout_mechanism_fields"
            selected_next_step = "switch the primary dense mechanism assay to the winning dense-rank control locally"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "primary_arm": PRIMARY_ARM if status == "pass" else "",
        "candidate_arms": list(CANDIDATE_ARMS),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "source_dirs": {
            "primary_assay": str(primary_assay_dir),
            "dense_observables": str(dense_observables_dir),
        },
        "coverage": coverage,
        "mechanism_comparison": comparisons,
        "criteria": criteria,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, strata, comparisons, criteria)
    return summary


def _mechanism_comparison(
    scorecard_rows: list[dict[str, str]],
    per_token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    scorecard = {row.get("arm", ""): row for row in scorecard_rows}
    comparisons: list[dict[str, Any]] = []
    for arm in CANDIDATE_ARMS:
        heldout = [row for row in per_token_rows if row.get("arm") == arm and row.get("split") == "heldout"]
        aggregate = scorecard.get(arm, {})
        comparisons.append(
            {
                "arm": arm,
                "family": aggregate.get("family", ""),
                "heldout_rows": len(heldout),
                "heldout_ce_loss": _mean(heldout, "ce_loss"),
                "heldout_delta_vs_base_ce": _mean(heldout, "delta_vs_base_ce"),
                "heldout_residual_update_l2": _mean(heldout, "residual_update_l2"),
                "heldout_logit_mse_vs_base": _mean(heldout, "logit_mse_vs_base"),
                "heldout_prediction_changed_vs_base": _mean_bool(heldout, "prediction_changed_vs_base"),
                "functional_churn": _float_or_blank(aggregate.get("functional_churn")),
                "retention_or_forgetting": _float_or_blank(aggregate.get("retention_or_forgetting")),
                "intervention_fingerprint_purity": _float_or_blank(
                    aggregate.get("intervention_fingerprint_purity")
                ),
                "active_params": _float_or_blank(aggregate.get("active_params")),
            }
        )
    _add_pairwise_wins(comparisons)
    return comparisons


def _add_pairwise_wins(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        wins = 0
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
                    wins += 1
            for field in HIGHER_IS_BETTER:
                value = _float(row.get(field))
                other_value = _float(other.get(field))
                if value is None or other_value is None:
                    continue
                comparisons += 1
                if value > other_value:
                    wins += 1
        row["pairwise_metric_wins"] = wins
        row["pairwise_metric_comparisons"] = comparisons


def _heldout_strata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in CANDIDATE_ARMS or row.get("split") != "heldout":
            continue
        position = int(_float(row.get("position_index")) or 0)
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
        heldout = [row for row in rows if row.get("arm") == arm and row.get("split") == "heldout"]
        coverage[arm] = {
            "heldout_rows": len(heldout),
            "has_heldout": bool(heldout),
            "positions": len({row.get("position_index", "") for row in heldout}),
        }
    return coverage


def _criteria(
    *,
    primary_summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    coverage: dict[str, Any],
    primary_row: dict[str, Any],
    min_heldout_rows_per_arm: int,
) -> list[dict[str, Any]]:
    arms_present = {row.get("arm") for row in comparisons}
    mechanism_fields_present = {
        row["arm"]: all(row.get(field) != "" for field in ("functional_churn", "retention_or_forgetting", "intervention_fingerprint_purity"))
        for row in comparisons
    }
    heldout_ok = {
        arm: data["heldout_rows"] >= min_heldout_rows_per_arm and data["has_heldout"]
        for arm, data in coverage.items()
    }
    return [
        _criterion(
            "primary_assay_selected_mlp",
            primary_summary.get("decision") == "dense_primary_mechanism_assay_selected"
            and primary_summary.get("primary_arm") == PRIMARY_ARM,
            "dense primary assay must select the parameter-matched causal MLP control",
            {
                "decision": primary_summary.get("decision"),
                "primary_arm": primary_summary.get("primary_arm"),
            },
            "dense primary assay has not selected the MLP control",
        ),
        _criterion(
            "candidate_arms_present",
            all(arm in arms_present for arm in CANDIDATE_ARMS),
            "MLP, dense rank-16, and dense rank-24 candidates must be present",
            sorted(arms_present),
            "one or more candidate arms is missing",
        ),
        _criterion(
            "heldout_coverage_present",
            all(heldout_ok.values()),
            f"each candidate has >= {min_heldout_rows_per_arm} heldout per-token rows",
            coverage,
            "one or more candidates lacks heldout coverage",
        ),
        _criterion(
            "mechanism_fields_present",
            all(mechanism_fields_present.values()),
            "each candidate has functional churn, retention, and fingerprint aggregate fields",
            mechanism_fields_present,
            "one or more candidates lacks aggregate mechanism fields",
        ),
        _criterion(
            "primary_compared_pairwise",
            bool(primary_row.get("pairwise_metric_comparisons")),
            "primary MLP has pairwise heldout mechanism comparisons",
            primary_row.get("pairwise_metric_comparisons", 0),
            "primary MLP has no pairwise comparison metrics",
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
    strata: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "heldout_strata.csv", strata)
    _write_csv(out_dir / "mechanism_comparison.csv", comparisons)
    _write_csv(out_dir / "decision_criteria.csv", criteria)
    lines = [
        "# MLP Dense Heldout Mechanism Follow-up",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Primary arm: `{summary['primary_arm']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This local report keeps sparse ACSR columns demoted to diagnostics and compares the selected "
        "parameter-matched causal MLP control against dense rank-16/24 on heldout CE, churn proxies, "
        "retention, and intervention-fingerprint fields.",
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


def _bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.5:
        return "low"
    if value < 1.5:
        return "mid"
    return "high"


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
    parser.add_argument("--primary-assay-dir", type=Path, default=DEFAULT_PRIMARY_ASSAY_DIR)
    parser.add_argument("--dense-observables-dir", type=Path, default=DEFAULT_DENSE_OBSERVABLES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-heldout-rows-per-arm", type=int, default=16)
    args = parser.parse_args()
    summary = run_mlp_dense_heldout_mechanism_followup(
        primary_assay_dir=args.primary_assay_dir,
        dense_observables_dir=args.dense_observables_dir,
        out_dir=args.out,
        min_heldout_rows_per_arm=args.min_heldout_rows_per_arm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
