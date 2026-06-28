"""Synthesize norm-budgeted residual pilot per-token strata."""

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


DEFAULT_PILOT_DIR = Path("results/reports/norm_budgeted_churn_regularized_residual_pilot")
DEFAULT_OUT_DIR = Path("results/reports/norm_budgeted_churn_strata_synthesis")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_status.csv",
    "arm_signal_summary.csv",
    "strata_summary.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_TOKEN_FIELDS = {
    "arm",
    "family",
    "intervention_stratum",
    "ce_loss",
    "base_ce_loss",
    "delta_vs_base_ce",
    "residual_update_l2",
    "anchor_kl_vs_base",
    "prediction_changed_vs_base",
}


def run_norm_budgeted_churn_strata_synthesis(
    *,
    pilot_dir: Path = DEFAULT_PILOT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a compact local synthesis over the norm-budgeted pilot rows."""

    start = time.time()
    pilot_summary = _read_json(pilot_dir / "summary.json")
    arm_rows = _read_csv(pilot_dir / "arm_metrics.csv")
    token_rows = _read_csv(pilot_dir / "per_token_metrics.csv")
    source_rows = _source_rows(pilot_dir, pilot_summary, arm_rows, token_rows)
    source_ok = all(row["passed"] for row in source_rows)
    strata_rows = _strata_rows(token_rows) if source_ok else []
    arm_signal_rows = _arm_signal_rows(arm_rows, strata_rows) if source_ok else []
    gate_rows = _gate_rows(source_rows, arm_signal_rows)
    failures = [row for row in source_rows + gate_rows if not row["passed"]]
    status = "pass" if source_ok else "fail"
    warrants_runpod = (
        status == "pass"
        and any(row["scientific_signal"] == "weak_local_signal_needs_repeat" for row in arm_signal_rows)
    )
    interference_signal = status == "pass" and _has_nontrivial_ce_interference_signal(arm_signal_rows)
    decision = (
        "norm_budgeted_churn_strata_synthesis_completed"
        if status == "pass"
        else "norm_budgeted_churn_strata_synthesis_failed_closed"
    )
    selected_next_step = (
        "prepare a bounded RunPod repeat only after confirming the weak local matched-strata signal is not a budget artifact"
        if warrants_runpod
        else "keep work local and add an explicit churn/anchor penalty to the scale-gated sparse norm objective before any RunPod repeat"
        if interference_signal
        else "keep work local and diagnose sparse/MLP budget underuse with stronger norm-use mechanics before any RunPod repeat"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": (
            "local_strata_signal_warrants_cautious_repeat_review"
            if warrants_runpod
            else "local_strata_signal_does_not_warrant_gpu_repeat"
        ),
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "runpod_repeat_warranted": warrants_runpod,
        "pilot_dir": str(pilot_dir),
        "out_dir": str(out_dir),
        "source_status": source_rows,
        "arm_signal_summary": arm_signal_rows,
        "strata_row_count": len(strata_rows),
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "interpretation": _interpretation(status, warrants_runpod, arm_signal_rows),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, arm_signal_rows, strata_rows, gate_rows)
    return summary


def _source_rows(
    pilot_dir: Path,
    pilot_summary: dict[str, Any],
    arm_rows: list[dict[str, str]],
    token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    token_fields = set(token_rows[0]) if token_rows else set()
    arms = {row.get("arm", "") for row in arm_rows}
    strata = {row.get("intervention_stratum", "") for row in token_rows}
    return [
        _criterion(
            "pilot_summary_passed",
            pilot_summary.get("status") == "pass",
            "source pilot summary has status pass",
            {"path": str(pilot_dir / "summary.json"), "status": pilot_summary.get("status")},
            "source pilot summary is missing or not passing",
        ),
        _criterion(
            "arm_metrics_present",
            bool(arm_rows) and "dense_rank24_norm_budgeted" in arms,
            "arm metrics include dense rank24 comparator",
            sorted(arms),
            "arm metrics missing or dense24 comparator absent",
        ),
        _criterion(
            "per_token_metrics_present",
            bool(token_rows) and REQUIRED_TOKEN_FIELDS.issubset(token_fields),
            "per-token metrics include required CE/L2/KL/churn fields",
            sorted(token_fields),
            "per-token metrics missing required fields",
        ),
        _criterion(
            "target_and_off_target_strata_present",
            {"target_heldout", "off_target_anchor"}.issubset(strata),
            "target heldout and off-target anchor strata are present",
            sorted(strata),
            "target/off-target strata missing",
        ),
    ]


def _strata_rows(token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        key = (
            row.get("arm", ""),
            row.get("intervention_stratum", ""),
            _residual_l2_bin(_float(row.get("residual_update_l2"))),
            _anchor_kl_bin(_float(row.get("anchor_kl_vs_base"))),
            _base_loss_bin(_float(row.get("base_ce_loss"))),
        )
        groups[key].append(row)

    rows: list[dict[str, Any]] = []
    for (arm, stratum, l2_bin, kl_bin, base_bin), values in sorted(groups.items()):
        deltas = [_float(row.get("delta_vs_base_ce")) for row in values]
        losses = [_float(row.get("ce_loss")) for row in values]
        l2s = [_float(row.get("residual_update_l2")) for row in values]
        kls = [_float(row.get("anchor_kl_vs_base")) for row in values]
        flips = [_bool(row.get("prediction_changed_vs_base")) for row in values]
        rows.append(
            {
                "arm": arm,
                "intervention_stratum": stratum,
                "residual_l2_bin": l2_bin,
                "anchor_kl_bin": kl_bin,
                "base_loss_bin": base_bin,
                "token_count": len(values),
                "mean_ce_loss": _mean(losses),
                "mean_delta_vs_base_ce": _mean(deltas),
                "mean_residual_update_l2": _mean(l2s),
                "mean_anchor_kl_vs_base": _mean(kls),
                "prediction_flip_rate": _mean([1.0 if value else 0.0 for value in flips]),
            }
        )
    return rows


def _arm_signal_rows(arm_rows: list[dict[str, str]], strata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense24 = next((row for row in arm_rows if row.get("arm") == "dense_rank24_norm_budgeted"), {})
    budget = _float(dense24.get("heldout_residual_update_l2")) or 0.0
    dense_bins = {
        (
            row["intervention_stratum"],
            row["residual_l2_bin"],
            row["anchor_kl_bin"],
            row["base_loss_bin"],
        ): row
        for row in strata_rows
        if row["arm"] == "dense_rank24_norm_budgeted"
    }
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strata_rows:
        by_arm[row["arm"]].append(row)

    rows: list[dict[str, Any]] = []
    for arm, rows_for_arm in sorted(by_arm.items()):
        arm_metric = next((row for row in arm_rows if row.get("arm") == arm), {})
        family = arm_metric.get("family", "")
        heldout_l2 = _float(arm_metric.get("heldout_residual_update_l2")) or 0.0
        budget_fraction = heldout_l2 / budget if budget > 0.0 else None
        matched = 0
        ce_better = 0
        churn_not_worse = 0
        for row in rows_for_arm:
            dense = dense_bins.get(
                (
                    row["intervention_stratum"],
                    row["residual_l2_bin"],
                    row["anchor_kl_bin"],
                    row["base_loss_bin"],
                )
            )
            if not dense:
                continue
            matched += 1
            if _float(row["mean_delta_vs_base_ce"]) < _float(dense["mean_delta_vs_base_ce"]):
                ce_better += 1
            if _float(row["prediction_flip_rate"]) <= _float(dense["prediction_flip_rate"]):
                churn_not_worse += 1
        ce_delta_vs_dense24 = _float(arm_metric.get("ce_delta_vs_dense24"))
        flip_delta_vs_dense24 = _float(arm_metric.get("flip_delta_vs_dense24"))
        anchor_delta_vs_dense24 = _float(arm_metric.get("anchor_kl_delta_vs_dense24"))
        nontrivial_budget = budget_fraction is not None and budget_fraction >= 0.5
        weak_signal = (
            arm != "dense_rank24_norm_budgeted"
            and family in {"sparse_acsr", "mlp_control"}
            and ce_delta_vs_dense24 is not None
            and ce_delta_vs_dense24 < 0.0
            and flip_delta_vs_dense24 is not None
            and flip_delta_vs_dense24 <= 0.0
            and anchor_delta_vs_dense24 is not None
            and anchor_delta_vs_dense24 <= 0.0
            and nontrivial_budget
            and matched > 0
            and ce_better >= max(1, matched // 2)
        )
        rows.append(
            {
                "arm": arm,
                "family": family,
                "heldout_ce_loss": _float(arm_metric.get("heldout_ce_loss")),
                "heldout_residual_update_l2": heldout_l2,
                "budget_fraction_vs_dense24": budget_fraction,
                "ce_delta_vs_dense24": ce_delta_vs_dense24,
                "anchor_kl_delta_vs_dense24": anchor_delta_vs_dense24,
                "flip_delta_vs_dense24": flip_delta_vs_dense24,
                "matched_dense24_bin_count": matched,
                "matched_bins_ce_better_than_dense24": ce_better,
                "matched_bins_churn_not_worse_than_dense24": churn_not_worse,
                "scientific_signal": (
                    "weak_local_signal_needs_repeat" if weak_signal else "blocked_or_control"
                ),
                "blocker": "" if weak_signal else _blocker(arm, family, budget_fraction, matched, ce_delta_vs_dense24),
            }
        )
    return rows


def _gate_rows(source_rows: list[dict[str, Any]], arm_signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = [row for row in arm_signal_rows if row["scientific_signal"] == "weak_local_signal_needs_repeat"]
    return [
        _criterion(
            "source_artifacts_pass",
            all(row["passed"] for row in source_rows),
            "all source artifacts are present and pass schema checks",
            [row["criterion"] for row in source_rows if not row["passed"]],
            "one or more source artifacts are missing or invalid",
        ),
        _criterion(
            "gpu_repeat_not_required_without_nontrivial_budget_signal",
            not signals,
            "no sparse/MLP challenger has matched-bin CE/churn signal at >=50% dense24 L2",
            [row["arm"] for row in signals],
            "at least one challenger has a weak local matched-strata signal that needs review",
        ),
    ]


def _interpretation(status: str, warrants_runpod: bool, arm_signal_rows: list[dict[str, Any]]) -> str:
    if status != "pass":
        return "The synthesis failed closed because required pilot artifacts were not available."
    if warrants_runpod:
        return (
            "A challenger has a weak local matched-strata signal at nontrivial residual budget, "
            "but promotion is still disallowed. Review the strata before any GPU repeat."
        )
    if _has_nontrivial_ce_interference_signal(arm_signal_rows):
        return (
            "A sparse/MLP challenger now reaches a nontrivial dense24 residual-L2 budget and improves CE, "
            "but the improvement comes with worse anchor KL or prediction-flip churn. Treat this as "
            "target adaptation with interference, not as a reusable low-churn correction."
        )
    low_budget = [
        row["arm"]
        for row in arm_signal_rows
        if row["family"] in {"sparse_acsr", "mlp_control"}
        and (row.get("budget_fraction_vs_dense24") or 0.0) < 0.5
        and (row.get("ce_delta_vs_dense24") or 0.0) < 0.0
    ]
    if low_budget:
        return (
            "Sparse/MLP challengers show lower CE or churn in some aggregate rows, but the "
            "effect occurs below the nontrivial dense24 residual-L2 budget fraction. Treat this "
            "as direction-quality evidence only; the next step should diagnose why the trainable "
            "sparse/MLP arms still underuse the dense24 budget before any GPU repeat."
        )
    return "No sparse/MLP challenger shows a matched-strata signal strong enough to justify GPU repetition."


def _has_nontrivial_ce_interference_signal(arm_signal_rows: list[dict[str, Any]]) -> bool:
    for row in arm_signal_rows:
        if row["family"] not in {"sparse_acsr", "mlp_control"}:
            continue
        budget_fraction = row.get("budget_fraction_vs_dense24") or 0.0
        ce_delta = row.get("ce_delta_vs_dense24")
        anchor_delta = row.get("anchor_kl_delta_vs_dense24")
        flip_delta = row.get("flip_delta_vs_dense24")
        if (
            budget_fraction >= 0.5
            and ce_delta is not None
            and ce_delta < 0.0
            and (
                (anchor_delta is not None and anchor_delta > 0.0)
                or (flip_delta is not None and flip_delta > 0.0)
            )
        ):
            return True
    return False


def _blocker(
    arm: str,
    family: str,
    budget_fraction: float | None,
    matched: int,
    ce_delta_vs_dense24: float | None,
) -> str:
    if arm == "dense_rank24_norm_budgeted":
        return "dense24 comparator"
    if family not in {"sparse_acsr", "mlp_control"}:
        return "control_or_null_arm"
    if budget_fraction is None or budget_fraction < 0.5:
        return "residual_l2_fraction_below_nontrivial_budget"
    if matched <= 0:
        return "no_matched_dense24_strata_bins"
    if ce_delta_vs_dense24 is None or ce_delta_vs_dense24 >= 0.0:
        return "does_not_beat_dense24_ce"
    return "matched_strata_churn_or_ce_gate_not_met"


def _residual_l2_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.05:
        return "l2_000_005"
    if value < 0.25:
        return "l2_005_025"
    if value < 0.75:
        return "l2_025_075"
    if value < 1.10:
        return "l2_075_110"
    return "l2_over_110"


def _anchor_kl_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.0001:
        return "kl_lt_0001"
    if value < 0.001:
        return "kl_0001_001"
    if value < 0.005:
        return "kl_001_005"
    return "kl_ge_005"


def _base_loss_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 3.25:
        return "base_ce_lt_325"
    if value < 3.75:
        return "base_ce_325_375"
    if value < 4.25:
        return "base_ce_375_425"
    return "base_ce_ge_425"


def _criterion(criterion: str, passed: bool, threshold: Any, actual: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    arm_signal_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_status.csv", source_rows)
    _write_csv(out_dir / "arm_signal_summary.csv", arm_signal_rows)
    _write_csv(out_dir / "strata_summary.csv", strata_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# Norm-Budgeted Churn Strata Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- RunPod repeat warranted: `{summary['runpod_repeat_warranted']}`",
        f"- Strata rows: `{summary['strata_row_count']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        summary["interpretation"],
    ]
    if summary["failures"]:
        lines.extend(["", "## Failed or Blocking Gates"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _mean(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return sum(real) / len(real)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_norm_budgeted_churn_strata_synthesis(pilot_dir=args.pilot_dir, out_dir=args.out)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": str(args.out)}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
