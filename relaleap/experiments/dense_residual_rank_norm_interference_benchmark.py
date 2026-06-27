"""Rank/norm/interference report for dense residual controls."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_COMMON_BENCHMARK_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
DEFAULT_OUT_DIR = Path("results/reports/dense_residual_rank_norm_interference_benchmark")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "rank_norm_rows.csv",
    "interference_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_ARMS = (
    "sparse_contextual_topk2",
    "sparse_rank_matched_topk1",
    "rank_flop_matched_causal_dense",
    "rank_flop_matched_token_position_dense",
)


def run_dense_residual_rank_norm_interference_benchmark(
    *,
    common_benchmark_dir: Path = DEFAULT_COMMON_BENCHMARK_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local post-hoc benchmark around dense residual controls."""

    start = time.time()
    common_summary = _read_json(common_benchmark_dir / "summary.json")
    arm_rows = _read_csv(common_benchmark_dir / "arm_metrics.csv")
    per_token_rows = _read_csv(common_benchmark_dir / "per_token_metrics.csv")
    preflight = _preflight_rows(common_benchmark_dir, common_summary, arm_rows, per_token_rows)

    if any(not row["passed"] for row in preflight):
        summary = _summary(
            status="fail",
            decision="dense_residual_rank_norm_interference_failed_closed",
            claim_status="benchmark_not_run",
            start=start,
            common_benchmark_dir=common_benchmark_dir,
            rank_norm_rows=[],
            interference_rows=[],
            gate_rows=preflight,
            out_dir=out_dir,
        )
        _write_artifacts(out_dir, summary, [], [], preflight)
        return summary

    rank_norm_rows = _rank_norm_rows(arm_rows)
    interference_rows = _interference_rows(per_token_rows)
    gate_rows = preflight + _benchmark_gate_rows(rank_norm_rows, interference_rows)
    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    summary = _summary(
        status=status,
        decision=(
            "dense_residual_rank_norm_interference_supported"
            if status == "pass"
            else "dense_residual_rank_norm_interference_failed_gate"
        ),
        claim_status=(
            "causal_dense_control_remains_active_local_baseline"
            if status == "pass"
            else "causal_dense_control_needs_repair_or_repeat_before_escalation"
        ),
        start=start,
        common_benchmark_dir=common_benchmark_dir,
        rank_norm_rows=rank_norm_rows,
        interference_rows=interference_rows,
        gate_rows=gate_rows,
        out_dir=out_dir,
    )
    _write_artifacts(out_dir, summary, rank_norm_rows, interference_rows, gate_rows)
    return summary


def _rank_norm_rows(arm_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_arm = {row.get("arm", ""): row for row in arm_rows}
    sparse_l2 = _float_or_none(by_arm.get("sparse_contextual_topk2", {}).get("heldout_residual_update_l2"))
    rows: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        source = by_arm.get(arm, {})
        heldout_delta = _float_or_none(source.get("heldout_delta_vs_base_ce"))
        heldout_l2 = _float_or_none(source.get("heldout_residual_update_l2"))
        active_params = _float_or_none(source.get("active_params_proxy"))
        flops = _float_or_none(source.get("flops_proxy"))
        rows.append(
            {
                "arm": arm,
                "family": source.get("family", ""),
                "top_k": source.get("top_k", ""),
                "rank": source.get("rank", ""),
                "heldout_delta_vs_base_ce": heldout_delta,
                "heldout_residual_update_l2": heldout_l2,
                "norm_ratio_to_sparse_topk2": _safe_divide(heldout_l2, sparse_l2),
                "ce_gain_per_l2": _safe_divide(heldout_delta, heldout_l2),
                "active_params_proxy": active_params,
                "flops_proxy": flops,
                "ce_gain_per_active_param": _safe_divide(heldout_delta, active_params),
                "ce_gain_per_flop_proxy": _safe_divide(heldout_delta, flops),
            }
        )
    return rows


def _interference_rows(per_token_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in REQUIRED_ARMS:
        for split in ("train", "heldout"):
            values = [
                row
                for row in per_token_rows
                if row.get("arm") == arm and row.get("split") == split
            ]
            deltas = [_float_or_none(row.get("delta_vs_base_ce")) for row in values]
            l2s = [_float_or_none(row.get("residual_update_l2")) for row in values]
            deltas = [value for value in deltas if value is not None]
            l2s = [value for value in l2s if value is not None]
            mean_delta = _mean(deltas)
            mean_l2 = _mean(l2s)
            rows.append(
                {
                    "row_type": "arm_split",
                    "arm": arm,
                    "split": split,
                    "token_count": len(deltas),
                    "mean_delta_vs_base_ce": mean_delta,
                    "mean_residual_update_l2": mean_l2,
                    "ce_gain_per_l2": _safe_divide(mean_delta, mean_l2),
                    "damage_fraction": _fraction(deltas, lambda value: value > 0.0),
                    "improvement_fraction": _fraction(deltas, lambda value: value < 0.0),
                }
            )

    rows.extend(_paired_interference_rows(per_token_rows, "rank_flop_matched_causal_dense", "sparse_contextual_topk2"))
    rows.extend(
        _paired_interference_rows(
            per_token_rows,
            "rank_flop_matched_causal_dense",
            "rank_flop_matched_token_position_dense",
        )
    )
    return rows


def _paired_interference_rows(
    per_token_rows: list[dict[str, str]],
    left_arm: str,
    right_arm: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "heldout"):
        left = {
            row.get("token_index"): _float_or_none(row.get("delta_vs_base_ce"))
            for row in per_token_rows
            if row.get("arm") == left_arm and row.get("split") == split
        }
        right = {
            row.get("token_index"): _float_or_none(row.get("delta_vs_base_ce"))
            for row in per_token_rows
            if row.get("arm") == right_arm and row.get("split") == split
        }
        paired = [
            (left[index], right[index])
            for index in sorted(set(left).intersection(right), key=lambda value: int(value or 0))
            if left[index] is not None and right[index] is not None
        ]
        advantages = [left_value - right_value for left_value, right_value in paired]
        out.append(
            {
                "row_type": "paired_arms",
                "arm": left_arm,
                "reference_arm": right_arm,
                "split": split,
                "token_count": len(advantages),
                "mean_delta_advantage_vs_reference": _mean(advantages),
                "left_wins_fraction": _fraction(advantages, lambda value: value < 0.0),
                "left_loses_fraction": _fraction(advantages, lambda value: value > 0.0),
            }
        )
    return out


def _benchmark_gate_rows(
    rank_norm_rows: list[dict[str, Any]],
    interference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rank_by_arm = {str(row.get("arm")): row for row in rank_norm_rows}
    heldout_by_arm = {
        str(row.get("arm")): row
        for row in interference_rows
        if row.get("row_type") == "arm_split" and row.get("split") == "heldout"
    }
    dense_sparse_pair = _paired_row(
        interference_rows,
        "rank_flop_matched_causal_dense",
        "sparse_contextual_topk2",
        "heldout",
    )
    dense_token_pair = _paired_row(
        interference_rows,
        "rank_flop_matched_causal_dense",
        "rank_flop_matched_token_position_dense",
        "heldout",
    )
    sparse = rank_by_arm.get("sparse_contextual_topk2", {})
    dense = rank_by_arm.get("rank_flop_matched_causal_dense", {})
    token_dense = rank_by_arm.get("rank_flop_matched_token_position_dense", {})
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    dense_delta = _float_or_none(dense.get("heldout_delta_vs_base_ce"))
    token_dense_delta = _float_or_none(token_dense.get("heldout_delta_vs_base_ce"))
    sparse_gain_l2 = _float_or_none(sparse.get("ce_gain_per_l2"))
    dense_gain_l2 = _float_or_none(dense.get("ce_gain_per_l2"))
    dense_norm_ratio = _float_or_none(dense.get("norm_ratio_to_sparse_topk2"))
    sparse_damage = _float_or_none(heldout_by_arm.get("sparse_contextual_topk2", {}).get("damage_fraction"))
    dense_damage = _float_or_none(heldout_by_arm.get("rank_flop_matched_causal_dense", {}).get("damage_fraction"))
    return [
        _criterion(
            "required_rank_norm_rows_present",
            set(REQUIRED_ARMS).issubset(rank_by_arm),
            "rank/norm rows exist for dense and sparse comparators",
            sorted(rank_by_arm),
            "missing required rank/norm comparator row",
        ),
        _criterion(
            "dense_norm_matched_to_sparse_topk2",
            dense_norm_ratio is not None and 0.9 <= dense_norm_ratio <= 1.1,
            "causal dense held-out residual L2 is within 10% of sparse top-k2",
            dense_norm_ratio,
            "causal dense residual norm is not tightly matched to sparse top-k2",
        ),
        _criterion(
            "causal_dense_beats_sparse_topk2_heldout",
            dense_delta is not None and sparse_delta is not None and dense_delta < sparse_delta,
            "causal dense held-out CE delta is better than sparse contextual top-k2",
            {"dense_delta": dense_delta, "sparse_delta": sparse_delta},
            "causal dense did not beat sparse top-k2 on held-out CE",
        ),
        _criterion(
            "causal_dense_beats_token_position_dense_heldout",
            dense_delta is not None and token_dense_delta is not None and dense_delta < token_dense_delta,
            "causal dense held-out CE delta beats token-position dense null",
            {"dense_delta": dense_delta, "token_position_dense_delta": token_dense_delta},
            "causal dense did not beat token-position dense null",
        ),
        _criterion(
            "causal_dense_gain_per_l2_beats_sparse",
            dense_gain_l2 is not None and sparse_gain_l2 is not None and dense_gain_l2 < sparse_gain_l2,
            "causal dense CE gain per residual L2 is better than sparse top-k2",
            {"dense_gain_per_l2": dense_gain_l2, "sparse_gain_per_l2": sparse_gain_l2},
            "causal dense gain per L2 did not beat sparse top-k2",
        ),
        _criterion(
            "causal_dense_damage_not_higher_than_sparse",
            dense_damage is not None and sparse_damage is not None and dense_damage <= sparse_damage + 0.05,
            "causal dense held-out per-token damage fraction is no worse than sparse top-k2 by more than 0.05",
            {"dense_damage": dense_damage, "sparse_damage": sparse_damage},
            "causal dense has materially higher held-out per-token damage",
        ),
        _criterion(
            "paired_dense_sparse_advantage_recorded",
            _float_or_none(dense_sparse_pair.get("mean_delta_advantage_vs_reference")) is not None,
            "paired held-out dense-vs-sparse advantage is recorded",
            dense_sparse_pair,
            "missing paired dense-vs-sparse held-out interference row",
        ),
        _criterion(
            "paired_dense_token_null_advantage_positive",
            (
                _float_or_none(dense_token_pair.get("mean_delta_advantage_vs_reference")) is not None
                and _float_or_none(dense_token_pair.get("mean_delta_advantage_vs_reference")) < 0.0
            ),
            "paired held-out causal dense advantage beats token-position dense null",
            dense_token_pair,
            "causal dense did not beat token-position dense null in paired held-out rows",
        ),
    ]


def _preflight_rows(
    common_benchmark_dir: Path,
    common_summary: dict[str, Any],
    arm_rows: list[dict[str, str]],
    per_token_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "common_summary_present",
            bool(common_summary),
            "common benchmark summary exists",
            str(common_benchmark_dir / "summary.json"),
            "missing common benchmark summary",
        ),
        _criterion(
            "common_arm_metrics_present",
            bool(arm_rows),
            "common benchmark arm metrics exist",
            str(common_benchmark_dir / "arm_metrics.csv"),
            "missing common benchmark arm metrics",
        ),
        _criterion(
            "common_per_token_metrics_present",
            bool(per_token_rows),
            "common benchmark per-token metrics exist",
            str(common_benchmark_dir / "per_token_metrics.csv"),
            "missing common benchmark per-token metrics",
        ),
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    start: float,
    common_benchmark_dir: Path,
    rank_norm_rows: list[dict[str, Any]],
    interference_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    failures = [row for row in gate_rows if not row["passed"]]
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "common_benchmark_dir": str(common_benchmark_dir),
        "rank_norm_row_count": len(rank_norm_rows),
        "interference_row_count": len(interference_rows),
        "rank_norm_rows": rank_norm_rows,
        "interference_rows": interference_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": (
            "run a seed-2 local repeat of the dense rank/norm/interference benchmark before any GPU validation"
            if status == "pass"
            else "repair common dense-control artifacts before repeating the benchmark"
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    rank_norm_rows: list[dict[str, Any]],
    interference_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "rank_norm_rows.csv", rank_norm_rows)
    _write_csv(out_dir / "interference_rows.csv", interference_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Residual Rank/Norm/Interference Benchmark",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Source: `{summary['common_benchmark_dir']}`",
        "",
        "This report treats sparse contextual top-k2 as a comparator and audits whether the causal dense control remains better under matched residual norm, rank/compute proxies, and per-token interference/damage observables.",
    ]
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _paired_row(
    rows: list[dict[str, Any]],
    arm: str,
    reference_arm: str,
    split: str,
) -> dict[str, Any]:
    for row in rows:
        if (
            row.get("row_type") == "paired_arms"
            and row.get("arm") == arm
            and row.get("reference_arm") == reference_arm
            and row.get("split") == split
        ):
            return row
    return {}


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


def _fraction(values: list[float], predicate: Any) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_divide(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or abs(right) < 1e-12:
        return None
    return left / right


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark-dir", type=Path, default=DEFAULT_COMMON_BENCHMARK_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_residual_rank_norm_interference_benchmark(
        common_benchmark_dir=args.common_benchmark_dir,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
