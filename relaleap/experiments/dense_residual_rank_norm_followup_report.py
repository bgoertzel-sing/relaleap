"""Follow-up report for dense residual rank/norm sensitivity."""

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
DEFAULT_RANK_NORM_BENCHMARK_DIR = Path(
    "results/reports/dense_residual_rank_norm_interference_benchmark"
)
DEFAULT_OUT_DIR = Path("results/reports/dense_residual_rank_norm_followup")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "rank_ladder.csv",
    "norm_sensitivity.csv",
    "next_matrix.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_residual_rank_norm_followup_report(
    *,
    common_benchmark_dir: Path = DEFAULT_COMMON_BENCHMARK_DIR,
    rank_norm_benchmark_dir: Path = DEFAULT_RANK_NORM_BENCHMARK_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Summarize current dense rank/norm evidence and define the next matrix."""

    start = time.time()
    common_summary = _read_json(common_benchmark_dir / "summary.json")
    arm_rows = _read_csv(common_benchmark_dir / "arm_metrics.csv")
    norm_rows = _read_csv(common_benchmark_dir / "norm_sweep.csv")
    rank_summary = _read_json(rank_norm_benchmark_dir / "summary.json")
    rank_gate_rows = _read_csv(rank_norm_benchmark_dir / "gate_criteria.csv")

    gate_rows = _preflight_rows(
        common_benchmark_dir,
        rank_norm_benchmark_dir,
        common_summary,
        arm_rows,
        norm_rows,
        rank_summary,
        rank_gate_rows,
    )
    rank_ladder = _rank_ladder_rows(arm_rows)
    norm_sensitivity = _norm_sensitivity_rows(norm_rows, arm_rows)
    next_matrix = _next_matrix_rows(rank_ladder, norm_sensitivity)
    gate_rows.extend(_followup_gate_rows(rank_ladder, norm_sensitivity, next_matrix))

    status = "pass" if all(row["passed"] for row in gate_rows) else "fail"
    decision = (
        "dense_rank_norm_followup_matrix_selected"
        if status == "pass"
        else "dense_rank_norm_followup_failed_closed"
    )
    selected_next_step = (
        "run local dense rank/norm matrix with ranks 1, 4, 8, 16, 24 and norm scales 0.50, 0.75, 1.00 before any GPU validation"
        if status == "pass"
        else "repair dense rank/norm source artifacts before selecting a matrix"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": (
            "dense_advantage_requires_rank_norm_sensitivity_test"
            if status == "pass"
            else "dense_rank_norm_sensitivity_not_established"
        ),
        "common_benchmark_dir": str(common_benchmark_dir),
        "rank_norm_benchmark_dir": str(rank_norm_benchmark_dir),
        "rank_ladder_rows": rank_ladder,
        "norm_sensitivity_rows": norm_sensitivity,
        "next_matrix_rows": next_matrix,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]],
        "selected_next_step": selected_next_step,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, rank_ladder, norm_sensitivity, next_matrix, gate_rows)
    return summary


def _preflight_rows(
    common_benchmark_dir: Path,
    rank_norm_benchmark_dir: Path,
    common_summary: dict[str, Any],
    arm_rows: list[dict[str, str]],
    norm_rows: list[dict[str, str]],
    rank_summary: dict[str, Any],
    rank_gate_rows: list[dict[str, str]],
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
            "missing common arm metrics",
        ),
        _criterion(
            "common_norm_sweep_present",
            bool(norm_rows),
            "common benchmark norm sweep exists",
            str(common_benchmark_dir / "norm_sweep.csv"),
            "missing common norm sweep",
        ),
        _criterion(
            "rank_norm_benchmark_passed",
            rank_summary.get("status") == "pass",
            "dense rank/norm/interference benchmark has already passed",
            {
                "path": str(rank_norm_benchmark_dir / "summary.json"),
                "status": rank_summary.get("status"),
            },
            "rank/norm/interference benchmark is missing or did not pass",
        ),
        _criterion(
            "rank_norm_gate_rows_present",
            bool(rank_gate_rows),
            "dense rank/norm/interference gate rows exist",
            str(rank_norm_benchmark_dir / "gate_criteria.csv"),
            "missing dense rank/norm/interference gate rows",
        ),
    ]


def _rank_ladder_rows(arm_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sparse = _row_by_arm(arm_rows, "sparse_contextual_topk2")
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    sparse_l2 = _float_or_none(sparse.get("heldout_residual_update_l2"))
    sparse_active = _float_or_none(sparse.get("active_params_proxy"))
    rows: list[dict[str, Any]] = []
    for row in arm_rows:
        arm = row.get("arm", "")
        if arm != "rank_flop_matched_causal_dense" and not arm.startswith("dense_bottleneck_causal_rank"):
            continue
        rank = _dense_rank(row)
        delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        l2 = _float_or_none(row.get("heldout_residual_update_l2"))
        active = _float_or_none(row.get("active_params_proxy"))
        rows.append(
            {
                "arm": arm,
                "rank": rank,
                "heldout_delta_vs_base_ce": delta,
                "heldout_residual_update_l2": l2,
                "delta_minus_sparse_topk2": _subtract(delta, sparse_delta),
                "norm_ratio_to_sparse_topk2": _safe_divide(l2, sparse_l2),
                "active_ratio_to_sparse_topk2": _safe_divide(active, sparse_active),
                "beats_sparse_topk2": (
                    delta is not None and sparse_delta is not None and delta < sparse_delta
                ),
            }
        )
    return sorted(rows, key=lambda row: int(row.get("rank") or 0))


def _norm_sensitivity_rows(
    norm_rows: list[dict[str, str]],
    arm_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sparse = _row_by_arm(arm_rows, "sparse_contextual_topk2")
    dense = _row_by_arm(arm_rows, "rank_flop_matched_causal_dense")
    sparse_delta = _float_or_none(sparse.get("heldout_delta_vs_base_ce"))
    sparse_l2 = _float_or_none(sparse.get("heldout_residual_update_l2"))
    raw_dense_l2 = _float_or_none(dense.get("raw_heldout_residual_update_l2"))
    matched_dense_l2 = _float_or_none(dense.get("heldout_residual_update_l2"))
    matched_scale = _float_or_none(dense.get("posthoc_residual_norm_scale"))
    rows: list[dict[str, Any]] = []
    for row in norm_rows:
        if row.get("family") != "dense" and row.get("arm") != "sparse_contextual_topk2":
            continue
        delta = _float_or_none(row.get("heldout_delta_vs_base_ce"))
        l2 = _float_or_none(row.get("heldout_residual_update_l2"))
        rows.append(
            {
                "arm": row.get("arm", ""),
                "family": row.get("family", ""),
                "rank": row.get("rank", ""),
                "heldout_delta_vs_base_ce": delta,
                "heldout_residual_update_l2": l2,
                "norm_ratio_to_sparse_topk2": _safe_divide(l2, sparse_l2),
                "delta_minus_sparse_topk2": _subtract(delta, sparse_delta),
                "active_compute_pareto_front": row.get("active_compute_pareto_front", ""),
            }
        )
    rows.append(
        {
            "arm": "rank_flop_matched_causal_dense_raw_before_posthoc_norm_match",
            "family": "dense",
            "rank": _dense_rank(dense),
            "heldout_delta_vs_base_ce": "",
            "heldout_residual_update_l2": raw_dense_l2,
            "norm_ratio_to_sparse_topk2": _safe_divide(raw_dense_l2, sparse_l2),
            "delta_minus_sparse_topk2": "",
            "active_compute_pareto_front": "",
        }
    )
    rows.append(
        {
            "arm": "rank_flop_matched_causal_dense_posthoc_norm_scale",
            "family": "dense",
            "rank": _dense_rank(dense),
            "heldout_delta_vs_base_ce": "",
            "heldout_residual_update_l2": matched_dense_l2,
            "norm_ratio_to_sparse_topk2": _safe_divide(matched_dense_l2, sparse_l2),
            "delta_minus_sparse_topk2": "",
            "active_compute_pareto_front": "",
            "posthoc_residual_norm_scale": matched_scale,
        }
    )
    return rows


def _next_matrix_rows(
    rank_ladder: list[dict[str, Any]],
    norm_sensitivity: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observed_ranks = {int(row["rank"]) for row in rank_ladder if row.get("rank") not in ("", None)}
    sparse_matched_rank = max(observed_ranks) if observed_ranks else 24
    candidate_ranks = [rank for rank in (1, 4, 8, 16, sparse_matched_rank) if rank > 0]
    rows: list[dict[str, Any]] = []
    for rank in sorted(set(candidate_ranks)):
        for scale in (0.5, 0.75, 1.0):
            rows.append(
                {
                    "candidate": f"dense_causal_rank{rank}_norm_scale_{scale:.2f}",
                    "rank": rank,
                    "norm_scale_vs_sparse_topk2": scale,
                    "purpose": _matrix_purpose(rank, sparse_matched_rank, scale),
                    "status": "selected_for_next_local_cpu_matrix",
                }
            )
    if not any(row.get("arm") == "rank_flop_matched_causal_dense_raw_before_posthoc_norm_match" for row in norm_sensitivity):
        rows.append(
            {
                "candidate": "raw_dense_norm_reconstruction",
                "rank": sparse_matched_rank,
                "norm_scale_vs_sparse_topk2": "raw",
                "purpose": "recover raw dense residual norm before post-hoc matching",
                "status": "blocked_missing_raw_norm_row",
            }
        )
    return rows


def _followup_gate_rows(
    rank_ladder: list[dict[str, Any]],
    norm_sensitivity: list[dict[str, Any]],
    next_matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dense_matched = _row_by_arm_any(rank_ladder, "rank_flop_matched_causal_dense")
    rank1 = _row_by_arm_any(rank_ladder, "dense_bottleneck_causal_rank1")
    sparse = _row_by_arm_any(norm_sensitivity, "sparse_contextual_topk2")
    return [
        _criterion(
            "matched_dense_rank_row_present",
            bool(dense_matched),
            "matched dense rank row is available",
            dense_matched.get("arm", ""),
            "missing matched causal dense row",
        ),
        _criterion(
            "rank1_dense_probe_present",
            bool(rank1),
            "rank-1 dense probe is available",
            rank1.get("arm", ""),
            "missing rank-1 dense bottleneck row",
        ),
        _criterion(
            "sparse_reference_row_present",
            bool(sparse),
            "sparse top-k2 reference row is available",
            sparse.get("arm", ""),
            "missing sparse top-k2 reference row",
        ),
        _criterion(
            "matched_dense_beats_sparse_but_rank1_does_not",
            bool(dense_matched.get("beats_sparse_topk2")) and not bool(rank1.get("beats_sparse_topk2")),
            "current evidence localizes dense advantage above rank 1",
            {
                "matched_dense_beats_sparse": dense_matched.get("beats_sparse_topk2"),
                "rank1_beats_sparse": rank1.get("beats_sparse_topk2"),
            },
            "rank ladder does not isolate a missing-rank question",
        ),
        _criterion(
            "next_matrix_selected",
            any(row.get("status") == "selected_for_next_local_cpu_matrix" for row in next_matrix),
            "next local dense rank/norm matrix is selected",
            len(next_matrix),
            "no next local matrix rows were selected",
        ),
    ]


def _matrix_purpose(rank: int, matched_rank: int, scale: float) -> str:
    if rank == matched_rank and scale == 1.0:
        return "repeat current matched dense baseline as an anchor"
    if rank == 1:
        return "check whether the dense effect appears at minimal rank"
    if scale < 1.0:
        return "test whether the dense advantage survives smaller residual norms"
    return "bracket the minimal dense rank needed to beat sparse top-k2"


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    rank_ladder: list[dict[str, Any]],
    norm_sensitivity: list[dict[str, Any]],
    next_matrix: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "rank_ladder.csv", rank_ladder)
    _write_csv(out_dir / "norm_sensitivity.csv", norm_sensitivity)
    _write_csv(out_dir / "next_matrix.csv", next_matrix)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Residual Rank/Norm Follow-up",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This report consumes existing local dense-control artifacts and selects the next bounded CPU matrix. It does not treat ACSR or sparse top-k2 as promoted; sparse top-k2 remains a comparator.",
    ]
    if summary.get("failures"):
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _row_by_arm(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _row_by_arm_any(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arm") == arm:
            return row
    return {}


def _dense_rank(row: dict[str, Any]) -> int:
    if row.get("rank") not in ("", None):
        value = _float_or_none(row.get("rank"))
        return int(value or 0)
    arm = str(row.get("arm", ""))
    if arm.startswith("dense_bottleneck_causal_rank"):
        return int(arm.rsplit("rank", 1)[1])
    return 0


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


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
    parser.add_argument("--common-benchmark-dir", type=Path, default=DEFAULT_COMMON_BENCHMARK_DIR)
    parser.add_argument(
        "--rank-norm-benchmark-dir",
        type=Path,
        default=DEFAULT_RANK_NORM_BENCHMARK_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_residual_rank_norm_followup_report(
        common_benchmark_dir=args.common_benchmark_dir,
        rank_norm_benchmark_dir=args.rank_norm_benchmark_dir,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
