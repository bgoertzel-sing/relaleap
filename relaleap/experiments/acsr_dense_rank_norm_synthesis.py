"""Synthesize ACSR evidence against the dense rank/norm control matrix."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ACSR_SYNTHESIS_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_synthesis"
)
DEFAULT_COMMON_BENCHMARK_DIR = Path("results/reports/acsr_common_causal_residual_benchmark")
DEFAULT_DENSE_MATRIX_DIR = Path("results/reports/dense_residual_rank_norm_matrix")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_dense_rank_norm_synthesis")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_status.csv",
    "comparison_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_dense_rank_norm_synthesis(
    *,
    acsr_synthesis_dir: Path = DEFAULT_ACSR_SYNTHESIS_DIR,
    common_benchmark_dir: Path = DEFAULT_COMMON_BENCHMARK_DIR,
    dense_matrix_dir: Path = DEFAULT_DENSE_MATRIX_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local decision report from existing artifacts."""

    start = time.time()
    acsr_summary = _read_json(acsr_synthesis_dir / "summary.json")
    common_summary = _read_json(common_benchmark_dir / "summary.json")
    dense_summary = _read_json(dense_matrix_dir / "summary.json")
    common_arms = _read_csv(common_benchmark_dir / "arm_metrics.csv")
    dense_ranks = _read_csv(dense_matrix_dir / "rank_summary.csv")

    source_rows = _source_rows(
        acsr_synthesis_dir=acsr_synthesis_dir,
        common_benchmark_dir=common_benchmark_dir,
        dense_matrix_dir=dense_matrix_dir,
        acsr_summary=acsr_summary,
        common_summary=common_summary,
        dense_summary=dense_summary,
    )
    comparison_rows = _comparison_rows(acsr_summary, common_arms, dense_ranks)
    strategy_review = _strategy_review(strategy_review_path)
    gate_rows = _gate_rows(source_rows, comparison_rows, strategy_review)
    source_ok = all(row["passed"] for row in source_rows)
    gates_ok = all(row["passed"] for row in gate_rows)

    if not source_ok:
        status = "fail"
        decision = "acsr_dense_rank_norm_synthesis_failed_closed"
        claim_status = "source_artifacts_missing_or_failed"
        selected_next_step = "repair missing or failing local source artifacts before any ACSR interpretation"
    elif gates_ok:
        status = "pass"
        decision = "acsr_sparse_mechanism_gate_still_coherent"
        claim_status = "acsr_sparse_claim_not_blocked_by_dense_rank_norm_controls"
        selected_next_step = "define the next local ACSR mechanism gate with dense rank/norm controls carried forward"
    else:
        status = "pass"
        decision = "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls"
        claim_status = "dense_rank16_24_controls_explain_ce_gain_threshold"
        selected_next_step = (
            "design a narrower local mechanism gate requiring ACSR to beat or separate from "
            "rank-16/24 dense controls on retention, intervention fingerprints, and churn before GPU work"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "source_dirs": {
            "acsr_synthesis": str(acsr_synthesis_dir),
            "common_benchmark": str(common_benchmark_dir),
            "dense_matrix": str(dense_matrix_dir),
        },
        "source_status": source_rows,
        "comparison_metrics": comparison_rows,
        "gate_criteria": gate_rows,
        "failures": [row for row in gate_rows if not row["passed"]]
        + [row for row in source_rows if not row["passed"]],
        "strategy_review": strategy_review,
        "interpretation": _interpretation(status, gates_ok, strategy_review),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, source_rows, comparison_rows, gate_rows)
    return summary


def _source_rows(
    *,
    acsr_synthesis_dir: Path,
    common_benchmark_dir: Path,
    dense_matrix_dir: Path,
    acsr_summary: dict[str, Any],
    common_summary: dict[str, Any],
    dense_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "acsr_synthesis_passed",
            acsr_summary.get("status") == "pass",
            "ACSR two-seed local synthesis passed",
            {"path": str(acsr_synthesis_dir / "summary.json"), "decision": acsr_summary.get("decision")},
            "ACSR synthesis is missing or not passing",
        ),
        _criterion(
            "common_benchmark_present",
            bool(common_summary.get("decision")),
            "common sparse/dense benchmark summary exists",
            {"path": str(common_benchmark_dir / "summary.json"), "decision": common_summary.get("decision")},
            "common benchmark summary is missing",
        ),
        _criterion(
            "dense_matrix_passed",
            dense_summary.get("status") == "pass",
            "dense rank/norm matrix passed",
            {"path": str(dense_matrix_dir / "summary.json"), "decision": dense_summary.get("decision")},
            "dense rank/norm matrix is missing or not passing",
        ),
    ]


def _comparison_rows(
    acsr_summary: dict[str, Any],
    common_arms: list[dict[str, str]],
    dense_ranks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    aggregates = _as_dict(acsr_summary.get("aggregates"))
    arms = {row.get("arm", ""): row for row in common_arms}
    sparse = arms.get("sparse_contextual_topk2", {})
    causal_dense = arms.get("rank_flop_matched_causal_dense", {})
    dense_rank_rows = [
        {
            "rank": int(_float(row.get("rank")) or 0),
            "best_delta": _float(row.get("best_heldout_delta_vs_base_ce")),
            "delta_minus_sparse": _float(row.get("best_delta_minus_sparse_topk2")),
            "beats_sparse": _bool(row.get("beats_sparse_topk2")),
        }
        for row in dense_ranks
    ]
    winning = [row for row in dense_rank_rows if row["beats_sparse"]]
    losing = [row for row in dense_rank_rows if not row["beats_sparse"]]
    acsr_seed_margin = _float(aggregates.get("mean_acsr_minus_causal_ce_loss"))
    dense_rank16 = next((row for row in dense_rank_rows if row["rank"] == 16), {})
    dense_rank24 = next((row for row in dense_rank_rows if row["rank"] == 24), {})
    return [
        {
            "metric": "mean_acsr_minus_causal_ce_loss",
            "value": acsr_seed_margin,
            "interpretation": "negative means ACSR beats causal-feature-safe router across local seeds",
        },
        {
            "metric": "common_benchmark_sparse_minus_rank_flop_dense_delta",
            "value": _delta(
                _float(sparse.get("heldout_delta_vs_base_ce")),
                _float(causal_dense.get("heldout_delta_vs_base_ce")),
            ),
            "interpretation": "positive means rank/FLOP dense beats sparse contextual top-k2",
        },
        {
            "metric": "minimal_dense_rank_beating_sparse",
            "value": min((row["rank"] for row in winning), default=None),
            "interpretation": "lowest dense rank whose full-norm cell beats sparse contextual top-k2",
        },
        {
            "metric": "max_losing_dense_rank",
            "value": max((row["rank"] for row in losing), default=None),
            "interpretation": "highest dense rank still below sparse contextual top-k2",
        },
        {
            "metric": "rank16_delta_minus_sparse",
            "value": dense_rank16.get("delta_minus_sparse"),
            "interpretation": "negative means rank-16 dense beats sparse",
        },
        {
            "metric": "rank24_delta_minus_sparse",
            "value": dense_rank24.get("delta_minus_sparse"),
            "interpretation": "negative means rank-24 dense beats sparse",
        },
        {
            "metric": "mean_acsr_teacher_support_churn",
            "value": _float(aggregates.get("mean_acsr_teacher_support_churn")),
            "interpretation": "lower churn remains supportive only if dense controls are separated on mechanism metrics",
        },
    ]


def _gate_rows(
    source_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = {row["metric"]: row.get("value") for row in comparison_rows}
    minimal_winning = metrics.get("minimal_dense_rank_beating_sparse")
    max_losing = metrics.get("max_losing_dense_rank")
    rank16_margin = metrics.get("rank16_delta_minus_sparse")
    common_dense_margin = metrics.get("common_benchmark_sparse_minus_rank_flop_dense_delta")
    acsr_margin = metrics.get("mean_acsr_minus_causal_ce_loss")
    return [
        _criterion(
            "source_artifacts_all_pass_or_present",
            all(row["passed"] for row in source_rows),
            "all selected source artifacts are available and locally interpretable",
            [row["criterion"] for row in source_rows if not row["passed"]],
            "one or more source artifacts are missing or failed",
        ),
        _criterion(
            "acsr_local_signal_positive",
            acsr_margin is not None and acsr_margin < 0.0,
            "ACSR local synthesis still beats causal-feature-safe router on CE",
            acsr_margin,
            "ACSR local synthesis no longer has a positive CE signal",
        ),
        _criterion(
            "dense_high_rank_does_not_beat_sparse",
            minimal_winning is None,
            "no dense rank/norm cell beats sparse contextual top-k2",
            {"minimal_winning_dense_rank": minimal_winning, "rank16_delta_minus_sparse": rank16_margin},
            "dense rank/norm controls beat sparse contextual top-k2",
        ),
        _criterion(
            "dense_rank_threshold_not_bracketed",
            minimal_winning is None or max_losing is None,
            "dense rank threshold is not bracketed by rank/norm controls",
            {"max_losing_dense_rank": max_losing, "minimal_winning_dense_rank": minimal_winning},
            "dense rank threshold is bracketed, so CE gain is not sparse-specific",
        ),
        _criterion(
            "common_dense_control_not_stronger",
            common_dense_margin is not None and common_dense_margin <= 0.0,
            "common rank/FLOP dense control does not beat sparse contextual top-k2",
            common_dense_margin,
            "common rank/FLOP dense control beats sparse contextual top-k2",
        ),
        _criterion(
            "strategy_review_pivot_recorded",
            strategy_review["present"],
            "latest external strategy review was read and recorded",
            strategy_review,
            "strategy review was absent",
        ),
    ]


def _interpretation(status: str, gates_ok: bool, strategy_review: dict[str, Any]) -> str:
    review_note = ""
    if strategy_review.get("ben_notification_required"):
        review_note = (
            " Latest review is major/notify_ben=true; Ben should be notified that the "
            "automation is retaining the ACSR pivot but narrowing it under dense controls."
        )
    if status != "pass":
        return "The synthesis failed closed because required source artifacts were not available." + review_note
    if gates_ok:
        return (
            "Current dense rank/norm evidence does not block a sparse-support ACSR mechanism "
            "claim, though promotion still requires non-CE gates."
        ) + review_note
    return (
        "ACSR remains locally promising versus shuffled/token-position/null routers, but broad "
        "sparse-support claims are blocked: dense rank/norm controls identify a rank threshold "
        "where dense causal residuals match or beat sparse contextual top-k2. The next work should "
        "test mechanism-specific retention/churn and intervention fingerprints against rank-16/24 "
        "dense controls, not spend GPU on promotion."
    ) + review_note


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "deferred_recommendation_reason": "review file absent",
        }
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action", "verdict"}:
            values[key] = value.strip()
    notify_ben = values.get("notify_ben", "").lower() == "true"
    return {
        "present": True,
        "strategic_change_level": values.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or values.get("strategic_change_level") == "major",
        "verdict": values.get("verdict"),
        "recommended_next_action": values.get("recommended_next_action"),
        "deferred_recommendation_reason": (
            "ACSR pilot skeleton already exists with local artifacts; this run instead synthesized "
            "the later dense rank/norm controls before any GPU or promotion step."
        ),
    }


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_status.csv", source_rows)
    _write_csv(out_dir / "comparison_metrics.csv", comparison_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# ACSR Dense Rank/Norm Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        summary["interpretation"],
    ]
    if summary["failures"]:
        lines.extend(["", "## Failed or Blocking Gates"])
        lines.extend(
            f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"]
        )
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acsr-synthesis-dir", type=Path, default=DEFAULT_ACSR_SYNTHESIS_DIR)
    parser.add_argument("--common-benchmark-dir", type=Path, default=DEFAULT_COMMON_BENCHMARK_DIR)
    parser.add_argument("--dense-matrix-dir", type=Path, default=DEFAULT_DENSE_MATRIX_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_dense_rank_norm_synthesis(
        acsr_synthesis_dir=args.acsr_synthesis_dir,
        common_benchmark_dir=args.common_benchmark_dir,
        dense_matrix_dir=args.dense_matrix_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": str(args.out)}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
