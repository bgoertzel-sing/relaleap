"""Synthesize dense-control retention/churn evidence after ACSR closeout."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/acsr_dense_control_retention_churn_closeout/summary.json")
DEFAULT_DENSE_TRANSFER = Path("results/reports/acsr_dense_residual_transfer_control/summary.json")
DEFAULT_RANK_NORM_DIRS = (
    Path("results/reports/dense_residual_rank_norm_interference_benchmark"),
    Path("results/reports/dense_residual_rank_norm_interference_benchmark_seed2"),
)
DEFAULT_TOPK1_STABILITY = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability/summary.json"
)
DEFAULT_ACSR_RETENTION_CHURN = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_retention_churn_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_residual_control_baseline_retention_churn_synthesis")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "dense_seed_rows.csv",
    "comparison_rows.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_dense_residual_control_baseline_retention_churn_synthesis(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    dense_transfer_path: Path = DEFAULT_DENSE_TRANSFER,
    rank_norm_dirs: tuple[Path, ...] = DEFAULT_RANK_NORM_DIRS,
    topk1_stability_path: Path = DEFAULT_TOPK1_STABILITY,
    acsr_retention_churn_path: Path = DEFAULT_ACSR_RETENTION_CHURN,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local synthesis comparing sparse, rank-matched sparse, and dense controls."""

    start = time.time()
    closeout = _read_json(closeout_path)
    dense_transfer = _read_json(dense_transfer_path)
    rank_norm = [_load_rank_norm(path) for path in rank_norm_dirs]
    topk1 = _read_json(topk1_stability_path)
    acsr_retention = _read_json(acsr_retention_churn_path)
    strategy = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("acsr_dense_control_retention_churn_closeout", closeout_path, closeout),
        _source_row("acsr_dense_residual_transfer_control", dense_transfer_path, dense_transfer),
        *[
            _source_row(f"dense_rank_norm_seed{index + 1}", path / "summary.json", packet["summary"])
            for index, (path, packet) in enumerate(zip(rank_norm_dirs, rank_norm))
        ],
        _source_row("active_rank_matched_topk1_stability", topk1_stability_path, topk1),
        _source_row("acsr_retention_churn_probe", acsr_retention_churn_path, acsr_retention),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    dense_seed_rows = [_dense_seed_row(packet) for packet in rank_norm]
    comparison_rows = _comparison_rows(
        dense_seed_rows=dense_seed_rows,
        dense_transfer=dense_transfer,
        topk1=topk1,
        acsr_retention=acsr_retention,
    )
    gate_rows = _gate_rows(closeout, dense_transfer, dense_seed_rows, topk1, acsr_retention)
    failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    status = "pass" if not failures else "fail"
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] == "claim_blocker"]
    decision = (
        "dense_retention_churn_synthesis_selects_heldout_context_intervention_design"
        if status == "pass"
        else "dense_retention_churn_synthesis_failed_closed"
    )
    claim_status = (
        "dense_controls_active_topk1_retention_local_support_acsr_not_promoted"
        if status == "pass"
        else "dense_control_retention_churn_not_interpretable"
    )
    selected_next_step = (
        "design_heldout_context_intervention_assay_dense_vs_rank_matched_topk1"
        if status == "pass"
        else "repair_dense_control_retention_churn_sources"
    )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "source_rows": source_rows,
        "dense_seed_rows": dense_seed_rows,
        "comparison_rows": comparison_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_blockers_preserved": claim_blockers,
        "strategy_review": strategy,
        "strategy_review_handling": (
            "Latest GPT-5.5-Pro recommendation is accepted as a local/no-RunPod boundary: "
            "support discovery remains frozen, and this synthesis selects a local heldout-context "
            "intervention design rather than GPU validation."
        ),
        "claim_statuses": {
            "dense_residual_controls": "active_baseline" if status == "pass" else "blocked",
            "rank_matched_topk1": "local_retention_churn_support_not_promoted",
            "acsr_support_discovery": "frozen_negative",
            "runpod_validation": "deferred_no_gpu_target",
            "ben_notification": "not_required" if not strategy["ben_notification_required"] else "required",
        },
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_rank_norm(path: Path) -> dict[str, Any]:
    summary = _read_json(path / "summary.json")
    return {
        "path": str(path),
        "summary": summary,
        "rank_norm_rows": _read_csv(path / "rank_norm_rows.csv"),
        "interference_rows": _read_csv(path / "interference_rows.csv"),
    }


def _dense_seed_row(packet: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(packet.get("summary"))
    sparse = _arm(summary, "sparse_contextual_topk2")
    rank_topk1 = _arm(summary, "sparse_rank_matched_topk1")
    dense = _arm(summary, "rank_flop_matched_causal_dense")
    token_dense = _arm(summary, "rank_flop_matched_token_position_dense")
    dense_vs_sparse = _paired(summary, "rank_flop_matched_causal_dense", "sparse_contextual_topk2")
    return {
        "source": packet.get("path", ""),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "claim_status": summary.get("claim_status"),
        "sparse_topk2_heldout_delta": _float_or_none(sparse.get("heldout_delta_vs_base_ce")),
        "rank_matched_topk1_heldout_delta": _float_or_none(rank_topk1.get("heldout_delta_vs_base_ce")),
        "causal_dense_heldout_delta": _float_or_none(dense.get("heldout_delta_vs_base_ce")),
        "token_position_dense_heldout_delta": _float_or_none(token_dense.get("heldout_delta_vs_base_ce")),
        "sparse_topk2_gain_per_l2": _float_or_none(sparse.get("heldout_ce_gain_per_l2")),
        "causal_dense_gain_per_l2": _float_or_none(dense.get("heldout_ce_gain_per_l2")),
        "causal_dense_damage_fraction": _float_or_none(dense.get("heldout_damage_fraction")),
        "sparse_topk2_damage_fraction": _float_or_none(sparse.get("heldout_damage_fraction")),
        "paired_dense_sparse_advantage": _float_or_none(
            dense_vs_sparse.get("mean_delta_advantage_vs_reference")
        ),
        "paired_dense_sparse_left_wins_fraction": _float_or_none(
            dense_vs_sparse.get("left_wins_fraction")
        ),
    }


def _comparison_rows(
    *,
    dense_seed_rows: list[dict[str, Any]],
    dense_transfer: dict[str, Any],
    topk1: dict[str, Any],
    acsr_retention: dict[str, Any],
) -> list[dict[str, Any]]:
    topk1_aggregates = _as_dict(topk1.get("aggregates"))
    return [
        {
            "comparison": "dense_vs_sparse_topk2_rank_norm",
            "packet_count": len(dense_seed_rows),
            "mean_dense_minus_sparse_heldout_delta": _mean(
                [
                    _diff(row.get("causal_dense_heldout_delta"), row.get("sparse_topk2_heldout_delta"))
                    for row in dense_seed_rows
                ]
            ),
            "all_dense_beats_sparse": all(
                _less(row.get("causal_dense_heldout_delta"), row.get("sparse_topk2_heldout_delta"))
                for row in dense_seed_rows
            ),
        },
        {
            "comparison": "dense_transfer_control",
            "packet_count": 1,
            "status": dense_transfer.get("status"),
            "claim_status": dense_transfer.get("claim_status"),
            "sparse_transfer_not_separated_from_dense": (
                dense_transfer.get("claim_status") == "sparse_transfer_not_separated_from_dense_control"
            ),
        },
        {
            "comparison": "rank_matched_topk1_retention_churn",
            "packet_count": topk1.get("packet_count"),
            "status": topk1.get("status"),
            "mean_support_churn_advantage": topk1_aggregates.get("mean_support_churn_advantage"),
            "mean_logit_churn_advantage": topk1_aggregates.get("mean_logit_churn_advantage"),
            "mean_transfer_improvement_advantage": topk1_aggregates.get(
                "mean_transfer_improvement_advantage"
            ),
        },
        {
            "comparison": "acsr_retention_churn_controls",
            "packet_count": _len(acsr_retention.get("comparison_rows")),
            "status": acsr_retention.get("status"),
            "claim_status": acsr_retention.get("claim_status"),
            "causal_mechanism_claim": _as_dict(acsr_retention.get("claim_statuses")).get(
                "causal_mechanism_claim"
            ),
        },
    ]


def _gate_rows(
    closeout: dict[str, Any],
    dense_transfer: dict[str, Any],
    dense_seed_rows: list[dict[str, Any]],
    topk1: dict[str, Any],
    acsr_retention: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "acsr_closeout_passed",
            closeout.get("status") == "pass",
            "hard",
            "ACSR dense-control closeout exists and passed",
            closeout.get("status"),
            "ACSR closeout is missing or failed",
        ),
        _criterion(
            "support_discovery_frozen",
            closeout.get("claim_statuses", {}).get("deployable_support_discovery")
            == "frozen_negative_tiny_headroom_sequence_holdout_and_tiny_commutator",
            "hard",
            "deployable support discovery remains frozen negative",
            closeout.get("claim_statuses", {}).get("deployable_support_discovery"),
            "support discovery was not frozen negative",
        ),
        _criterion(
            "dense_transfer_blocks_sparse_separation",
            dense_transfer.get("claim_status") == "sparse_transfer_not_separated_from_dense_control",
            "claim_blocker",
            "dense transfer control blocks sparse transfer separation",
            dense_transfer.get("claim_status"),
            "dense transfer control no longer blocks sparse separation",
        ),
        _criterion(
            "dense_rank_norm_seeds_present",
            len(dense_seed_rows) >= 2 and all(row["status"] == "pass" for row in dense_seed_rows),
            "hard",
            "at least two dense rank/norm seed reports pass",
            [row.get("status") for row in dense_seed_rows],
            "dense rank/norm seed reports are missing or failed",
        ),
        _criterion(
            "dense_beats_sparse_topk2_all_seeds",
            all(
                _less(row.get("causal_dense_heldout_delta"), row.get("sparse_topk2_heldout_delta"))
                for row in dense_seed_rows
            ),
            "claim_blocker",
            "causal dense heldout CE delta beats sparse top-k2 in all local seeds",
            [
                {
                    "dense": row.get("causal_dense_heldout_delta"),
                    "sparse_topk2": row.get("sparse_topk2_heldout_delta"),
                }
                for row in dense_seed_rows
            ],
            "causal dense no longer beats sparse top-k2 in all seeds",
        ),
        _criterion(
            "active_topk1_retention_churn_stable",
            topk1.get("status") == "pass",
            "claim_blocker",
            "rank-matched top-k1 retention/churn stability report passes locally",
            topk1.get("status"),
            "rank-matched top-k1 retention/churn stability is missing or failed",
        ),
        _criterion(
            "acsr_retention_churn_not_promoted",
            acsr_retention.get("claim_status")
            == "stronger_non_ce_acsr_control_supported_not_promoted",
            "claim_blocker",
            "ACSR retention/churn controls are supported locally but not promoted",
            acsr_retention.get("claim_status"),
            "ACSR retention/churn claim status changed and needs separate handling",
        ),
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


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": bool(packet),
        "status": packet.get("status", "missing") if packet else "missing",
        "decision": packet.get("decision", "") if packet else "",
        "claim_status": packet.get("claim_status") or packet.get("claim_statuses", ""),
    }


def _arm(summary: dict[str, Any], arm: str) -> dict[str, Any]:
    for row in _as_list(summary.get("rank_norm_rows")) + _as_list(summary.get("arm_metrics")):
        if isinstance(row, dict) and row.get("arm") == arm:
            return row
    for row in _as_list(summary.get("interference_rows")):
        if isinstance(row, dict) and row.get("arm") == arm and row.get("split") == "heldout":
            return row
    return {}


def _paired(summary: dict[str, Any], arm: str, reference_arm: str) -> dict[str, Any]:
    for row in _as_list(summary.get("interference_rows")):
        if (
            isinstance(row, dict)
            and row.get("row_type") == "paired_arms"
            and row.get("split") == "heldout"
            and row.get("arm") == arm
            and row.get("reference_arm") == reference_arm
        ):
            return row
    return {}


def _strategy_review(path: Path) -> dict[str, Any]:
    header_keys = {
        "strategic_change_level",
        "notify_ben",
        "recommended_next_action",
        "verdict",
    }
    fields = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": "",
        "recommended_next_action": "",
        "verdict": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in header_keys:
            fields[key] = value.strip()
    fields["ben_notification_required"] = (
        fields["strategic_change_level"] == "major" or fields["notify_ben"] == "true"
    )
    return fields


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "dense_seed_rows.csv", summary["dense_seed_rows"])
    _write_csv(out_dir / "comparison_rows.csv", summary["comparison_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Residual Control Baseline Retention/Churn Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        "",
        "Dense controls remain the active local baseline. Rank-matched top-k1 has local retention/churn support, but ACSR support discovery remains frozen negative and no GPU validation target is selected.",
    ]
    if summary["claim_blockers_preserved"]:
        lines.extend(["", "## Preserved Claim Blockers"])
        for row in summary["claim_blockers_preserved"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _less(left: Any, right: Any) -> bool:
    left_num = _float_or_none(left)
    right_num = _float_or_none(right)
    return left_num is not None and right_num is not None and left_num < right_num


def _diff(left: Any, right: Any) -> float | None:
    left_num = _float_or_none(left)
    right_num = _float_or_none(right)
    if left_num is None or right_num is None:
        return None
    return left_num - right_num


def _mean(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def _len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--dense-transfer", type=Path, default=DEFAULT_DENSE_TRANSFER)
    parser.add_argument("--rank-norm-dir", type=Path, action="append", default=None)
    parser.add_argument("--topk1-stability", type=Path, default=DEFAULT_TOPK1_STABILITY)
    parser.add_argument("--acsr-retention-churn", type=Path, default=DEFAULT_ACSR_RETENTION_CHURN)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_residual_control_baseline_retention_churn_synthesis(
        closeout_path=args.closeout,
        dense_transfer_path=args.dense_transfer,
        rank_norm_dirs=tuple(args.rank_norm_dir) if args.rank_norm_dir else DEFAULT_RANK_NORM_DIRS,
        topk1_stability_path=args.topk1_stability,
        acsr_retention_churn_path=args.acsr_retention_churn,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
