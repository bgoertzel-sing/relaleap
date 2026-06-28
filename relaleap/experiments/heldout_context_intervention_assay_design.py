"""Design the heldout-context dense-vs-rank-matched top-k1 intervention assay."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SYNTHESIS = Path(
    "results/reports/dense_residual_control_baseline_retention_churn_synthesis/summary.json"
)
DEFAULT_TOPK1_STABILITY = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability/summary.json"
)
DEFAULT_TOPK1_METRICS = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability/"
    "probe_metrics.csv"
)
DEFAULT_RANK_NORM_DIRS = (
    Path("results/reports/dense_residual_rank_norm_interference_benchmark"),
    Path("results/reports/dense_residual_rank_norm_interference_benchmark_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/heldout_context_intervention_assay_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "assay_design.csv",
    "arm_comparison.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_heldout_context_intervention_assay_design(
    *,
    synthesis_path: Path = DEFAULT_SYNTHESIS,
    topk1_stability_path: Path = DEFAULT_TOPK1_STABILITY,
    topk1_metrics_path: Path = DEFAULT_TOPK1_METRICS,
    rank_norm_dirs: tuple[Path, ...] = DEFAULT_RANK_NORM_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a design-only local assay spec from completed dense/top-k1 evidence."""

    start = time.time()
    synthesis = _read_json(synthesis_path)
    topk1 = _read_json(topk1_stability_path)
    topk1_metrics = _read_csv(topk1_metrics_path)
    rank_norm_packets = [_load_rank_norm(path) for path in rank_norm_dirs]
    strategy = _strategy_review(strategy_review_path)

    source_rows = _source_rows(
        synthesis_path=synthesis_path,
        synthesis=synthesis,
        topk1_stability_path=topk1_stability_path,
        topk1=topk1,
        topk1_metrics_path=topk1_metrics_path,
        topk1_metrics=topk1_metrics,
        rank_norm_dirs=rank_norm_dirs,
        rank_norm_packets=rank_norm_packets,
        strategy_review_path=strategy_review_path,
        strategy=strategy,
    )
    arm_rows = _arm_comparison_rows(rank_norm_packets, topk1, topk1_metrics)
    design_rows = _assay_design_rows(synthesis, topk1, arm_rows)
    gate_rows = _gate_rows(synthesis, topk1, topk1_metrics, rank_norm_packets, strategy)
    failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    status = "pass" if not failures else "fail"
    selected_next_step = (
        "implement_local_heldout_context_intervention_probe_dense_vs_rank_matched_topk1"
        if status == "pass"
        else "repair_heldout_context_intervention_design_sources"
    )
    summary = {
        "status": status,
        "decision": (
            "heldout_context_intervention_assay_design_recorded"
            if status == "pass"
            else "heldout_context_intervention_assay_design_failed_closed"
        ),
        "claim_status": "design_only_dense_topk1_mechanism_not_retested",
        "selected_next_step": selected_next_step,
        "recommended_probe_contract": {
            "scope": "local CPU artifact/probe before any GPU validation",
            "split": "same heldout-context retention/churn split used by active top-k1 stability",
            "primary_arms": [
                "rank_flop_matched_causal_dense",
                "sparse_rank_matched_topk1",
            ],
            "required_nulls": [
                "token_position_dense",
                "sparse_contextual_topk2_reference",
                "frozen_base",
            ],
            "primary_metrics": [
                "heldout_context_ce_delta_vs_base",
                "anchor_support_churn_after_transfer",
                "anchor_logit_mse_drift",
                "transfer_ce_improvement",
                "residual_update_l2",
                "damage_fraction",
            ],
            "promotion_boundary": (
                "rank-matched top-k1 only reopens a sparse-mechanism claim if it beats "
                "causal dense on heldout-context CE/transfer while preserving its churn "
                "advantage under the same split and residual norm reporting"
            ),
        },
        "source_rows": source_rows,
        "assay_design": design_rows,
        "arm_comparison": arm_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_blockers_preserved": claim_blockers,
        "strategy_review": strategy,
        "strategy_review_handling": (
            "Accepted the latest GPT-5.5-Pro no-RunPod/null-complete boundary. "
            "Because the support-discovery gate remains frozen negative, this report "
            "keeps work local and designs the dense-vs-rank-matched top-k1 intervention."
        ),
        "claim_boundaries": {
            "supported": [
                "causal dense remains the active local baseline",
                "rank-matched top-k1 has local retention/churn support",
                "the next executable probe can be specified from existing artifact schemas",
            ],
            "not_supported": [
                "sparse support identity promotion",
                "deployable support discovery",
                "GPU validation target",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_rank_norm(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "summary": _read_json(path / "summary.json"),
        "rank_norm_rows": _read_csv(path / "rank_norm_rows.csv"),
        "interference_rows": _read_csv(path / "interference_rows.csv"),
    }


def _source_rows(
    *,
    synthesis_path: Path,
    synthesis: dict[str, Any],
    topk1_stability_path: Path,
    topk1: dict[str, Any],
    topk1_metrics_path: Path,
    topk1_metrics: list[dict[str, str]],
    rank_norm_dirs: tuple[Path, ...],
    rank_norm_packets: list[dict[str, Any]],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _source_row("dense_topk1_synthesis", synthesis_path, synthesis),
        _source_row("active_topk1_stability", topk1_stability_path, topk1),
        {
            "source": "active_topk1_probe_metrics",
            "path": str(topk1_metrics_path),
            "present": topk1_metrics_path.is_file(),
            "status": "present" if topk1_metrics else "missing",
            "decision": f"rows={len(topk1_metrics)}",
            "claim_status": "retention_churn_seed_metrics",
        },
    ]
    for index, (path, packet) in enumerate(zip(rank_norm_dirs, rank_norm_packets), start=1):
        rows.append(_source_row(f"dense_rank_norm_seed{index}", path / "summary.json", packet["summary"]))
    rows.append(
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        }
    )
    return rows


def _arm_comparison_rows(
    rank_norm_packets: list[dict[str, Any]],
    topk1: dict[str, Any],
    topk1_metrics: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, packet in enumerate(rank_norm_packets, start=1):
        dense = _arm(packet, "rank_flop_matched_causal_dense")
        sparse_topk1 = _arm(packet, "sparse_rank_matched_topk1")
        sparse_topk2 = _arm(packet, "sparse_contextual_topk2")
        token_dense = _arm(packet, "rank_flop_matched_token_position_dense")
        paired_dense_topk2 = _paired(packet, "rank_flop_matched_causal_dense", "sparse_contextual_topk2")
        rows.append(
            {
                "row_type": "rank_norm_seed",
                "seed": f"seed{index}",
                "source": packet["path"],
                "causal_dense_heldout_delta": _float_or_none(
                    dense.get("heldout_delta_vs_base_ce") or dense.get("mean_delta_vs_base_ce")
                ),
                "rank_matched_topk1_heldout_delta": _float_or_none(
                    sparse_topk1.get("heldout_delta_vs_base_ce")
                    or sparse_topk1.get("mean_delta_vs_base_ce")
                ),
                "sparse_topk2_heldout_delta": _float_or_none(
                    sparse_topk2.get("heldout_delta_vs_base_ce")
                    or sparse_topk2.get("mean_delta_vs_base_ce")
                ),
                "token_position_dense_heldout_delta": _float_or_none(
                    token_dense.get("heldout_delta_vs_base_ce")
                    or token_dense.get("mean_delta_vs_base_ce")
                ),
                "dense_minus_topk1_heldout_delta": _diff(
                    dense.get("heldout_delta_vs_base_ce") or dense.get("mean_delta_vs_base_ce"),
                    sparse_topk1.get("heldout_delta_vs_base_ce")
                    or sparse_topk1.get("mean_delta_vs_base_ce"),
                ),
                "dense_minus_topk2_heldout_delta": _diff(
                    dense.get("heldout_delta_vs_base_ce") or dense.get("mean_delta_vs_base_ce"),
                    sparse_topk2.get("heldout_delta_vs_base_ce")
                    or sparse_topk2.get("mean_delta_vs_base_ce"),
                ),
                "dense_vs_topk2_left_wins_fraction": _float_or_none(
                    paired_dense_topk2.get("left_wins_fraction")
                ),
            }
        )
    aggregates = _as_dict(topk1.get("aggregates"))
    rows.append(
        {
            "row_type": "topk1_retention_churn_aggregate",
            "seed": "aggregate",
            "source": "topk1_stability",
            "topk1_support_churn": aggregates.get("mean_topk1_support_churn"),
            "topk2_support_churn": aggregates.get("mean_topk2_support_churn"),
            "support_churn_advantage": aggregates.get("mean_support_churn_advantage"),
            "logit_churn_advantage": aggregates.get("mean_logit_churn_advantage"),
            "transfer_improvement_advantage": aggregates.get("mean_transfer_improvement_advantage"),
        }
    )
    for row in topk1_metrics:
        rows.append(
            {
                "row_type": "topk1_retention_churn_seed",
                "seed": row.get("packet", ""),
                "source": row.get("probe_dir", ""),
                "topk1_anchor_support_churn_after_transfer": row.get(
                    "topk1_anchor_support_churn_after_transfer", ""
                ),
                "topk2_anchor_support_churn_after_transfer": row.get(
                    "topk2_anchor_support_churn_after_transfer", ""
                ),
                "topk1_anchor_logit_mse_drift": row.get("topk1_anchor_logit_mse_drift", ""),
                "topk2_anchor_logit_mse_drift": row.get("topk2_anchor_logit_mse_drift", ""),
                "topk1_transfer_ce_improvement": row.get("topk1_transfer_ce_improvement", ""),
                "topk2_transfer_ce_improvement": row.get("topk2_transfer_ce_improvement", ""),
                "dense_transfer_ce_improvement": row.get("dense_transfer_ce_improvement", ""),
            }
        )
    return rows


def _assay_design_rows(
    synthesis: dict[str, Any],
    topk1: dict[str, Any],
    arm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = _as_dict(topk1.get("aggregates"))
    return [
        {
            "component": "data_split",
            "requirement": "reuse heldout-context retention/churn split",
            "rationale": (
                "sequence-heldout/context-heldout evidence is the relevant deconfounder; "
                "flattened token splits can leak token-position regularities"
            ),
            "source_metric": f"topk1 packets={topk1.get('packet_count', '')}",
            "fail_closed_if_missing": True,
        },
        {
            "component": "primary_dense_arm",
            "requirement": "rank_flop_matched_causal_dense",
            "rationale": "dense controls currently beat sparse top-k2 on heldout CE delta",
            "source_metric": f"mean_dense_minus_sparse_topk2={_mean_dense_minus(arm_rows, 'sparse_topk2')}",
            "fail_closed_if_missing": True,
        },
        {
            "component": "primary_sparse_arm",
            "requirement": "sparse_rank_matched_topk1",
            "rationale": "active sparse candidate with lower support/logit churn and transfer improvement",
            "source_metric": (
                f"support_churn_advantage={aggregates.get('mean_support_churn_advantage')}; "
                f"transfer_improvement_advantage={aggregates.get('mean_transfer_improvement_advantage')}"
            ),
            "fail_closed_if_missing": True,
        },
        {
            "component": "nulls",
            "requirement": "token-position dense, sparse top-k2 reference, frozen base",
            "rationale": "separates causal dense gains from token/position priors and prior sparse baseline",
            "source_metric": synthesis.get("claim_status", ""),
            "fail_closed_if_missing": True,
        },
        {
            "component": "gate",
            "requirement": (
                "top-k1 must beat causal dense on heldout-context transfer/CE while preserving "
                "support-churn and logit-churn advantages"
            ),
            "rationale": "retention/churn alone cannot override dense residual heldout CE baseline",
            "source_metric": synthesis.get("selected_next_step", ""),
            "fail_closed_if_missing": True,
        },
    ]


def _gate_rows(
    synthesis: dict[str, Any],
    topk1: dict[str, Any],
    topk1_metrics: list[dict[str, str]],
    rank_norm_packets: list[dict[str, Any]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "strategy_review_consumed",
            strategy["present"],
            "hard",
            "latest GPT-5.5-Pro review is present and read",
            strategy["recommended_next_action"],
            "strategy review missing",
        ),
        _criterion(
            "synthesis_selected_this_step",
            synthesis.get("status") == "pass"
            and synthesis.get("selected_next_step")
            == "design_heldout_context_intervention_assay_dense_vs_rank_matched_topk1",
            "hard",
            "previous synthesis selected heldout-context dense-vs-top-k1 design",
            synthesis.get("selected_next_step"),
            "previous synthesis did not select this design step",
        ),
        _criterion(
            "dense_control_active_baseline",
            _as_dict(synthesis.get("claim_statuses")).get("dense_residual_controls")
            == "active_baseline",
            "hard",
            "dense residual controls are the active baseline",
            _as_dict(synthesis.get("claim_statuses")).get("dense_residual_controls"),
            "dense controls are not marked active",
        ),
        _criterion(
            "topk1_retention_churn_stable",
            topk1.get("status") == "pass" and bool(topk1_metrics),
            "hard",
            "rank-matched top-k1 stability and seed metrics are present",
            {"status": topk1.get("status"), "metric_rows": len(topk1_metrics)},
            "rank-matched top-k1 stability sources are missing or failed",
        ),
        _criterion(
            "dense_rank_norm_sources_pass",
            len(rank_norm_packets) >= 2
            and all(_as_dict(packet.get("summary")).get("status") == "pass" for packet in rank_norm_packets),
            "hard",
            "at least two dense rank/norm sources pass",
            [_as_dict(packet.get("summary")).get("status") for packet in rank_norm_packets],
            "dense rank/norm sources missing or failed",
        ),
        _criterion(
            "causal_dense_beats_rank_matched_topk1_heldout",
            all(
                _less(_arm(packet, "rank_flop_matched_causal_dense").get("heldout_delta_vs_base_ce"),
                      _arm(packet, "sparse_rank_matched_topk1").get("heldout_delta_vs_base_ce"))
                for packet in rank_norm_packets
            ),
            "claim_blocker",
            "current sources show dense beats top-k1 on heldout CE, so probe must overturn this directly",
            [
                {
                    "dense": _arm(packet, "rank_flop_matched_causal_dense").get(
                        "heldout_delta_vs_base_ce"
                    ),
                    "topk1": _arm(packet, "sparse_rank_matched_topk1").get(
                        "heldout_delta_vs_base_ce"
                    ),
                }
                for packet in rank_norm_packets
            ],
            "dense no longer beats top-k1 in all rank/norm sources; interpretation needs update",
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
        "present": path.is_file(),
        "status": packet.get("status", "missing") if packet else "missing",
        "decision": packet.get("decision", "") if packet else "",
        "claim_status": packet.get("claim_status") or packet.get("claim_statuses", ""),
    }


def _arm(packet: dict[str, Any], arm: str) -> dict[str, Any]:
    for row in _as_list(_as_dict(packet.get("summary")).get("rank_norm_rows")) + _as_list(
        packet.get("rank_norm_rows")
    ):
        if isinstance(row, dict) and row.get("arm") == arm:
            return row
    for row in _as_list(_as_dict(packet.get("summary")).get("interference_rows")) + _as_list(
        packet.get("interference_rows")
    ):
        if isinstance(row, dict) and row.get("arm") == arm and row.get("split") == "heldout":
            return row
    return {}


def _paired(packet: dict[str, Any], arm: str, reference_arm: str) -> dict[str, Any]:
    for row in _as_list(_as_dict(packet.get("summary")).get("interference_rows")) + _as_list(
        packet.get("interference_rows")
    ):
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
        if key.strip() in fields:
            fields[key.strip()] = value.strip()
    fields["ben_notification_required"] = (
        fields["strategic_change_level"] == "major" or fields["notify_ben"] == "true"
    )
    return fields


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "assay_design.csv", summary["assay_design"])
    _write_csv(out_dir / "arm_comparison.csv", summary["arm_comparison"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Heldout-Context Intervention Assay Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        "",
        "This is a design-only local artifact. It keeps dense residual controls as the active baseline and specifies the next probe needed to test rank-matched top-k1 under the same heldout-context retention/churn split.",
    ]
    if summary["claim_blockers_preserved"]:
        lines.extend(["", "## Preserved Claim Blockers"])
        for row in summary["claim_blockers_preserved"]:
            lines.append(f"- `{row['criterion']}`: {row['requirement']}")
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


def _mean_dense_minus(rows: list[dict[str, Any]], reference: str) -> float | None:
    key_reference = {"sparse_topk2": "topk2"}.get(reference, reference)
    key = f"dense_minus_{key_reference}_heldout_delta"
    values = [_float_or_none(row.get(key)) for row in rows if row.get("row_type") == "rank_norm_seed"]
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--topk1-stability", type=Path, default=DEFAULT_TOPK1_STABILITY)
    parser.add_argument("--topk1-metrics", type=Path, default=DEFAULT_TOPK1_METRICS)
    parser.add_argument("--rank-norm-dir", type=Path, action="append", default=None)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_heldout_context_intervention_assay_design(
        synthesis_path=args.synthesis,
        topk1_stability_path=args.topk1_stability,
        topk1_metrics_path=args.topk1_metrics,
        rank_norm_dirs=tuple(args.rank_norm_dir) if args.rank_norm_dir else DEFAULT_RANK_NORM_DIRS,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
