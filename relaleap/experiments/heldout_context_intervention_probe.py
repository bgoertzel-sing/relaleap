"""Probe dense-vs-rank-matched top-k1 heldout-context intervention evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DESIGN = Path("results/reports/heldout_context_intervention_assay_design/summary.json")
DEFAULT_TOPK1_METRICS = Path(
    "results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability/"
    "probe_metrics.csv"
)
DEFAULT_RANK_NORM_DIRS = (
    Path("results/reports/dense_residual_rank_norm_interference_benchmark"),
    Path("results/reports/dense_residual_rank_norm_interference_benchmark_seed2"),
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/heldout_context_intervention_probe")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "arm_metrics.csv",
    "paired_deltas.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_ARMS = (
    "frozen_base",
    "rank_flop_matched_causal_dense",
    "rank_flop_matched_token_position_dense",
    "rank_flop_matched_shuffled_context_dense",
    "rank_flop_matched_ablated_context_dense",
    "sparse_rank_matched_topk1",
    "sparse_frequency_matched_random_topk1",
    "sparse_contextual_topk2",
)

PRIMARY_DENSE = "rank_flop_matched_causal_dense"
PRIMARY_SPARSE = "sparse_rank_matched_topk1"


def run_heldout_context_intervention_probe(
    *,
    design_path: Path = DEFAULT_DESIGN,
    topk1_metrics_path: Path = DEFAULT_TOPK1_METRICS,
    rank_norm_dirs: tuple[Path, ...] = DEFAULT_RANK_NORM_DIRS,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local probe over heldout-context source packets."""

    start = time.time()
    design = _read_json(design_path)
    topk1_metrics = _read_csv(topk1_metrics_path)
    packets = [_load_packet(path) for path in rank_norm_dirs]
    strategy = _strategy_review(strategy_review_path)
    source_rows = _source_rows(
        design_path, design, topk1_metrics_path, topk1_metrics, packets, strategy_review_path, strategy
    )
    arm_rows = _arm_rows(packets, topk1_metrics)
    paired_rows = _paired_rows(arm_rows)
    gate_rows = _gate_rows(design, packets, arm_rows, paired_rows, topk1_metrics, strategy)
    failures = [row for row in gate_rows if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [row for row in gate_rows if not row["passed"] and row["severity"] != "hard"]
    status = "pass" if not failures else "fail"
    dense_minus_topk1 = _mean(
        _float(row.get("dense_minus_topk1_heldout_delta"))
        for row in paired_rows
        if row.get("comparison") == "primary_dense_minus_sparse"
    )
    summary = {
        "status": status,
        "decision": (
            "heldout_context_intervention_probe_passed"
            if status == "pass"
            else "heldout_context_intervention_probe_failed_closed"
        ),
        "claim_status": (
            "rank_matched_topk1_reopened_sparse_mechanism_claim"
            if status == "pass" and dense_minus_topk1 is not None and dense_minus_topk1 > 0.0
            else "rank_matched_topk1_remains_diagnostic_dense_baseline_active"
        ),
        "selected_next_step": (
            "run_gpu_validation_only_after_local_probe_passes_with_required_nulls"
            if status == "pass"
            else "add_shuffled_ablated_context_and_random_support_nulls_to_source_probe"
        ),
        "primary_result": {
            "mean_dense_minus_topk1_heldout_delta": dense_minus_topk1,
            "interpretation": (
                "negative means causal dense has lower heldout CE delta than rank-matched top-k1"
            ),
        },
        "source_rows": source_rows,
        "arm_metrics": arm_rows,
        "paired_deltas": paired_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_blockers_preserved": claim_blockers,
        "strategy_review": strategy,
        "strategy_review_handling": (
            "Accepted the GPT-5.5-Pro recommendation to require shuffled-context, "
            "ablated-context, random-support, residual-norm, and active-compute accounting "
            "before treating the local dense-vs-top-k1 probe as scientific evidence."
        ),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _load_packet(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "summary": _read_json(path / "summary.json"),
        "rank_norm_rows": _read_csv(path / "rank_norm_rows.csv"),
        "interference_rows": _read_csv(path / "interference_rows.csv"),
    }


def _source_rows(
    design_path: Path,
    design: dict[str, Any],
    topk1_metrics_path: Path,
    topk1_metrics: list[dict[str, str]],
    packets: list[dict[str, Any]],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "source": "heldout_context_intervention_assay_design",
            "path": str(design_path),
            "present": design_path.is_file(),
            "status": design.get("status", "missing"),
            "decision": design.get("decision", ""),
        },
        {
            "source": "topk1_retention_churn_metrics",
            "path": str(topk1_metrics_path),
            "present": topk1_metrics_path.is_file(),
            "status": "present" if topk1_metrics else "missing",
            "decision": f"rows={len(topk1_metrics)}",
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
        },
    ]
    for packet in packets:
        rows.append(
            {
                "source": "rank_norm_packet",
                "path": packet["path"],
                "present": Path(packet["path"]).is_dir(),
                "status": _as_dict(packet.get("summary")).get("status", "missing"),
                "decision": _as_dict(packet.get("summary")).get("decision", ""),
            }
        )
    return rows


def _arm_rows(
    packets: list[dict[str, Any]],
    topk1_metrics: list[dict[str, str]],
) -> list[dict[str, Any]]:
    topk1_by_seed = {row.get("packet", ""): row for row in topk1_metrics}
    rows: list[dict[str, Any]] = []
    for index, packet in enumerate(packets, start=1):
        seed = f"seed{index}"
        by_arm = {row.get("arm", ""): row for row in packet["rank_norm_rows"]}
        split_rows = {
            row.get("arm", ""): row
            for row in packet["interference_rows"]
            if row.get("row_type") == "arm_split" and row.get("split") == "heldout"
        }
        for arm in REQUIRED_ARMS:
            rank_row = by_arm.get(arm, {})
            split_row = split_rows.get(arm, {})
            retention = topk1_by_seed.get(seed, {}) if arm == PRIMARY_SPARSE else {}
            rows.append(
                {
                    "seed": seed,
                    "source": packet["path"],
                    "arm": arm,
                    "present": bool(rank_row) or arm == "frozen_base",
                    "family": rank_row.get("family") or ("base" if arm == "frozen_base" else ""),
                    "heldout_delta_vs_base_ce": 0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_delta_vs_base_ce")),
                    "heldout_residual_update_l2": 0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_residual_update_l2")),
                    "active_params_proxy": 0.0 if arm == "frozen_base" else _float(rank_row.get("active_params_proxy")),
                    "flops_proxy": 0.0 if arm == "frozen_base" else _float(rank_row.get("flops_proxy")),
                    "damage_fraction": 0.0 if arm == "frozen_base" else _float(split_row.get("damage_fraction")),
                    "improvement_fraction": 0.0 if arm == "frozen_base" else _float(split_row.get("improvement_fraction")),
                    "ce_gain_per_l2": _safe_divide(
                        0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_delta_vs_base_ce")),
                        0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_residual_update_l2")),
                    ),
                    "ce_gain_per_active_param": _safe_divide(
                        0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_delta_vs_base_ce")),
                        0.0 if arm == "frozen_base" else _float(rank_row.get("active_params_proxy")),
                    ),
                    "ce_gain_per_flop_proxy": _safe_divide(
                        0.0 if arm == "frozen_base" else _float(rank_row.get("heldout_delta_vs_base_ce")),
                        0.0 if arm == "frozen_base" else _float(rank_row.get("flops_proxy")),
                    ),
                    "topk1_anchor_support_churn_after_transfer": retention.get("topk1_anchor_support_churn_after_transfer", ""),
                    "topk1_anchor_logit_mse_drift": retention.get("topk1_anchor_logit_mse_drift", ""),
                    "topk1_transfer_ce_improvement": retention.get("topk1_transfer_ce_improvement", ""),
                }
            )
    return rows


def _paired_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_seed_arm = {(row["seed"], row["arm"]): row for row in arm_rows}
    seeds = sorted({row["seed"] for row in arm_rows})
    comparisons = (
        ("primary_dense_minus_sparse", PRIMARY_DENSE, PRIMARY_SPARSE),
        ("causal_dense_minus_token_position_null", PRIMARY_DENSE, "rank_flop_matched_token_position_dense"),
        ("causal_dense_minus_shuffled_context_null", PRIMARY_DENSE, "rank_flop_matched_shuffled_context_dense"),
        ("causal_dense_minus_ablated_context_null", PRIMARY_DENSE, "rank_flop_matched_ablated_context_dense"),
        ("topk1_minus_random_support_null", PRIMARY_SPARSE, "sparse_frequency_matched_random_topk1"),
        ("topk1_minus_topk2_reference", PRIMARY_SPARSE, "sparse_contextual_topk2"),
    )
    for seed in seeds:
        for comparison, left_arm, right_arm in comparisons:
            left = by_seed_arm.get((seed, left_arm), {})
            right = by_seed_arm.get((seed, right_arm), {})
            left_delta = _float(left.get("heldout_delta_vs_base_ce"))
            right_delta = _float(right.get("heldout_delta_vs_base_ce"))
            rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "left_present": bool(left.get("present")),
                    "right_present": bool(right.get("present")),
                    "dense_minus_topk1_heldout_delta": (
                        left_delta - right_delta
                        if comparison == "primary_dense_minus_sparse"
                        and left_delta is not None
                        and right_delta is not None
                        else ""
                    ),
                    "left_minus_right_heldout_delta": (
                        left_delta - right_delta
                        if left_delta is not None and right_delta is not None
                        else ""
                    ),
                }
            )
    return rows


def _gate_rows(
    design: dict[str, Any],
    packets: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    topk1_metrics: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    present_by_seed = {}
    for row in arm_rows:
        present_by_seed.setdefault(row["seed"], set())
        if row["present"]:
            present_by_seed[row["seed"]].add(row["arm"])
    missing_by_seed = {
        seed: sorted(set(REQUIRED_ARMS) - arms)
        for seed, arms in present_by_seed.items()
        if set(REQUIRED_ARMS) - arms
    }
    norm_compute_missing = [
        {"seed": row["seed"], "arm": row["arm"]}
        for row in arm_rows
        if row["arm"] != "frozen_base"
        and row["present"]
        and (
            _float(row.get("heldout_residual_update_l2")) is None
            or _float(row.get("active_params_proxy")) is None
            or _float(row.get("flops_proxy")) is None
        )
    ]
    primary_deltas = [
        _float(row.get("dense_minus_topk1_heldout_delta"))
        for row in paired_rows
        if row.get("comparison") == "primary_dense_minus_sparse"
    ]
    primary_deltas = [value for value in primary_deltas if value is not None]
    topk1_retention_pass = all(
        _bool(row.get("topk1_support_churn_lower_than_topk2"))
        and _bool(row.get("topk1_logit_churn_not_higher_than_topk2"))
        and _bool(row.get("topk1_transfer_improvement_at_least_topk2"))
        for row in topk1_metrics
    )
    return [
        _criterion(
            "strategy_review_consumed",
            strategy["present"],
            "hard",
            "latest strategy review is present and read",
            strategy["recommended_next_action"],
            "strategy review missing",
        ),
        _criterion(
            "design_selected_probe",
            design.get("status") == "pass"
            and design.get("selected_next_step")
            == "implement_local_heldout_context_intervention_probe_dense_vs_rank_matched_topk1",
            "hard",
            "heldout-context assay design selected this probe",
            design.get("selected_next_step"),
            "design report did not select this probe",
        ),
        _criterion(
            "rank_norm_packets_pass",
            bool(packets)
            and all(_as_dict(packet.get("summary")).get("status") == "pass" for packet in packets),
            "hard",
            "all source rank/norm packets pass",
            [_as_dict(packet.get("summary")).get("status") for packet in packets],
            "one or more source rank/norm packets are missing or failed",
        ),
        _criterion(
            "required_arms_and_nulls_present",
            not missing_by_seed,
            "hard",
            "all required arms/nulls are present, including shuffled/ablated context and random support",
            missing_by_seed,
            "required heldout-context null/control arms are missing",
        ),
        _criterion(
            "residual_norm_and_active_compute_accounting_present",
            not norm_compute_missing,
            "hard",
            "residual L2, active-parameter, and FLOP proxy fields exist for non-base arms",
            norm_compute_missing,
            "norm or compute accounting is missing",
        ),
        _criterion(
            "topk1_retention_churn_guardrail_passes",
            bool(topk1_metrics) and topk1_retention_pass,
            "claim_blocker",
            "rank-matched top-k1 preserves support/logit churn and transfer guardrails",
            {"rows": len(topk1_metrics), "passed": topk1_retention_pass},
            "top-k1 retention/churn guardrails do not pass",
        ),
        _criterion(
            "topk1_beats_causal_dense_on_heldout_ce",
            bool(primary_deltas) and all(value > 0.0 for value in primary_deltas),
            "claim_blocker",
            "top-k1 must beat causal dense on heldout CE delta before sparse-mechanism promotion",
            primary_deltas,
            "causal dense still beats or ties rank-matched top-k1 on heldout CE",
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
    _write_csv(out_dir / "arm_metrics.csv", summary["arm_metrics"])
    _write_csv(out_dir / "paired_deltas.csv", summary["paired_deltas"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Heldout-Context Intervention Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        "",
        "This command consumes local source packets and fails closed unless all required nulls and residual-norm/compute fields are present.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        for row in summary["failures"]:
            lines.append(f"- `{row['criterion']}`: {row['failure_reason']}")
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
    return str(value).strip().lower() == "true"


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _mean(values: Any) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return sum(real) / len(real)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--topk1-metrics", type=Path, default=DEFAULT_TOPK1_METRICS)
    parser.add_argument("--rank-norm-dir", type=Path, action="append", default=None)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_heldout_context_intervention_probe(
        design_path=args.design,
        topk1_metrics_path=args.topk1_metrics,
        rank_norm_dirs=tuple(args.rank_norm_dir) if args.rank_norm_dir else DEFAULT_RANK_NORM_DIRS,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
