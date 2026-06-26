"""Post-RunPod repeat decision for the causal contextual support router."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_TOKEN_SEQUENCE_REPORT = Path(
    "results/runpod_fetch/reports/"
    "runpod_token_larger_contextual_router_sequence_kfold_ablation/summary.json"
)
DEFAULT_CHAR_SEQUENCE_REPORT = Path(
    "results/runpod_fetch/reports/"
    "runpod_char_larger_seed2_contextual_router_sequence_kfold_ablation/summary.json"
)
DEFAULT_RUNPOD_GATE_REPORT = Path(
    "results/runpod_fetch/reports/runpod_token_larger_causal_contextual_router_gate/summary.json"
)
DEFAULT_FUTURE_PERTURBATION_REPORT = Path(
    "results/runpod_fetch/reports/runpod_causal_router_future_perturbation/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_runpod_repeat_decision"
)

RUNPOD_REPEAT_GATE_PASSED = "causal_contextual_router_runpod_repeat_gate_passed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
NEXT_SUPPORT_AUDIT = "causal_contextual_router_support_audit"

FULL_CONTEXT_VARIANT = "promoted_contextual_topk2:actual_full_context"
CAUSAL_VARIANT = "causal_contextual_topk2:actual_causal_context"
LINEAR_VARIANT = "linear_topk2_control:linear_actual"

MIN_FOLD_WIN_FRACTION = 0.75
FUTURE_CONTEXT_MATERIAL_DELTA = 0.01


def run_causal_contextual_router_runpod_repeat_decision(
    *,
    token_sequence_report_path: Path = DEFAULT_TOKEN_SEQUENCE_REPORT,
    char_sequence_report_path: Path = DEFAULT_CHAR_SEQUENCE_REPORT,
    runpod_gate_report_path: Path = DEFAULT_RUNPOD_GATE_REPORT,
    future_perturbation_report_path: Path = DEFAULT_FUTURE_PERTURBATION_REPORT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume fetched RunPod repeat artifacts and decide the next causal step."""

    start = time.time()
    token_sequence = _read_json_object(token_sequence_report_path)
    char_sequence = _read_json_object(char_sequence_report_path)
    runpod_gate = _read_json_object(runpod_gate_report_path)
    future_perturbation = _read_json_object(future_perturbation_report_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row("runpod_token_larger_sequence_kfold", token_sequence_report_path, token_sequence),
        _source_row("runpod_char_larger_seed2_sequence_kfold", char_sequence_report_path, char_sequence),
        _source_row("runpod_token_larger_causal_gate", runpod_gate_report_path, runpod_gate),
        _source_row(
            "runpod_causal_router_future_perturbation",
            future_perturbation_report_path,
            future_perturbation,
        ),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    repeat_rows = [
        _repeat_row("token_larger", token_sequence),
        _repeat_row("char_larger_seed2", char_sequence),
    ]
    gate_status = _gate_status(
        repeat_rows=repeat_rows,
        runpod_gate=runpod_gate,
        future_perturbation=future_perturbation,
    )
    failures = _failures(
        source_rows=source_rows,
        repeat_rows=repeat_rows,
        runpod_gate=runpod_gate,
        future_perturbation=future_perturbation,
        gate_status=gate_status,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_next_step = "repair_missing_or_failed_runpod_repeat_sources"
        rationale = (
            "The post-RunPod causal contextual-router repeat cannot be closed out "
            "because required fetched artifacts or gate fields are missing, failed, "
            "or inconsistent."
        )
    else:
        status = "pass"
        decision = RUNPOD_REPEAT_GATE_PASSED
        claim_status = "contextual_mlp_causal_runpod_repeat_supported_not_promoted"
        selected_next_step = NEXT_SUPPORT_AUDIT
        rationale = (
            "Fetched RunPod sequence-heldout repeats support contextual_mlp_causal "
            "as a causal-feature-safe candidate on token-larger and larger-char "
            "seed-2 settings, and the future-perturbation invariant is present. "
            "The full-context contextual_mlp path remains a nondeployable oracle "
            "diagnostic baseline. Default promotion remains blocked until causal "
            "support-audit evidence checks support quality, oracle regret, and churn."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": None,
        "claim_statuses": {
            "contextual_mlp": "nondeployable_full_context_oracle_diagnostic_only",
            "contextual_mlp_causal": claim_status,
            "promoted_default": "blocked_pending_causal_support_audit",
            "autoregressive_deployable_router": (
                "not_established_for_full_context_contextual_mlp"
            ),
        },
        "strategy_review": strategy_review,
        "repeat_gate_status": gate_status,
        "repeat_rows": repeat_rows,
        "source_rows": source_rows,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "repeat_metrics_csv": str(out_dir / "repeat_metrics.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_rows.csv",
        ["source", "path", "present", "status", "decision", "claim_status"],
        source_rows,
    )
    _write_csv(
        out_dir / "repeat_metrics.csv",
        [
            "dataset",
            "status",
            "decision",
            "fold_count",
            "causal_vs_linear_mean_delta",
            "causal_vs_linear_left_wins",
            "causal_vs_linear_fold_win_fraction",
            "causal_vs_linear_fold_deltas",
            "full_context_mean_ce",
            "causal_mean_ce",
            "linear_mean_ce",
            "causal_vs_full_mean_delta",
            "future_context_material_loss_delta",
            "causal_mean_used_columns",
            "linear_mean_used_columns",
            "causal_mean_unique_support_sets",
            "linear_mean_unique_support_sets",
        ],
        repeat_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        gate_status["criteria"],
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _repeat_row(dataset: str, packet: dict[str, Any]) -> dict[str, Any]:
    ablation = packet.get("ablation", {}) if isinstance(packet.get("ablation"), dict) else {}
    variants = ablation.get("variants", {}) if isinstance(ablation.get("variants"), dict) else {}
    comparisons = (
        ablation.get("key_comparisons", {})
        if isinstance(ablation.get("key_comparisons"), dict)
        else {}
    )
    causal_vs_linear = comparisons.get("causal_contextual_vs_linear", {})
    causal_vs_full = comparisons.get("causal_contextual_vs_full_context_oracle_baseline", {})
    causal = variants.get(CAUSAL_VARIANT, {})
    linear = variants.get(LINEAR_VARIANT, {})
    full = variants.get(FULL_CONTEXT_VARIANT, {})
    fold_count = _number(causal_vs_linear.get("fold_count"))
    left_wins = _number(causal_vs_linear.get("left_wins"))
    return {
        "dataset": dataset,
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
        "fold_count": ablation.get("fold_count"),
        "causal_vs_linear_mean_delta": ablation.get(
            "causal_contextual_vs_linear_loss_delta"
        ),
        "causal_vs_linear_left_wins": causal_vs_linear.get("left_wins"),
        "causal_vs_linear_right_wins": causal_vs_linear.get("right_wins"),
        "causal_vs_linear_fold_win_fraction": (
            left_wins / fold_count
            if fold_count is not None and fold_count > 0 and left_wins is not None
            else None
        ),
        "causal_vs_linear_fold_deltas": [
            row.get("loss_delta") for row in causal_vs_linear.get("fold_deltas", [])
        ],
        "full_context_mean_ce": full.get("mean_router_loss"),
        "full_context_uses_future_context": full.get("uses_future_context"),
        "causal_mean_ce": causal.get("mean_router_loss"),
        "causal_feature_safe": causal.get("causal_feature_safe"),
        "linear_mean_ce": linear.get("mean_router_loss"),
        "causal_vs_full_mean_delta": causal_vs_full.get("mean_loss_delta"),
        "future_context_material_loss_delta": ablation.get(
            "future_context_material_loss_delta"
        ),
        "causal_mean_used_columns": causal.get("mean_used_columns"),
        "linear_mean_used_columns": linear.get("mean_used_columns"),
        "causal_mean_unique_support_sets": causal.get("mean_unique_support_sets"),
        "linear_mean_unique_support_sets": linear.get("mean_unique_support_sets"),
    }


def _gate_status(
    *,
    repeat_rows: list[dict[str, Any]],
    runpod_gate: dict[str, Any],
    future_perturbation: dict[str, Any],
) -> dict[str, Any]:
    criteria = [
        _criterion(
            "runpod_local_gate_passed",
            runpod_gate.get("status") == "pass"
            and runpod_gate.get("decision")
            == "causal_contextual_router_local_gate_passed"
            and (
                runpod_gate.get("local_gate_status", {}).get("passes_full_local_gate")
                is True
            ),
            "fetched RunPod local gate has full local gate true",
            {
                "status": runpod_gate.get("status"),
                "decision": runpod_gate.get("decision"),
                "passes_full_local_gate": runpod_gate.get("local_gate_status", {}).get(
                    "passes_full_local_gate"
                ),
            },
        ),
        _criterion(
            "future_perturbation_invariance",
            future_perturbation.get("status") == "pass"
            and future_perturbation.get("future_perturbation_invariance") is True,
            "future perturbation summary passes and reports invariance true",
            {
                "status": future_perturbation.get("status"),
                "future_perturbation_invariance": future_perturbation.get(
                    "future_perturbation_invariance"
                ),
            },
        ),
    ]
    for row in repeat_rows:
        prefix = row["dataset"]
        criteria.extend(
            [
                _criterion(
                    f"{prefix}_sequence_candidate",
                    row["status"] == "ok"
                    and row["decision"]
                    == "causal_contextual_router_sequence_holdout_candidate",
                    "sequence K-fold report is an ok causal candidate",
                    {"status": row["status"], "decision": row["decision"]},
                ),
                _criterion(
                    f"{prefix}_causal_beats_linear_mean_ce",
                    _le(row["causal_vs_linear_mean_delta"], 0.0),
                    "causal contextual mean CE delta versus linear is <= 0",
                    row["causal_vs_linear_mean_delta"],
                ),
                _criterion(
                    f"{prefix}_fold_consistency",
                    _ge(row["causal_vs_linear_fold_win_fraction"], MIN_FOLD_WIN_FRACTION),
                    f"causal contextual wins at least {MIN_FOLD_WIN_FRACTION:.0%} of folds",
                    row["causal_vs_linear_fold_win_fraction"],
                ),
                _criterion(
                    f"{prefix}_future_context_material",
                    _gt(
                        row["future_context_material_loss_delta"],
                        FUTURE_CONTEXT_MATERIAL_DELTA,
                    ),
                    (
                        "full-context future-feature ablation delta is material, "
                        "preserving oracle-only classification"
                    ),
                    row["future_context_material_loss_delta"],
                ),
                _criterion(
                    f"{prefix}_causal_feature_safe",
                    row["causal_feature_safe"] is True,
                    "causal variant reports causal_feature_safe true",
                    row["causal_feature_safe"],
                ),
                _criterion(
                    f"{prefix}_full_context_uses_future_context",
                    row["full_context_uses_future_context"] is True,
                    "full-context oracle baseline records future-context usage",
                    row["full_context_uses_future_context"],
                ),
                _criterion(
                    f"{prefix}_support_utilization_not_collapsed",
                    _ge(row["causal_mean_used_columns"], row["linear_mean_used_columns"])
                    and _ge(
                        row["causal_mean_unique_support_sets"],
                        row["linear_mean_unique_support_sets"],
                    ),
                    "causal used columns and unique support sets are at least linear",
                    {
                        "causal_used_columns": row["causal_mean_used_columns"],
                        "linear_used_columns": row["linear_mean_used_columns"],
                        "causal_unique_support_sets": row[
                            "causal_mean_unique_support_sets"
                        ],
                        "linear_unique_support_sets": row[
                            "linear_mean_unique_support_sets"
                        ],
                    },
                ),
            ]
        )
    return {
        "criteria": criteria,
        "passes_runpod_repeat_gate": all(row["passed"] for row in criteria),
    }


def _criterion(
    criterion: str, passed: bool, threshold: str, actual: Any
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": json.dumps(actual, sort_keys=True),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    runpod_gate: dict[str, Any],
    future_perturbation: dict[str, Any],
    gate_status: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:4]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    for row in repeat_rows:
        if row["status"] != "ok":
            failures.append(
                {
                    "source": row["dataset"],
                    "field": "status",
                    "expected": "ok",
                    "actual": row["status"],
                }
            )
    if runpod_gate.get("status") != "pass":
        failures.append(
            {
                "source": "runpod_token_larger_causal_gate",
                "field": "status",
                "expected": "pass",
                "actual": runpod_gate.get("status"),
            }
        )
    if future_perturbation.get("status") != "pass":
        failures.append(
            {
                "source": "runpod_causal_router_future_perturbation",
                "field": "status",
                "expected": "pass",
                "actual": future_perturbation.get("status"),
            }
        )
    for criterion in gate_status["criteria"]:
        if not criterion["passed"]:
            failures.append(
                {
                    "source": "repeat_gate",
                    "field": criterion["criterion"],
                    "expected": criterion["threshold"],
                    "actual": criterion["actual"],
                }
            )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status") or packet.get("claim_policy"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {
            "strategic_change_level",
            "notify_ben",
            "recommended_next_action",
        }:
            header[key.strip()] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted: kept full-context contextual_mlp as a nondeployable oracle "
            "diagnostic, treated contextual_mlp_causal as the deployable candidate, "
            "and closed only the RunPod repeat gate while leaving default promotion "
            "blocked pending causal support-audit evidence"
        ),
        "ben_notification_required": bool(notify_ben) or major,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gt(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number > right_number


def _ge(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number >= right_number


def _le(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number <= right_number


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Causal Contextual Router RunPod Repeat Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
        "",
        "## Claim Statuses",
    ]
    for name, claim_status in summary["claim_statuses"].items():
        lines.append(f"- {name}: `{claim_status}`")
    lines.extend(["", "## Repeat Evidence"])
    for row in summary["repeat_rows"]:
        lines.extend(
            [
                f"- {row['dataset']} causal-vs-linear mean delta: "
                f"`{row['causal_vs_linear_mean_delta']}`",
                f"- {row['dataset']} fold deltas: "
                f"`{row['causal_vs_linear_fold_deltas']}`",
                f"- {row['dataset']} mean CE full/contextual/linear: "
                f"`{row['full_context_mean_ce']}` / `{row['causal_mean_ce']}` / "
                f"`{row['linear_mean_ce']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Repeat Gate",
            f"- Passes RunPod repeat gate: `{summary['repeat_gate_status']['passes_runpod_repeat_gate']}`",
        ]
    )
    for row in summary["repeat_gate_status"]["criteria"]:
        lines.append(f"- {row['criterion']}: `{row['passed']}`")
    lines.extend(["", "## Rationale", "", summary["rationale"], ""])
    if summary["failures"]:
        lines.extend(["## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-sequence-report", type=Path, default=DEFAULT_TOKEN_SEQUENCE_REPORT
    )
    parser.add_argument(
        "--char-sequence-report", type=Path, default=DEFAULT_CHAR_SEQUENCE_REPORT
    )
    parser.add_argument("--runpod-gate-report", type=Path, default=DEFAULT_RUNPOD_GATE_REPORT)
    parser.add_argument(
        "--future-perturbation-report",
        type=Path,
        default=DEFAULT_FUTURE_PERTURBATION_REPORT,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_runpod_repeat_decision(
        token_sequence_report_path=args.token_sequence_report,
        char_sequence_report_path=args.char_sequence_report,
        runpod_gate_report_path=args.runpod_gate_report,
        future_perturbation_report_path=args.future_perturbation_report,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "selected_next_step": summary["selected_next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
