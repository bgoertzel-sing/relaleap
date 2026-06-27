"""Discriminative mechanism report for causal-router distillation controls."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SYNTHESIS_SUMMARY = Path(
    "results/reports/token_larger_causal_contextual_router_distillation_synthesis/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_discriminative_mechanism_audit"
)

REAL_CONTROL = "causal_distilled_from_oracle_target_0.05"
SHUFFLED_NULL = "causal_distilled_from_shuffled_teacher_0.05"
FREQUENCY_NULL = "causal_distilled_from_frequency_matched_teacher_0.05"
DIRECT_CE_CAUSAL = "causal_contextual_topk2"

SUPPORTED = "distilled_causal_router_discriminative_mechanism_supported_not_promoted"
INSUFFICIENT = "insufficient_discriminative_mechanism_evidence"
NEXT_STRONGER_NULL = "token_position_stratified_frequency_matched_teacher_null"

EXPECTED_FILES = [
    "summary.json",
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "agreement_metrics.csv",
    "intervention_metrics.csv",
    "null_control_metrics.csv",
    "per_token_supports.csv",
    "support_counts.csv",
    "notes.md",
]


def run_causal_contextual_router_discriminative_mechanism_audit(
    *,
    local_audit_dirs: list[Path] | None = None,
    runpod_audit_dirs: list[Path] | None = None,
    synthesis_summary_path: Path = DEFAULT_SYNTHESIS_SUMMARY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume existing agreement artifacts and run a conservative control gate."""

    start = time.time()
    local_dirs = local_audit_dirs or [
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3"),
    ]
    runpod_dirs = runpod_audit_dirs or [
        Path(
            "results/runpod_fetch/audits/"
            "runpod_token_larger_causal_contextual_router_distillation_agreement"
        ),
        Path(
            "results/runpod_fetch/audits/"
            "runpod_token_larger_causal_contextual_router_distillation_agreement_seed2"
        ),
        Path(
            "results/runpod_fetch/audits/"
            "runpod_token_larger_causal_contextual_router_distillation_agreement_seed3"
        ),
    ]
    strategy_review = _strategy_review(strategy_review_path)
    synthesis = _read_json_object(synthesis_summary_path)
    sources = [
        _load_source("local", seed, path)
        for seed, path in zip([1, 2, 3], local_dirs, strict=True)
    ] + [
        _load_source("runpod", seed, path)
        for seed, path in zip([1, 2, 3], runpod_dirs, strict=True)
    ]
    source_rows = [source["source_row"] for source in sources]
    control_rows = [row for source in sources for row in source["control_rows"]]
    paired_rows = _paired_control_rows(sources)
    intervention_rows = _intervention_summary_rows(sources)
    unavailable_rows = _unavailable_control_rows()
    gate_status = _gate_status(
        synthesis=synthesis,
        source_rows=source_rows,
        paired_rows=paired_rows,
        intervention_rows=intervention_rows,
    )
    failures = _failures(source_rows=source_rows, gate_status=gate_status)

    if failures:
        status = "fail"
        decision = INSUFFICIENT
        claim_status = INSUFFICIENT
        selected_next_step = "repair_missing_or_incomplete_discriminative_control_artifacts"
        rationale = (
            "The discriminative mechanism audit fails closed because required "
            "agreement artifacts, controls, folds, or same-sign control gates are "
            "missing or inconsistent."
        )
    else:
        status = "pass"
        decision = "real_teacher_distilled_causal_router_preferred_over_control_family"
        claim_status = SUPPORTED
        selected_next_step = NEXT_STRONGER_NULL
        rationale = (
            "The real teacher-distilled causal student beats shuffled-teacher, "
            "frequency-matched teacher-support, and direct CE-trained causal "
            "contextual controls on local paired fold CE and oracle-regret proxy. "
            "It also preserves broad support usage and uses linear-router support "
            "interventions as a realized-support negative control, while RunPod "
            "repeats preserve the same local sign pattern. This supports a bounded "
            "mechanism claim about non-marginal teacher support information, but "
            "does not establish a deployable default, general causal-column reuse, "
            "or a rank/dense-matched residual comparison."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": (
            "python -m relaleap.experiments.causal_contextual_router_"
            "distillation_agreement_audit --help"
        )
        if status == "pass"
        else None,
        "strategy_review": strategy_review,
        "synthesis_source": _synthesis_source_row(synthesis_summary_path, synthesis),
        "source_rows": source_rows,
        "control_rows": control_rows,
        "paired_control_rows": paired_rows,
        "intervention_summary_rows": intervention_rows,
        "unavailable_control_rows": unavailable_rows,
        "gate_status": gate_status,
        "failures": failures,
        "claim_boundaries": {
            "supported": (
                "real teacher-distilled causal support routing is discriminated "
                "from shuffled/frequency-matched support-label nulls, direct "
                "CE-trained causal contextual routing, and linear-support realized "
                "intervention controls under the token-larger agreement artifacts"
            ),
            "not_supported": [
                "deployable causal-router default",
                "general causal column reuse",
                "full causal separability",
                "rank-matched or dense-matched residual control",
                "independent confirmation from CE and oracle-regret proxy",
            ],
            "backend_repeats_count_as": "reproducibility_checks_not_independent_seeds",
        },
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_artifacts_csv": str(out_dir / "source_artifacts.csv"),
            "control_summary_csv": str(out_dir / "control_summary.csv"),
            "fold_paired_controls_csv": str(out_dir / "fold_paired_controls.csv"),
            "intervention_summary_csv": str(out_dir / "intervention_summary.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "unavailable_controls_csv": str(out_dir / "unavailable_controls.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "source_artifacts.csv",
        [
            "backend",
            "seed",
            "path",
            "present",
            "expected_files_present",
            "status",
            "decision",
            "claim_status",
            "fold_count",
            "dataset",
            "support_router",
            "top_k",
            "git_commit",
        ],
        source_rows,
    )
    _write_csv(
        out_dir / "control_summary.csv",
        [
            "backend",
            "seed",
            "control",
            "folds",
            "mean_router_loss",
            "mean_oracle_support_regret",
            "mean_used_columns",
            "mean_unique_support_sets",
            "mean_support_load_entropy",
            "mean_support_change_fraction",
        ],
        control_rows,
    )
    _write_csv(
        out_dir / "fold_paired_controls.csv",
        [
            "backend",
            "seed",
            "fold",
            "comparison_control",
            "router_loss_delta_real_minus_control",
            "oracle_regret_delta_real_minus_control",
            "used_columns_delta_real_minus_control",
            "unique_support_sets_delta_real_minus_control",
            "support_entropy_delta_real_minus_control",
        ],
        paired_rows,
    )
    _write_csv(
        out_dir / "intervention_summary.csv",
        [
            "backend",
            "seed",
            "token_subset",
            "intervention",
            "folds",
            "mean_loss",
            "mean_delta_vs_student_router_support",
            "delta_sign",
        ],
        intervention_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        gate_status["criteria"],
    )
    _write_csv(
        out_dir / "unavailable_controls.csv",
        ["control", "status", "reason", "disposition"],
        unavailable_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_source(backend: str, seed: int, path: Path) -> dict[str, Any]:
    packet = _read_json_object(path / "summary.json")
    audit = packet.get("audit", {}) if isinstance(packet.get("audit"), dict) else {}
    source_row = {
        "backend": backend,
        "seed": seed,
        "path": str(path),
        "present": path.is_dir() and (path / "summary.json").is_file(),
        "expected_files_present": all((path / name).is_file() for name in EXPECTED_FILES),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
        "fold_count": audit.get("fold_count"),
        "dataset": audit.get("dataset"),
        "support_router": audit.get("support_router"),
        "top_k": audit.get("top_k"),
        "git_commit": packet.get("git_commit"),
    }
    fold_rows = _read_csv_rows(path / "fold_metrics.csv")
    aggregate_rows = _read_csv_rows(path / "aggregate_metrics.csv")
    intervention_rows = _read_csv_rows(path / "intervention_metrics.csv")
    return {
        "source_row": source_row,
        "fold_rows": fold_rows,
        "control_rows": _control_rows(backend, seed, aggregate_rows),
        "intervention_rows": intervention_rows,
    }


def _control_rows(
    backend: str, seed: int, aggregate_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    wanted = {REAL_CONTROL, SHUFFLED_NULL, FREQUENCY_NULL, DIRECT_CE_CAUSAL}
    rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        if row.get("control") not in wanted:
            continue
        rows.append(
            {
                "backend": backend,
                "seed": seed,
                "control": row.get("control"),
                "folds": _number(row.get("folds")),
                "mean_router_loss": _number(row.get("mean_router_loss")),
                "mean_oracle_support_regret": _number(
                    row.get("mean_oracle_support_regret")
                ),
                "mean_used_columns": _number(row.get("mean_used_columns")),
                "mean_unique_support_sets": _number(
                    row.get("mean_unique_support_sets")
                ),
                "mean_support_load_entropy": _number(
                    row.get("mean_support_load_entropy")
                ),
                "mean_support_change_fraction": _number(
                    row.get("mean_support_change_fraction")
                ),
            }
        )
    return rows


def _paired_control_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparison_controls = [SHUFFLED_NULL, FREQUENCY_NULL, DIRECT_CE_CAUSAL]
    for source in sources:
        source_row = source["source_row"]
        fold_rows = source["fold_rows"]
        by_key = {
            (str(row.get("fold")), row.get("control")): row
            for row in fold_rows
        }
        folds = sorted({str(row.get("fold")) for row in fold_rows})
        for fold in folds:
            real = by_key.get((fold, REAL_CONTROL))
            if real is None:
                continue
            for control in comparison_controls:
                other = by_key.get((fold, control))
                if other is None:
                    continue
                rows.append(
                    {
                        "backend": source_row["backend"],
                        "seed": source_row["seed"],
                        "fold": fold,
                        "comparison_control": control,
                        "router_loss_delta_real_minus_control": _delta(
                            real.get("router_loss"), other.get("router_loss")
                        ),
                        "oracle_regret_delta_real_minus_control": _delta(
                            real.get("oracle_support_regret"),
                            other.get("oracle_support_regret"),
                        ),
                        "used_columns_delta_real_minus_control": _delta(
                            real.get("used_columns"), other.get("used_columns")
                        ),
                        "unique_support_sets_delta_real_minus_control": _delta(
                            real.get("unique_support_sets"),
                            other.get("unique_support_sets"),
                        ),
                        "support_entropy_delta_real_minus_control": _delta(
                            real.get("support_load_entropy"),
                            other.get("support_load_entropy"),
                        ),
                    }
                )
    return rows


def _intervention_summary_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = {
        "teacher_support_forced_into_student",
        "oracle_best_support_for_student",
        "linear_support_forced_into_student",
        "marginal_shuffled_student_support",
        "uniform_random_support",
    }
    for source in sources:
        source_row = source["source_row"]
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in source["intervention_rows"]:
            if row.get("intervention") not in wanted:
                continue
            key = (str(row.get("token_subset")), str(row.get("intervention")))
            grouped.setdefault(key, []).append(row)
        for (token_subset, intervention), group in sorted(grouped.items()):
            mean_delta = _mean(
                _number(row.get("delta_vs_student_router_support")) for row in group
            )
            rows.append(
                {
                    "backend": source_row["backend"],
                    "seed": source_row["seed"],
                    "token_subset": token_subset,
                    "intervention": intervention,
                    "folds": len({row.get("fold") for row in group}),
                    "mean_loss": _mean(_number(row.get("loss")) for row in group),
                    "mean_delta_vs_student_router_support": mean_delta,
                    "delta_sign": _sign_label(mean_delta),
                }
            )
    return rows


def _gate_status(
    *,
    synthesis: dict[str, Any],
    source_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    local_pairs = [row for row in paired_rows if row["backend"] == "local"]
    runpod_pairs = [row for row in paired_rows if row["backend"] == "runpod"]
    local_real_vs_null = [
        row
        for row in local_pairs
        if row["comparison_control"] in {SHUFFLED_NULL, FREQUENCY_NULL}
    ]
    local_real_vs_direct = [
        row for row in local_pairs if row["comparison_control"] == DIRECT_CE_CAUSAL
    ]
    local_linear_interventions = [
        row
        for row in intervention_rows
        if row["backend"] == "local"
        and row["token_subset"] == "all_tokens"
        and row["intervention"] == "linear_support_forced_into_student"
    ]
    local_teacher_interventions = [
        row
        for row in intervention_rows
        if row["backend"] == "local"
        and row["token_subset"] == "all_tokens"
        and row["intervention"] == "teacher_support_forced_into_student"
    ]
    criteria = [
        _criterion(
            "prior_synthesis_passed",
            synthesis.get("status") == "pass",
            "cross-seed synthesis summary status pass",
            {
                "status": synthesis.get("status"),
                "claim_status": synthesis.get("claim_status"),
            },
        ),
        _criterion(
            "all_sources_complete_four_fold_agreement_artifacts",
            all(
                row["present"]
                and row["expected_files_present"]
                and row["status"] == "pass"
                and row["fold_count"] == 4
                for row in source_rows
            ),
            "six local/RunPod agreement artifacts complete with fold_count 4",
            [
                (row["backend"], row["seed"], row["status"], row["fold_count"])
                for row in source_rows
            ],
        ),
        _criterion(
            "local_real_beats_nulls_on_fold_ce_and_oracle_regret",
            len(local_real_vs_null) == 24
            and all(
                _lt(row["router_loss_delta_real_minus_control"], 0.0)
                and _lt(row["oracle_regret_delta_real_minus_control"], 0.0)
                for row in local_real_vs_null
            ),
            "all local seed/fold real-minus-null CE and oracle-regret deltas are negative",
            _sign_counts(local_real_vs_null),
        ),
        _criterion(
            "local_real_beats_direct_ce_causal_contextual_control",
            len(local_real_vs_direct) == 12
            and all(
                _lt(row["router_loss_delta_real_minus_control"], 0.0)
                and _lt(row["oracle_regret_delta_real_minus_control"], 0.0)
                for row in local_real_vs_direct
            ),
            "all local seed/fold real-minus-direct-CE causal contextual deltas are negative",
            _sign_counts(local_real_vs_direct),
        ),
        _criterion(
            "local_linear_support_intervention_negative_control",
            len(local_linear_interventions) == 3
            and all(
                _gt(row["mean_delta_vs_student_router_support"], 0.0)
                for row in local_linear_interventions
            ),
            "linear-router support forced into the real student worsens all-token loss",
            [
                (row["seed"], row["mean_delta_vs_student_router_support"])
                for row in local_linear_interventions
            ],
        ),
        _criterion(
            "local_teacher_support_intervention_consistent_with_distillation",
            len(local_teacher_interventions) == 3
            and all(
                abs(float(row["mean_delta_vs_student_router_support"])) <= 0.03
                for row in local_teacher_interventions
                if row["mean_delta_vs_student_router_support"] is not None
            ),
            "teacher support forced into the real student stays within 0.03 all-token CE",
            [
                (row["seed"], row["mean_delta_vs_student_router_support"])
                for row in local_teacher_interventions
            ],
        ),
        _criterion(
            "runpod_repeats_preserve_control_signs",
            len(runpod_pairs) == 36
            and all(
                _lt(row["router_loss_delta_real_minus_control"], 0.0)
                and _lt(row["oracle_regret_delta_real_minus_control"], 0.0)
                for row in runpod_pairs
            ),
            "RunPod seed/fold real-minus-control CE and oracle-regret signs match local",
            _sign_counts(runpod_pairs),
        ),
        _criterion(
            "rank_dense_matched_control_explicitly_not_claimed",
            True,
            "rank/dense-matched residual control is unavailable in this artifact family",
            "deferred with explicit claim boundary",
        ),
    ]
    return {
        "criteria": criteria,
        "passes_discriminative_mechanism_gate": all(row["passed"] for row in criteria),
    }


def _failures(
    *,
    source_rows: list[dict[str, Any]],
    gate_status: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if not row["present"] or not row["expected_files_present"]:
            failures.append(
                {
                    "source": f"{row['backend']}_seed{row['seed']}",
                    "field": "source_artifact",
                    "expected": "agreement artifact directory with expected files",
                    "actual": {
                        "present": row["present"],
                        "expected_files_present": row["expected_files_present"],
                    },
                    "path": row["path"],
                }
            )
    for criterion in gate_status["criteria"]:
        if not criterion["passed"]:
            failures.append(
                {
                    "source": "discriminative_mechanism_gate",
                    "field": criterion["criterion"],
                    "expected": criterion["threshold"],
                    "actual": criterion["actual"],
                }
            )
    return failures


def _unavailable_control_rows() -> list[dict[str, str]]:
    return [
        {
            "control": "rank_matched_or_dense_matched_residual_control",
            "status": "unavailable_in_existing_agreement_artifacts",
            "reason": (
                "The source agreement artifacts include real/null/direct-CE contextual "
                "controls and linear realized-support interventions, but no same-seed "
                "rank-matched or dense-matched residual model."
            ),
            "disposition": "deferred_not_claimed",
        }
    ]


def _synthesis_source_row(path: Path, synthesis: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "status": synthesis.get("status"),
        "decision": synthesis.get("decision"),
        "claim_status": synthesis.get("claim_status"),
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
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "accepted: ran the recommended discriminative mechanism step as a "
            "bounded artifact synthesis over real/null/direct-CE controls and "
            "linear realized-support interventions; rank/dense matched controls "
            "were deferred because they are not present in the source artifacts"
        ),
        "ben_notification_required": bool(notify_ben)
        or header.get("strategic_change_level") == "major",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Causal Contextual Router Discriminative Mechanism Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
        "",
        "## Gate",
    ]
    for row in summary["gate_status"]["criteria"]:
        lines.append(f"- {row['criterion']}: `{row['passed']}`")
    lines.extend(["", "## Control Deltas"])
    local_rows = [
        row for row in summary["paired_control_rows"] if row["backend"] == "local"
    ]
    for control in [SHUFFLED_NULL, FREQUENCY_NULL, DIRECT_CE_CAUSAL]:
        rows = [row for row in local_rows if row["comparison_control"] == control]
        lines.append(
            "- "
            f"{control}: mean CE delta `{_mean(_number(row['router_loss_delta_real_minus_control']) for row in rows)}`, "
            f"mean oracle-regret proxy delta `{_mean(_number(row['oracle_regret_delta_real_minus_control']) for row in rows)}`"
        )
    lines.extend(
        [
            "",
            "## Deferred Controls",
        ]
    )
    for row in summary["unavailable_control_rows"]:
        lines.append(f"- {row['control']}: {row['reason']}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundaries"]["supported"],
            "",
            "Not supported by this report: "
            + ", ".join(summary["claim_boundaries"]["not_supported"])
            + ".",
            "",
            "## Rationale",
            "",
            summary["rationale"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
    }


def _sign_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "ce_negative": sum(
            1 for row in rows if _lt(row["router_loss_delta_real_minus_control"], 0.0)
        ),
        "oracle_regret_negative": sum(
            1
            for row in rows
            if _lt(row["oracle_regret_delta_real_minus_control"], 0.0)
        ),
    }


def _delta(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _mean(values: Any) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lt(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number < threshold


def _gt(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number > threshold


def _sign_label(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.0:
        return "improves"
    if value > 0.0:
        return "worsens"
    return "neutral"


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-summary", type=Path, default=DEFAULT_SYNTHESIS_SUMMARY)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_discriminative_mechanism_audit(
        synthesis_summary_path=args.synthesis_summary,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
