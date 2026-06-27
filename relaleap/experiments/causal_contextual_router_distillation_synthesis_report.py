"""Cross-seed synthesis for causal-router distillation agreement audits."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_distillation_synthesis"
)

SUPPORTED = "distilled_causal_router_cross_seed_mechanism_supported_not_promoted"
INSUFFICIENT = "insufficient_cross_seed_distillation_evidence"
NEXT_MECHANISM_AUDIT = "causal_router_discriminative_mechanism_control_audit"

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


def run_causal_contextual_router_distillation_synthesis_report(
    *,
    local_audit_dirs: list[Path] | None = None,
    runpod_audit_dirs: list[Path] | None = None,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume existing local/RunPod seed-1/2/3 agreement artifacts."""

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
    sources = [
        _load_source("local", seed, path)
        for seed, path in zip([1, 2, 3], local_dirs, strict=True)
    ] + [
        _load_source("runpod", seed, path)
        for seed, path in zip([1, 2, 3], runpod_dirs, strict=True)
    ]
    source_rows = [source["source_row"] for source in sources]
    delta_rows = [
        row for source in sources for row in source["delta_rows"]
    ]
    fold_delta_rows = [
        row for source in sources for row in source["fold_delta_rows"]
    ]
    backend_rows = _backend_reproducibility_rows(delta_rows)
    gate_status = _gate_status(
        sources=sources,
        delta_rows=delta_rows,
        fold_delta_rows=fold_delta_rows,
        backend_rows=backend_rows,
    )
    failures = _failures(source_rows, gate_status)

    if failures:
        status = "fail"
        decision = INSUFFICIENT
        claim_status = INSUFFICIENT
        selected_next_step = "repair_missing_or_failed_distillation_agreement_artifacts"
        rationale = (
            "The cross-seed synthesis fails closed because required source "
            "artifacts, fold counts, pass statuses, or real-vs-null sign gates "
            "are missing or inconsistent."
        )
    else:
        status = "pass"
        decision = "teacher_support_targets_carry_non_marginal_context_information"
        claim_status = SUPPORTED
        selected_next_step = NEXT_MECHANISM_AUDIT
        rationale = (
            "Across three local seeds the teacher-distilled causal student beats "
            "both shuffled-teacher and frequency-matched support-target nulls on "
            "paired fold CE, the student-adapter oracle-regret proxy, and teacher "
            "support agreement. RunPod repeats preserve the same sign pattern and "
            "serve as backend reproducibility checks rather than extra seeds. "
            "This supports the bounded mechanism claim that teacher support "
            "targets carry non-marginal, context-specific information under the "
            "token-larger causal-router distillation harness. It does not support "
            "default promotion, full causal separability, or general reusable "
            "causal-column claims."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": (
            "python -m relaleap.experiments.causal_contextual_router_"
            "discriminative_mechanism_audit"
        )
        if status == "pass"
        else None,
        "strategy_review": strategy_review,
        "source_rows": source_rows,
        "delta_rows": delta_rows,
        "fold_delta_rows": fold_delta_rows,
        "backend_reproducibility_rows": backend_rows,
        "gate_status": gate_status,
        "failures": failures,
        "claim_boundaries": {
            "supported": (
                "teacher support targets carry non-marginal, context-specific "
                "information under the token-larger causal-router distillation harness"
            ),
            "not_supported": [
                "deployable causal router default",
                "general causal column reuse",
                "full causal separability",
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
            "seed_backend_deltas_csv": str(out_dir / "seed_backend_deltas.csv"),
            "fold_paired_deltas_csv": str(out_dir / "fold_paired_deltas.csv"),
            "backend_reproducibility_csv": str(out_dir / "backend_reproducibility.csv"),
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
        out_dir / "seed_backend_deltas.csv",
        [
            "backend",
            "seed",
            "null_control_kind",
            "fold_count",
            "mean_student_minus_null_router_loss",
            "mean_student_minus_null_oracle_regret",
            "mean_student_minus_null_teacher_exact_pair_agreement",
            "router_loss_real_beats_null_folds",
            "oracle_regret_real_beats_null_folds",
            "teacher_agreement_real_beats_null_folds",
        ],
        delta_rows,
    )
    _write_csv(
        out_dir / "fold_paired_deltas.csv",
        [
            "backend",
            "seed",
            "fold",
            "null_control_kind",
            "student_minus_null_router_loss",
            "student_minus_null_oracle_regret",
            "student_minus_null_teacher_exact_pair_agreement",
        ],
        fold_delta_rows,
    )
    _write_csv(
        out_dir / "backend_reproducibility.csv",
        [
            "seed",
            "null_control_kind",
            "local_mean_ce_delta",
            "runpod_mean_ce_delta",
            "ce_delta_abs_diff",
            "local_agreement_delta",
            "runpod_agreement_delta",
            "same_sign",
        ],
        backend_rows,
    )
    _write_csv(
        out_dir / "gate_criteria.csv",
        ["criterion", "passed", "threshold", "actual"],
        gate_status["criteria"],
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
    fold_rows = _read_csv_rows(path / "null_control_metrics.csv")
    delta_rows = _delta_rows(backend, seed, audit, fold_rows)
    fold_delta_rows = _fold_delta_rows(backend, seed, fold_rows)
    return {
        "source_row": source_row,
        "delta_rows": delta_rows,
        "fold_delta_rows": fold_delta_rows,
    }


def _delta_rows(
    backend: str,
    seed: int,
    audit: dict[str, Any],
    fold_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    aggregates = (
        audit.get("null_control_aggregates", {})
        if isinstance(audit.get("null_control_aggregates"), dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for name, aggregate in sorted(aggregates.items()):
        kind = str(aggregate.get("null_control", name)).replace(
            "causal_distilled_from_", ""
        )
        rows.append(
            {
                "backend": backend,
                "seed": seed,
                "null_control": name,
                "null_control_kind": _kind_from_name(kind),
                "fold_count": aggregate.get("folds"),
                "mean_student_minus_null_router_loss": aggregate.get(
                    "mean_student_minus_null_router_loss"
                ),
                "mean_student_minus_null_oracle_regret": aggregate.get(
                    "mean_student_minus_null_oracle_regret"
                ),
                "mean_student_minus_null_teacher_exact_pair_agreement": aggregate.get(
                    "mean_student_minus_null_teacher_exact_pair_agreement"
                ),
                "router_loss_real_beats_null_folds": sum(
                    1
                    for row in fold_rows
                    if row.get("null_control") == name
                    and _lt(row.get("student_minus_null_router_loss"), 0.0)
                ),
                "oracle_regret_real_beats_null_folds": sum(
                    1
                    for row in fold_rows
                    if row.get("null_control") == name
                    and _lt(row.get("student_minus_null_oracle_regret"), 0.0)
                ),
                "teacher_agreement_real_beats_null_folds": sum(
                    1
                    for row in fold_rows
                    if row.get("null_control") == name
                    and _gt(
                        row.get("student_minus_null_teacher_exact_pair_agreement"),
                        0.0,
                    )
                ),
            }
        )
    return rows


def _fold_delta_rows(
    backend: str, seed: int, fold_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fold_rows:
        rows.append(
            {
                "backend": backend,
                "seed": seed,
                "fold": row.get("fold"),
                "null_control": row.get("null_control"),
                "null_control_kind": _kind_from_name(row.get("null_control_kind")),
                "student_minus_null_router_loss": row.get(
                    "student_minus_null_router_loss"
                ),
                "student_minus_null_oracle_regret": row.get(
                    "student_minus_null_oracle_regret"
                ),
                "student_minus_null_teacher_exact_pair_agreement": row.get(
                    "student_minus_null_teacher_exact_pair_agreement"
                ),
            }
        )
    return rows


def _backend_reproducibility_rows(
    delta_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed = {
        (row["backend"], row["seed"], row["null_control_kind"]): row
        for row in delta_rows
    }
    rows: list[dict[str, Any]] = []
    for seed in [1, 2, 3]:
        for kind in ["frequency_matched_teacher", "shuffled_teacher"]:
            local = indexed.get(("local", seed, kind), {})
            runpod = indexed.get(("runpod", seed, kind), {})
            local_ce = _number(local.get("mean_student_minus_null_router_loss"))
            runpod_ce = _number(runpod.get("mean_student_minus_null_router_loss"))
            local_agreement = _number(
                local.get("mean_student_minus_null_teacher_exact_pair_agreement")
            )
            runpod_agreement = _number(
                runpod.get("mean_student_minus_null_teacher_exact_pair_agreement")
            )
            rows.append(
                {
                    "seed": seed,
                    "null_control_kind": kind,
                    "local_mean_ce_delta": local_ce,
                    "runpod_mean_ce_delta": runpod_ce,
                    "ce_delta_abs_diff": (
                        abs(local_ce - runpod_ce)
                        if local_ce is not None and runpod_ce is not None
                        else None
                    ),
                    "local_agreement_delta": local_agreement,
                    "runpod_agreement_delta": runpod_agreement,
                    "same_sign": (
                        local_ce is not None
                        and runpod_ce is not None
                        and local_ce < 0.0
                        and runpod_ce < 0.0
                        and local_agreement is not None
                        and runpod_agreement is not None
                        and local_agreement > 0.0
                        and runpod_agreement > 0.0
                    ),
                }
            )
    return rows


def _gate_status(
    *,
    sources: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    fold_delta_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_rows = [source["source_row"] for source in sources]
    local_deltas = [row for row in delta_rows if row["backend"] == "local"]
    criteria = [
        _criterion(
            "all_expected_sources_present",
            all(row["present"] and row["expected_files_present"] for row in source_rows),
            "six seed/backend artifact directories and expected files present",
            {
                "present_sources": sum(1 for row in source_rows if row["present"]),
                "complete_sources": sum(
                    1 for row in source_rows if row["expected_files_present"]
                ),
            },
        ),
        _criterion(
            "all_sources_pass_four_folds",
            all(row["status"] == "pass" and row["fold_count"] == 4 for row in source_rows),
            "each source summary status pass with fold_count 4",
            [(row["backend"], row["seed"], row["status"], row["fold_count"]) for row in source_rows],
        ),
        _criterion(
            "local_three_seed_mean_deltas_pass",
            len(local_deltas) == 6
            and all(
                _lt(row["mean_student_minus_null_router_loss"], 0.0)
                and _lt(row["mean_student_minus_null_oracle_regret"], 0.0)
                and _gt(row["mean_student_minus_null_teacher_exact_pair_agreement"], 0.0)
                for row in local_deltas
            ),
            "three local seeds beat both nulls on CE, oracle-regret proxy, and agreement",
            [
                (
                    row["seed"],
                    row["null_control_kind"],
                    row["mean_student_minus_null_router_loss"],
                    row["mean_student_minus_null_oracle_regret"],
                    row["mean_student_minus_null_teacher_exact_pair_agreement"],
                )
                for row in local_deltas
            ],
        ),
        _criterion(
            "local_fold_sign_consistency",
            len([row for row in fold_delta_rows if row["backend"] == "local"]) == 24
            and all(
                _lt(row["student_minus_null_router_loss"], 0.0)
                and _lt(row["student_minus_null_oracle_regret"], 0.0)
                and _gt(row["student_minus_null_teacher_exact_pair_agreement"], 0.0)
                for row in fold_delta_rows
                if row["backend"] == "local"
            ),
            "all 24 local seed/fold/null paired deltas have the expected sign",
            _sign_counts([row for row in fold_delta_rows if row["backend"] == "local"]),
        ),
        _criterion(
            "runpod_backend_reproducibility_same_sign",
            len(backend_rows) == 6 and all(row["same_sign"] is True for row in backend_rows),
            "RunPod repeats preserve local CE and agreement delta signs",
            backend_rows,
        ),
    ]
    return {
        "criteria": criteria,
        "passes_synthesis_gate": all(row["passed"] for row in criteria),
    }


def _sign_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "ce_negative": sum(
            1 for row in rows if _lt(row["student_minus_null_router_loss"], 0.0)
        ),
        "oracle_regret_negative": sum(
            1 for row in rows if _lt(row["student_minus_null_oracle_regret"], 0.0)
        ),
        "agreement_positive": sum(
            1
            for row in rows
            if _gt(row["student_minus_null_teacher_exact_pair_agreement"], 0.0)
        ),
    }


def _failures(
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
                    "expected": "artifact directory with expected files",
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
                    "source": "synthesis_gate",
                    "field": criterion["criterion"],
                    "expected": criterion["threshold"],
                    "actual": criterion["actual"],
                }
            )
    return failures


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
            "accepted: implemented the recommended cross-seed causal-router "
            "distillation synthesis with paired fold-level real-vs-null deltas, "
            "backend reproducibility checks, conservative claim boundaries, and "
            "promotion/default-change blocking"
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
        "# Causal Contextual Router Distillation Synthesis",
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
    lines.extend(["", "## Seed/Backend Deltas"])
    for row in summary["delta_rows"]:
        lines.append(
            "- "
            f"{row['backend']} seed {row['seed']} {row['null_control_kind']}: "
            f"CE `{row['mean_student_minus_null_router_loss']}`, "
            f"oracle-regret proxy `{row['mean_student_minus_null_oracle_regret']}`, "
            f"teacher agreement `{row['mean_student_minus_null_teacher_exact_pair_agreement']}`"
        )
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
    if summary["failures"]:
        lines.extend(["## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def _criterion(
    criterion: str, passed: bool, threshold: str, actual: Any
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": json.dumps(actual, sort_keys=True),
    }


def _kind_from_name(value: str | None) -> str:
    text = str(value or "")
    if "frequency_matched_teacher" in text:
        return "frequency_matched_teacher"
    if "shuffled_teacher" in text:
        return "shuffled_teacher"
    return text


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


def _lt(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number < right_number


def _gt(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number > right_number


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_distillation_synthesis_report(
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
