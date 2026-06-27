"""Fail-closed synthesis for the token/position-stratified distillation null."""

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
    "results/reports/token_larger_causal_contextual_router_stratified_null_reversal"
)

STRATIFIED_NULL = "token_position_frequency_matched_teacher"
CLAIM_NOT_ESTABLISHED = (
    "distilled_causal_router_functional_mechanism_not_established_under_token_position_null"
)
INSUFFICIENT = "insufficient_stratified_null_reversal_evidence"

EXPECTED_FILES = [
    "summary.json",
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "agreement_metrics.csv",
    "intervention_metrics.csv",
    "null_control_metrics.csv",
    "null_sampling_diagnostics.csv",
    "per_token_supports.csv",
    "support_counts.csv",
    "notes.md",
]


def run_causal_contextual_router_stratified_null_reversal_report(
    *,
    local_audit_dirs: list[Path] | None = None,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    start = time.time()
    local_dirs = local_audit_dirs or [
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3"),
    ]
    strategy_review = _strategy_review(strategy_review_path)
    sources = [
        _load_source(seed, path)
        for seed, path in zip([1, 2, 3], local_dirs, strict=True)
    ]
    source_rows = [source["source_row"] for source in sources]
    seed_rows = [source["seed_row"] for source in sources]
    fold_rows = [row for source in sources for row in source["fold_rows"]]
    sampling_rows = [row for source in sources for row in source["sampling_rows"]]
    criteria = _criteria(source_rows, seed_rows, fold_rows, strategy_review)
    failures = [row for row in criteria if not row["passed"]]

    status = "fail" if failures else "pass"
    claim_status = INSUFFICIENT if failures else CLAIM_NOT_ESTABLISHED
    decision = (
        "repair_missing_or_invalid_stratified_null_artifacts"
        if failures
        else "prior_distillation_mechanism_claim_superseded_by_stratified_null"
    )
    selected_next_step = (
        "repair_missing_or_invalid_stratified_null_artifacts"
        if failures
        else "conditional_token_position_vs_context_ablation_before_runpod_repeat"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "next_command": None,
        "strategy_review": strategy_review,
        "source_rows": source_rows,
        "seed_rows": seed_rows,
        "fold_rows": fold_rows,
        "sampling_rows": sampling_rows,
        "gate_status": {"criteria": criteria, "passes_reversal_gate": not failures},
        "failures": failures,
        "claim_boundaries": {
            "supported": (
                "teacher support targets are strongly confounded with target-token/"
                "position structure in the current token-larger causal-router "
                "distillation harness"
            )
            if not failures
            else None,
            "not_supported": [
                "functional causal-router distillation mechanism under token/position null",
                "deployable causal-router default",
                "teacher exact-pair agreement as sufficient mechanism evidence",
                "RunPod validation of the new stratified-null result",
            ],
            "supersedes": (
                "prior shuffled/global-frequency null wins and the discriminative "
                "mechanism report are qualified by this stronger null"
            ),
        },
        "rationale": _rationale(seed_rows, fold_rows, sampling_rows, failures),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_artifacts_csv": str(out_dir / "source_artifacts.csv"),
            "seed_stratified_null_deltas_csv": str(
                out_dir / "seed_stratified_null_deltas.csv"
            ),
            "fold_stratified_null_deltas_csv": str(
                out_dir / "fold_stratified_null_deltas.csv"
            ),
            "null_sampling_summary_csv": str(out_dir / "null_sampling_summary.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_artifacts.csv", source_rows)
    _write_csv(out_dir / "seed_stratified_null_deltas.csv", seed_rows)
    _write_csv(out_dir / "fold_stratified_null_deltas.csv", fold_rows)
    _write_csv(out_dir / "null_sampling_summary.csv", sampling_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_source(seed: int, path: Path) -> dict[str, Any]:
    packet = _read_json_object(path / "summary.json")
    audit = packet.get("audit", {}) if isinstance(packet.get("audit"), dict) else {}
    expected_present = all((path / name).is_file() for name in EXPECTED_FILES)
    source_row = {
        "backend": "local",
        "seed": seed,
        "path": str(path),
        "present": path.is_dir() and (path / "summary.json").is_file(),
        "expected_files_present": expected_present,
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
        "fold_count": audit.get("fold_count"),
        "dataset": audit.get("dataset"),
        "support_router": audit.get("support_router"),
        "top_k": audit.get("top_k"),
        "git_commit": packet.get("git_commit"),
    }
    null_rows = [
        row
        for row in _read_csv_rows(path / "null_control_metrics.csv")
        if row.get("null_control_kind") == STRATIFIED_NULL
    ]
    seed_row = _seed_row(seed, path, null_rows, audit)
    return {
        "source_row": source_row,
        "seed_row": seed_row,
        "fold_rows": [_fold_row(seed, row) for row in null_rows],
        "sampling_rows": _sampling_rows(seed, audit),
    }


def _seed_row(
    seed: int,
    path: Path,
    null_rows: list[dict[str, str]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    aggregate = {}
    for key, value in (audit.get("null_control_aggregates") or {}).items():
        if isinstance(value, dict) and value.get("mean_null_control_kind") == STRATIFIED_NULL:
            aggregate = value
        elif key.startswith("causal_distilled_from_token_position_frequency_matched_teacher"):
            aggregate = value
    ce_values = [_float(row["student_minus_null_router_loss"]) for row in null_rows]
    regret_values = [_float(row["student_minus_null_oracle_regret"]) for row in null_rows]
    agreement_values = [
        _float(row["student_minus_null_teacher_exact_pair_agreement"])
        for row in null_rows
    ]
    return {
        "seed": seed,
        "path": str(path),
        "folds": len(null_rows),
        "mean_student_minus_stratified_null_router_loss": aggregate.get(
            "mean_student_minus_null_router_loss",
            _mean(ce_values),
        ),
        "mean_student_minus_stratified_null_oracle_regret": aggregate.get(
            "mean_student_minus_null_oracle_regret",
            _mean(regret_values),
        ),
        "mean_student_minus_stratified_null_teacher_exact_pair_agreement": aggregate.get(
            "mean_student_minus_null_teacher_exact_pair_agreement",
            _mean(agreement_values),
        ),
        "real_beats_stratified_null_ce_folds": sum(value < 0.0 for value in ce_values),
        "real_beats_stratified_null_regret_folds": sum(
            value < 0.0 for value in regret_values
        ),
        "real_beats_stratified_null_agreement_folds": sum(
            value > 0.0 for value in agreement_values
        ),
    }


def _fold_row(seed: int, row: dict[str, str]) -> dict[str, Any]:
    return {
        "seed": seed,
        "fold": row.get("fold"),
        "null_control": row.get("null_control"),
        "student_minus_stratified_null_router_loss": _float(
            row.get("student_minus_null_router_loss")
        ),
        "student_minus_stratified_null_oracle_regret": _float(
            row.get("student_minus_null_oracle_regret")
        ),
        "student_minus_stratified_null_teacher_exact_pair_agreement": _float(
            row.get("student_minus_null_teacher_exact_pair_agreement")
        ),
    }


def _sampling_rows(seed: int, audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for null_control, row in (audit.get("null_sampling_aggregates") or {}).items():
        rows.append({"seed": seed, "null_control": null_control, **row})
    return rows


def _criteria(
    source_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "urgent_review_consumed_and_notifies_ben",
            strategy_review.get("notify_ben") is True
            and strategy_review.get("strategic_change_level") == "major",
            "latest urgent review has notify_ben true and major change level",
            {
                "notify_ben": strategy_review.get("notify_ben"),
                "strategic_change_level": strategy_review.get("strategic_change_level"),
            },
        ),
        _criterion(
            "all_local_seed_artifacts_present",
            all(row["expected_files_present"] for row in source_rows),
            "all expected files including null_sampling_diagnostics.csv are present",
            [row["expected_files_present"] for row in source_rows],
        ),
        _criterion(
            "all_sources_four_fold",
            all(row.get("fold_count") == 4 for row in source_rows)
            and all(row.get("folds") == 4 for row in seed_rows),
            "each seed has four folds and four stratified-null rows",
            [(row.get("seed"), row.get("fold_count")) for row in source_rows],
        ),
        _criterion(
            "stratified_null_erases_functional_margin",
            max(
                abs(float(row["mean_student_minus_stratified_null_router_loss"]))
                for row in seed_rows
            )
            < 0.02,
            "max absolute seed CE delta vs stratified null < 0.02",
            [
                row["mean_student_minus_stratified_null_router_loss"]
                for row in seed_rows
            ],
        ),
        _criterion(
            "fold_signs_are_mixed",
            0
            < sum(
                float(row["student_minus_stratified_null_router_loss"]) < 0.0
                for row in fold_rows
            )
            < len(fold_rows),
            "real-vs-stratified-null CE fold signs are mixed",
            [
                row["student_minus_stratified_null_router_loss"]
                for row in fold_rows
            ],
        ),
        _criterion(
            "teacher_agreement_survives_without_functional_gate",
            all(
                float(
                    row[
                        "mean_student_minus_stratified_null_teacher_exact_pair_agreement"
                    ]
                )
                > 0.0
                for row in seed_rows
            ),
            "real student keeps positive teacher-agreement delta",
            [
                row["mean_student_minus_stratified_null_teacher_exact_pair_agreement"]
                for row in seed_rows
            ],
        ),
    ]


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _rationale(
    seed_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    sampling_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    if failures:
        return (
            "The reversal report fails closed because required strengthened-null "
            "artifacts or diagnostic gates are incomplete."
        )
    ce_deltas = [
        float(row["mean_student_minus_stratified_null_router_loss"]) for row in seed_rows
    ]
    negative_folds = sum(
        float(row["student_minus_stratified_null_router_loss"]) < 0.0
        for row in fold_rows
    )
    sampling = sampling_rows[0] if sampling_rows else {}
    return (
        "The token/position-stratified support null reduces the prior functional "
        f"distillation advantage to near-zero seed CE deltas {ce_deltas}, with "
        f"mixed fold signs ({negative_folds}/{len(fold_rows)} real-better folds). "
        "Teacher exact-pair agreement remains positive, so the prior mechanism "
        "claim is downgraded rather than erased: exact teacher-label recovery is "
        "not sufficient evidence for functional causal-support usefulness. "
        f"Null sampling diagnostics include target-position fraction "
        f"{sampling.get('target_position_fraction')} for the first seed."
    )


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": False,
            "recommended_next_action": None,
        }
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": _header_value(lines, "strategic_change_level"),
        "notify_ben": _header_value(lines, "notify_ben") == "true",
        "recommended_next_action": _header_value(lines, "recommended_next_action"),
    }


def _header_value(lines: list[str], key: str) -> str | None:
    prefix = f"{key}:"
    for line in lines[:10]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Token/Position-Stratified Null Reversal",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
        f"- Rationale: {summary['rationale']}",
        "",
        "## Seed Deltas",
    ]
    for row in summary["seed_rows"]:
        lines.append(
            "- "
            f"seed `{row['seed']}`: CE delta "
            f"`{row['mean_student_minus_stratified_null_router_loss']}`, "
            f"regret delta `{row['mean_student_minus_stratified_null_oracle_regret']}`, "
            f"agreement delta "
            f"`{row['mean_student_minus_stratified_null_teacher_exact_pair_agreement']}`"
        )
    lines.extend(["", "## Gate Criteria"])
    for row in summary["gate_status"]["criteria"]:
        lines.append(
            f"- {row['criterion']}: `{row['passed']}` "
            f"(threshold `{row['threshold']}`, actual `{row['actual']}`)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_contextual_router_stratified_null_reversal_report(
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
