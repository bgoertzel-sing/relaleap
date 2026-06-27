"""Same-student support intervention matrix for causal-router distillation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_same_student_intervention_matrix"
)

AVAILABLE_INTERVENTIONS = {
    "student_router_support": "student_router_support_loss",
    "teacher_support_forced_into_student": "teacher_support_forced_into_student_loss",
    "oracle_best_support_for_student": "oracle_best_support_for_student_loss",
    "linear_support_forced_into_student": "linear_support_forced_into_student_loss",
    "marginal_shuffled_student_support": "marginal_shuffled_student_support_loss",
    "uniform_random_support": "uniform_random_support_loss",
}
REQUIRED_MISSING_INTERVENTION = "token_position_null_support_forced_into_student"

INCOMPLETE_MATRIX = "same_student_token_position_null_extension_required"
INSUFFICIENT = "insufficient_same_student_intervention_evidence"


def run_causal_contextual_router_same_student_intervention_matrix(
    *,
    local_audit_dirs: list[Path] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    teacher_gain_margin: float = 0.0,
) -> dict[str, Any]:
    start = time.time()
    audit_dirs = local_audit_dirs or [
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3"),
    ]
    sources = [
        _load_seed(seed, path)
        for seed, path in zip([1, 2, 3], audit_dirs, strict=True)
    ]
    source_rows = [source["source_row"] for source in sources]
    token_rows = [row for source in sources for row in source["token_rows"]]
    matrix_rows = _matrix_rows(token_rows)
    seed_matrix_rows = _matrix_rows(
        token_rows,
        keys=("seed", "token_subset", "intervention"),
    )
    separate_student_null_rows = [
        row for source in sources for row in source["separate_student_null_rows"]
    ]
    key_metrics = _key_metrics(matrix_rows, separate_student_null_rows)
    criteria = _criteria(source_rows, token_rows, matrix_rows)
    failures = [row for row in criteria if not row["passed"]]
    status = (
        "fail"
        if any(
            row["criterion"] != "token_position_null_same_student_arm_present"
            for row in failures
        )
        else "pass"
    )

    teacher_all_token_gain = key_metrics.get("teacher_forced_gain_all_tokens")
    teacher_helps_all_tokens = (
        teacher_all_token_gain is not None and teacher_all_token_gain > teacher_gain_margin
    )
    if status == "fail":
        decision = "repair_missing_or_invalid_same_student_sources"
        claim_status = INSUFFICIENT
        selected_next_step = "repair_same_student_intervention_sources"
    else:
        decision = "same_student_matrix_requires_token_position_null_artifact_extension"
        claim_status = INCOMPLETE_MATRIX
        selected_next_step = (
            "extend_distillation_agreement_audit_with_token_position_null_forced_support"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "experiment_id": "token_larger_causal_contextual_router_same_student_intervention_matrix",
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "source_rows": source_rows,
        "matrix_rows": matrix_rows,
        "seed_matrix_rows": seed_matrix_rows,
        "separate_student_null_rows": separate_student_null_rows,
        "key_metrics": key_metrics,
        "gate_status": {
            "criteria": criteria,
            "passes_artifact_gate": status == "pass",
            "teacher_forced_all_token_gain_positive": teacher_helps_all_tokens,
        },
        "failures": failures,
        "claim_boundaries": {
            "supported": _supported_claim(key_metrics),
            "not_supported": [
                "functional causal-router distillation mechanism",
                "teacher exact-pair agreement as sufficient mechanism evidence",
                "token/position-null-discriminative same-student support usefulness",
                "RunPod validation of the strengthened-null reversal",
            ],
        },
        "rationale": _rationale(status, key_metrics, failures),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "same_student_matrix_csv": str(out_dir / "same_student_matrix.csv"),
            "seed_same_student_matrix_csv": str(out_dir / "seed_same_student_matrix.csv"),
            "separate_student_null_reference_csv": str(
                out_dir / "separate_student_null_reference.csv"
            ),
            "source_artifacts_csv": str(out_dir / "source_artifacts.csv"),
            "gate_criteria_csv": str(out_dir / "gate_criteria.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_artifacts.csv", source_rows)
    _write_csv(out_dir / "same_student_matrix.csv", matrix_rows)
    _write_csv(out_dir / "seed_same_student_matrix.csv", seed_matrix_rows)
    _write_csv(out_dir / "separate_student_null_reference.csv", separate_student_null_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_seed(seed: int, path: Path) -> dict[str, Any]:
    packet = _read_json_object(path / "summary.json")
    audit = packet.get("audit", {}) if isinstance(packet.get("audit"), dict) else {}
    token_rows = [
        _token_row(seed, row)
        for row in _read_csv_rows(path / "per_token_supports.csv")
    ]
    return {
        "source_row": {
            "seed": seed,
            "path": str(path),
            "present": path.is_dir(),
            "summary_present": (path / "summary.json").is_file(),
            "per_token_supports_present": (path / "per_token_supports.csv").is_file(),
            "null_control_metrics_present": (path / "null_control_metrics.csv").is_file(),
            "status": packet.get("status"),
            "decision": packet.get("decision"),
            "claim_status": packet.get("claim_status"),
            "fold_count": audit.get("fold_count"),
            "dataset": audit.get("dataset"),
            "support_router": audit.get("support_router"),
            "top_k": audit.get("top_k"),
            "positions": len(token_rows),
            "has_token_position_null_same_student_arm": any(
                REQUIRED_MISSING_INTERVENTION in row for row in token_rows
            ),
            "git_commit": packet.get("git_commit"),
        },
        "token_rows": token_rows,
        "separate_student_null_rows": _separate_student_null_rows(
            seed, path / "null_control_metrics.csv"
        ),
    }


def _token_row(seed: int, row: dict[str, str]) -> dict[str, Any]:
    student_loss = _float(row.get("student_router_support_loss"))
    oracle_loss = _float(row.get("oracle_best_support_for_student_loss"))
    teacher_match = str(row.get("teacher_student_exact_pair_match", "")).lower() == "true"
    output: dict[str, Any] = {
        "seed": seed,
        "fold": int(row.get("fold", 0)),
        "flat_position": int(row.get("flat_position", 0)),
        "target_token": int(row.get("target_token", 0)),
        "teacher_student_exact_pair_match": teacher_match,
        "student_oracle_gap": student_loss - oracle_loss,
    }
    for intervention, column in AVAILABLE_INTERVENTIONS.items():
        output[intervention] = _float(row.get(column))
    return output


def _matrix_rows(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ("token_subset", "intervention"),
) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        for subset in _subsets(row):
            for intervention in AVAILABLE_INTERVENTIONS:
                expanded.append(
                    {
                        **{key: row[key] for key in ("seed", "fold")},
                        "token_subset": subset,
                        "intervention": intervention,
                        "loss": row[intervention],
                        "student_loss": row["student_router_support"],
                        "oracle_loss": row["oracle_best_support_for_student"],
                    }
                )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(groups.items()):
        loss = _mean([row["loss"] for row in group_rows])
        student = _mean([row["student_loss"] for row in group_rows])
        oracle = _mean([row["oracle_loss"] for row in group_rows])
        result = {key: value for key, value in zip(keys, group_key, strict=True)}
        result.update(
            {
                "positions": len(group_rows),
                "mean_loss": loss,
                "delta_vs_student_router": loss - student,
                "gain_vs_student_router": student - loss,
                "delta_vs_oracle": loss - oracle,
                "mean_student_router_loss": student,
                "mean_oracle_loss": oracle,
            }
        )
        output.append(result)
    return output


def _subsets(row: dict[str, Any]) -> list[str]:
    subsets = ["all_tokens"]
    if not row["teacher_student_exact_pair_match"]:
        subsets.append("teacher_student_disagreement_tokens")
    if row["student_oracle_gap"] > 0.0:
        subsets.append("student_positive_oracle_regret_tokens")
    return subsets


def _separate_student_null_rows(seed: int, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv_rows(path):
        if row.get("null_control_kind") != "token_position_frequency_matched_teacher":
            continue
        rows.append(
            {
                "seed": seed,
                "fold": row.get("fold"),
                "null_control": row.get("null_control"),
                "student_minus_null_router_loss": _float(
                    row.get("student_minus_null_router_loss")
                ),
                "student_minus_null_oracle_regret": _float(
                    row.get("student_minus_null_oracle_regret")
                ),
                "student_minus_null_teacher_exact_pair_agreement": _float(
                    row.get("student_minus_null_teacher_exact_pair_agreement")
                ),
                "comparison_type": "separate_student_reference_not_same_student_intervention",
            }
        )
    return rows


def _key_metrics(
    matrix_rows: list[dict[str, Any]],
    separate_student_null_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (row["token_subset"], row["intervention"]): row
        for row in matrix_rows
        if "seed" not in row
    }
    teacher_all = by_key.get(("all_tokens", "teacher_support_forced_into_student"), {})
    oracle_all = by_key.get(("all_tokens", "oracle_best_support_for_student"), {})
    teacher_disagree = by_key.get(
        ("teacher_student_disagreement_tokens", "teacher_support_forced_into_student"),
        {},
    )
    return {
        "teacher_forced_gain_all_tokens": teacher_all.get("gain_vs_student_router"),
        "teacher_forced_gain_disagreement_tokens": teacher_disagree.get(
            "gain_vs_student_router"
        ),
        "oracle_gain_all_tokens": oracle_all.get("gain_vs_student_router"),
        "mean_separate_student_minus_token_position_null_router_loss": _mean(
            [row["student_minus_null_router_loss"] for row in separate_student_null_rows]
        ),
        "separate_student_token_position_null_fold_wins": sum(
            row["student_minus_null_router_loss"] < 0.0
            for row in separate_student_null_rows
        ),
        "separate_student_token_position_null_folds": len(separate_student_null_rows),
        "token_position_null_same_student_arm_available": False,
    }


def _criteria(
    source_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    present_interventions = {
        row["intervention"] for row in matrix_rows if row.get("token_subset") == "all_tokens"
    }
    return [
        _criterion(
            "all_local_seed_sources_present",
            all(
                row["summary_present"] and row["per_token_supports_present"]
                for row in source_rows
            ),
            "summary.json and per_token_supports.csv present for seeds 1-3",
            [
                (
                    row["seed"],
                    row["summary_present"],
                    row["per_token_supports_present"],
                )
                for row in source_rows
            ],
        ),
        _criterion(
            "per_token_rows_present",
            len(token_rows) > 0,
            "at least one per-token row",
            len(token_rows),
        ),
        _criterion(
            "available_same_student_arms_present",
            set(AVAILABLE_INTERVENTIONS).issubset(present_interventions),
            sorted(AVAILABLE_INTERVENTIONS),
            sorted(present_interventions),
        ),
        _criterion(
            "token_position_null_same_student_arm_present",
            False,
            REQUIRED_MISSING_INTERVENTION,
            "not present in current per_token_supports.csv schema",
        ),
    ]


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _supported_claim(key_metrics: dict[str, Any]) -> str:
    return (
        "The current artifacts quantify available same-student support interventions, "
        "but they do not contain the required token/position-null forced-support arm."
    )


def _rationale(
    status: str,
    key_metrics: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    if status == "fail":
        return "Same-student matrix failed because required source artifacts are missing or invalid."
    teacher_gain = key_metrics.get("teacher_forced_gain_all_tokens")
    null_delta = key_metrics.get("mean_separate_student_minus_token_position_null_router_loss")
    return (
        "The available same-student arms show teacher-forced support changes all-token "
        f"loss by {teacher_gain:.6f} versus the learned student router, while the "
        "token/position-null evidence is still only a separate-student reference "
        f"(mean student-minus-null router loss {null_delta:.6f}). The next artifact "
        "extension must force token/position-null supports through the same trained "
        "student before any mechanism claim can be reconsidered."
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Same-Student Support Intervention Matrix",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        f"Claim status: `{summary['claim_status']}`",
        f"Selected next step: `{summary['selected_next_step']}`",
        "",
        "## Key metrics",
    ]
    for key, value in summary["key_metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Rationale", summary["rationale"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, action="append", dest="audit_dirs")
    args = parser.parse_args()
    summary = run_causal_contextual_router_same_student_intervention_matrix(
        local_audit_dirs=args.audit_dirs,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
