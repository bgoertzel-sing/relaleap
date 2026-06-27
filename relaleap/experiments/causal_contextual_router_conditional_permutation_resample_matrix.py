"""Conditional-permutation resample matrix for causal-router support assignments."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_conditional_permutation_resample_matrix"
)

EXPECTED_FILES = [
    "summary.json",
    "per_token_supports.csv",
    "null_control_metrics.csv",
    "null_sampling_diagnostics.csv",
]

INSUFFICIENT = "insufficient_conditional_permutation_evidence"
ASSIGNMENT_ONLY = "teacher_support_assignment_exceeds_conditional_null_but_functional_claim_blocked"
FUNCTIONAL_NOT_ESTABLISHED = (
    "distilled_causal_router_functional_mechanism_not_established_under_conditional_"
    "permutation_null"
)


def run_causal_contextual_router_conditional_permutation_resample_matrix(
    *,
    local_audit_dirs: list[Path] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    resamples: int = 256,
    random_seed: int = 4309,
    p_value_threshold: float = 0.05,
) -> dict[str, Any]:
    start = time.time()
    if resamples < 1:
        raise ValueError("resamples must be positive")
    audit_dirs = local_audit_dirs or [
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3"),
    ]
    sources = [_load_seed(seed, path) for seed, path in zip([1, 2, 3], audit_dirs, strict=True)]
    source_rows = [source["source_row"] for source in sources]
    token_rows = [row for source in sources for row in source["token_rows"]]
    seed_observed_rows = [_observed_assignment_row(source["token_rows"], source["seed"]) for source in sources]
    observed_row = _observed_assignment_row(token_rows, "all")
    resample_rows = [
        row
        for source in sources
        for row in _resample_seed(
            source["token_rows"],
            seed=source["seed"],
            resamples=resamples,
            random_seed=random_seed + int(source["seed"]) * 1000,
        )
    ]
    resample_summary_rows = _resample_summary_rows(resample_rows)
    null_quality_rows = _null_quality_rows(resample_rows)
    functional_rows = _functional_rows(token_rows)
    key_metrics = _key_metrics(
        observed_row=observed_row,
        resample_rows=resample_rows,
        functional_rows=functional_rows,
    )
    criteria = _criteria(source_rows, token_rows, resample_rows, key_metrics)
    failures = [row for row in criteria if not row["passed"]]
    status = "fail" if failures else "pass"

    assignment_passes = bool(
        key_metrics.get("student_exact_agreement_empirical_p_upper") is not None
        and key_metrics["student_exact_agreement_empirical_p_upper"] <= p_value_threshold
        and key_metrics.get("student_exact_agreement_effect_vs_null_mean", 0.0) > 0.0
    )
    functional_passes = bool(
        key_metrics.get("teacher_minus_token_position_null_gain_all_tokens") is not None
        and key_metrics["teacher_minus_token_position_null_gain_all_tokens"] > 0.0
        and key_metrics.get("teacher_forced_gain_all_tokens", 0.0) > 0.0
    )
    if status == "fail":
        decision = "repair_missing_or_invalid_conditional_permutation_sources"
        claim_status = INSUFFICIENT
        selected_next_step = "repair_conditional_permutation_sources"
    elif assignment_passes and functional_passes:
        decision = "conditional_permutation_assignment_and_functional_gates_pass_locally"
        claim_status = ASSIGNMENT_ONLY
        selected_next_step = "design_fresh_functional_conditional_permutation_audit_before_runpod"
    elif assignment_passes:
        decision = "conditional_permutation_assignment_signal_survives_functional_gate_blocks"
        claim_status = ASSIGNMENT_ONLY
        selected_next_step = "keep_causal_router_distillation_promotion_frozen"
    else:
        decision = "conditional_permutation_resamples_do_not_support_mechanism_claim"
        claim_status = FUNCTIONAL_NOT_ESTABLISHED
        selected_next_step = "keep_causal_router_distillation_promotion_frozen"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "experiment_id": "token_larger_causal_contextual_router_conditional_permutation_resample_matrix",
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "resamples_per_seed": resamples,
        "random_seed": random_seed,
        "source_rows": source_rows,
        "observed_assignment_row": observed_row,
        "seed_observed_assignment_rows": seed_observed_rows,
        "resample_summary_rows": resample_summary_rows,
        "null_quality_rows": null_quality_rows,
        "functional_rows": functional_rows,
        "key_metrics": key_metrics,
        "gate_status": {
            "criteria": criteria,
            "passes_artifact_gate": status == "pass",
            "assignment_gate_passes": assignment_passes,
            "functional_gate_passes": functional_passes,
        },
        "failures": failures,
        "claim_boundaries": {
            "supported": _supported_claim(assignment_passes),
            "not_supported": [
                "functional causal-router distillation mechanism",
                "default causal-router distillation promotion",
                "RunPod validation of the strengthened-null reversal",
                "functional conditional-permutation loss superiority from support-label resampling alone",
            ],
        },
        "rationale": _rationale(status, key_metrics, failures),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "observed_assignment_metrics_csv": str(out_dir / "observed_assignment_metrics.csv"),
            "seed_observed_assignment_metrics_csv": str(
                out_dir / "seed_observed_assignment_metrics.csv"
            ),
            "conditional_resample_metrics_csv": str(out_dir / "conditional_resample_metrics.csv"),
            "conditional_resample_summary_csv": str(out_dir / "conditional_resample_summary.csv"),
            "null_quality_csv": str(out_dir / "null_quality.csv"),
            "functional_same_student_reference_csv": str(
                out_dir / "functional_same_student_reference.csv"
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
    _write_csv(out_dir / "observed_assignment_metrics.csv", [observed_row])
    _write_csv(out_dir / "seed_observed_assignment_metrics.csv", seed_observed_rows)
    _write_csv(out_dir / "conditional_resample_metrics.csv", resample_rows)
    _write_csv(out_dir / "conditional_resample_summary.csv", resample_summary_rows)
    _write_csv(out_dir / "null_quality.csv", null_quality_rows)
    _write_csv(out_dir / "functional_same_student_reference.csv", functional_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_seed(seed: int, path: Path) -> dict[str, Any]:
    packet = _read_json_object(path / "summary.json")
    audit = packet.get("audit", {}) if isinstance(packet.get("audit"), dict) else {}
    expected_present = all((path / name).is_file() for name in EXPECTED_FILES)
    token_rows = [_token_row(seed, row) for row in _read_csv_rows(path / "per_token_supports.csv")]
    return {
        "seed": seed,
        "source_row": {
            "seed": seed,
            "path": str(path),
            "present": path.is_dir(),
            "expected_files_present": expected_present,
            "status": packet.get("status"),
            "decision": packet.get("decision"),
            "claim_status": packet.get("claim_status"),
            "fold_count": audit.get("fold_count"),
            "dataset": audit.get("dataset"),
            "support_router": audit.get("support_router"),
            "top_k": audit.get("top_k"),
            "positions": len(token_rows),
            "has_token_position_null_same_student_arm": any(
                row.get("token_position_null_support") for row in token_rows
            ),
            "git_commit": packet.get("git_commit"),
        },
        "token_rows": token_rows,
    }


def _token_row(seed: int, row: dict[str, str]) -> dict[str, Any]:
    student_loss = _float(row.get("student_router_support_loss"))
    teacher_loss = _float(row.get("teacher_support_forced_into_student_loss"))
    null_loss = _float(row.get("token_position_null_support_forced_into_student_loss"))
    oracle_loss = _float(row.get("oracle_best_support_for_student_loss"))
    return {
        "seed": seed,
        "fold": int(row.get("fold", 0)),
        "flat_position": int(row.get("flat_position", 0)),
        "target_token": int(row.get("target_token", 0)),
        "teacher_support": _support_key(row.get("teacher_support")),
        "student_support": _support_key(row.get("student_support")),
        "oracle_support": _support_key(row.get("oracle_support")),
        "token_position_null_support": _support_key(row.get("token_position_null_support")),
        "student_loss": student_loss,
        "teacher_forced_loss": teacher_loss,
        "token_position_null_forced_loss": null_loss,
        "oracle_loss": oracle_loss,
        "teacher_forced_gain": student_loss - teacher_loss,
        "token_position_null_forced_gain": student_loss - null_loss,
        "teacher_minus_token_position_null_gain": null_loss - teacher_loss,
        "teacher_student_exact_pair_match": str(
            row.get("teacher_student_exact_pair_match", "")
        ).lower()
        == "true",
    }


def _observed_assignment_row(rows: list[dict[str, Any]], seed: int | str) -> dict[str, Any]:
    return {
        "seed": seed,
        "positions": len(rows),
        "student_exact_agreement": _mean_bool(
            [row["teacher_support"] == row["student_support"] for row in rows]
        ),
        "oracle_exact_agreement": _mean_bool(
            [row["teacher_support"] == row["oracle_support"] for row in rows]
        ),
        "token_position_null_exact_agreement": _mean_bool(
            [
                row["teacher_support"] == row["token_position_null_support"]
                for row in rows
                if row.get("token_position_null_support")
            ]
        ),
        "unique_teacher_supports": len({row["teacher_support"] for row in rows}),
        "unique_student_supports": len({row["student_support"] for row in rows}),
        "unique_oracle_supports": len({row["oracle_support"] for row in rows}),
    }


def _resample_seed(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    target_position = _index(rows, ("target_token", "flat_position"))
    target_only = _index(rows, ("target_token",))
    position_only = _index(rows, ("flat_position",))
    output = []
    for resample_index in range(resamples):
        sampled = []
        modes = []
        for index, row in enumerate(rows):
            candidates, mode = _candidates(
                index=index,
                row=row,
                rows=rows,
                target_position=target_position,
                target_only=target_only,
                position_only=position_only,
            )
            sampled_support = rng.choice(candidates)["teacher_support"]
            sampled.append((row, sampled_support))
            modes.append(mode)
        output.append(
            {
                "seed": seed,
                "resample": resample_index,
                "positions": len(sampled),
                "student_exact_agreement": _mean_bool(
                    [support == row["student_support"] for row, support in sampled]
                ),
                "oracle_exact_agreement": _mean_bool(
                    [support == row["oracle_support"] for row, support in sampled]
                ),
                "teacher_exact_agreement": _mean_bool(
                    [support == row["teacher_support"] for row, support in sampled]
                ),
                "target_position_fraction": modes.count("target_position") / len(modes),
                "target_only_fraction": modes.count("target_only") / len(modes),
                "position_only_fraction": modes.count("position_only") / len(modes),
                "global_fraction": modes.count("global") / len(modes),
            }
        )
    return output


def _candidates(
    *,
    index: int,
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    target_position: dict[tuple[Any, ...], list[int]],
    target_only: dict[tuple[Any, ...], list[int]],
    position_only: dict[tuple[Any, ...], list[int]],
) -> tuple[list[dict[str, Any]], str]:
    keys = [
        ("target_position", target_position, (row["target_token"], row["flat_position"])),
        ("target_only", target_only, (row["target_token"],)),
        ("position_only", position_only, (row["flat_position"],)),
    ]
    for mode, lookup, key in keys:
        indices = [item for item in lookup.get(key, []) if item != index]
        if indices:
            return [rows[item] for item in indices], mode
    return [candidate for pos, candidate in enumerate(rows) if pos != index], "global"


def _index(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[int]]:
    output: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        output[tuple(row[key] for key in keys)].append(index)
    return output


def _resample_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scope": "all_resamples",
            "resamples": len(rows),
            "mean_student_exact_agreement": _mean(
                [row["student_exact_agreement"] for row in rows]
            ),
            "std_student_exact_agreement": _std(
                [row["student_exact_agreement"] for row in rows]
            ),
            "mean_oracle_exact_agreement": _mean(
                [row["oracle_exact_agreement"] for row in rows]
            ),
            "std_oracle_exact_agreement": _std(
                [row["oracle_exact_agreement"] for row in rows]
            ),
            "mean_teacher_exact_agreement": _mean(
                [row["teacher_exact_agreement"] for row in rows]
            ),
        }
    ]


def _null_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scope": "all_resamples",
            "resamples": len(rows),
            "mean_target_position_fraction": _mean(
                [row["target_position_fraction"] for row in rows]
            ),
            "mean_target_only_fraction": _mean([row["target_only_fraction"] for row in rows]),
            "mean_position_only_fraction": _mean(
                [row["position_only_fraction"] for row in rows]
            ),
            "mean_global_fraction": _mean([row["global_fraction"] for row in rows]),
        }
    ]


def _functional_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all_tokens": rows,
        "teacher_student_disagreement_tokens": [
            row for row in rows if not row["teacher_student_exact_pair_match"]
        ],
    }
    output = []
    for subset, group in groups.items():
        output.append(
            {
                "token_subset": subset,
                "positions": len(group),
                "teacher_forced_gain": _mean([row["teacher_forced_gain"] for row in group]),
                "token_position_null_forced_gain": _mean(
                    [row["token_position_null_forced_gain"] for row in group]
                ),
                "teacher_minus_token_position_null_gain": _mean(
                    [row["teacher_minus_token_position_null_gain"] for row in group]
                ),
            }
        )
    return output


def _key_metrics(
    *,
    observed_row: dict[str, Any],
    resample_rows: list[dict[str, Any]],
    functional_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    null_student = [row["student_exact_agreement"] for row in resample_rows]
    null_oracle = [row["oracle_exact_agreement"] for row in resample_rows]
    all_tokens = {
        row["token_subset"]: row for row in functional_rows
    }.get("all_tokens", {})
    disagreement = {
        row["token_subset"]: row for row in functional_rows
    }.get("teacher_student_disagreement_tokens", {})
    return {
        "observed_student_exact_agreement": observed_row["student_exact_agreement"],
        "null_mean_student_exact_agreement": _mean(null_student),
        "student_exact_agreement_effect_vs_null_mean": (
            observed_row["student_exact_agreement"] - _mean(null_student)
        ),
        "student_exact_agreement_empirical_p_upper": _empirical_p_upper(
            observed_row["student_exact_agreement"],
            null_student,
        ),
        "observed_oracle_exact_agreement": observed_row["oracle_exact_agreement"],
        "null_mean_oracle_exact_agreement": _mean(null_oracle),
        "oracle_exact_agreement_effect_vs_null_mean": (
            observed_row["oracle_exact_agreement"] - _mean(null_oracle)
        ),
        "oracle_exact_agreement_empirical_p_upper": _empirical_p_upper(
            observed_row["oracle_exact_agreement"],
            null_oracle,
        ),
        "teacher_forced_gain_all_tokens": all_tokens.get("teacher_forced_gain"),
        "token_position_null_forced_gain_all_tokens": all_tokens.get(
            "token_position_null_forced_gain"
        ),
        "teacher_minus_token_position_null_gain_all_tokens": all_tokens.get(
            "teacher_minus_token_position_null_gain"
        ),
        "teacher_forced_gain_disagreement_tokens": disagreement.get("teacher_forced_gain"),
        "token_position_null_forced_gain_disagreement_tokens": disagreement.get(
            "token_position_null_forced_gain"
        ),
        "teacher_minus_token_position_null_gain_disagreement_tokens": disagreement.get(
            "teacher_minus_token_position_null_gain"
        ),
    }


def _criteria(
    source_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    resample_rows: list[dict[str, Any]],
    key_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _criterion(
            "all_local_seed_sources_present",
            all(row["expected_files_present"] for row in source_rows),
            "summary, per-token, null-control, and sampling artifacts present",
            [(row["seed"], row["expected_files_present"]) for row in source_rows],
        ),
        _criterion(
            "per_token_rows_present",
            len(token_rows) > 0,
            "at least one per-token row",
            len(token_rows),
        ),
        _criterion(
            "token_position_null_same_student_arm_present",
            all(row["has_token_position_null_same_student_arm"] for row in source_rows),
            "token_position_null_support and forced loss present",
            [(row["seed"], row["has_token_position_null_same_student_arm"]) for row in source_rows],
        ),
        _criterion(
            "conditional_resamples_present",
            len(resample_rows) > 0,
            "at least one conditional resample row",
            len(resample_rows),
        ),
        _criterion(
            "functional_reference_present",
            key_metrics.get("teacher_minus_token_position_null_gain_all_tokens") is not None,
            "teacher-minus-token-position-null same-student gain is computable",
            key_metrics.get("teacher_minus_token_position_null_gain_all_tokens"),
        ),
    ]


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _supported_claim(assignment_passes: bool) -> str | None:
    if assignment_passes:
        return (
            "Teacher support labels show above-null exact agreement with the trained "
            "student support under post-hoc conditional resampling within token/position "
            "strata; this is support-assignment evidence only."
        )
    return None


def _rationale(
    status: str,
    key_metrics: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    if status == "fail":
        return "Conditional-permutation report failed because required local source artifacts are missing or invalid."
    return (
        "The conditional-permutation matrix estimates support-label assignment ranks "
        "from existing per-token artifacts. It does not create new forced-loss "
        "evaluations for each sampled support assignment, so the functional mechanism "
        "claim remains governed by the same-student teacher-vs-token-position-null "
        f"gain ({key_metrics.get('teacher_minus_token_position_null_gain_all_tokens')})."
    )


def _support_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        return ",".join(str(item) for item in sorted(int(part) for part in parts))
    except ValueError:
        return text


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
        "# Conditional-Permutation Resample Matrix",
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


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _empirical_p_upper(observed: float | None, null_values: list[float]) -> float | None:
    if observed is None or not null_values:
        return None
    return (1 + sum(value >= observed for value in null_values)) / (len(null_values) + 1)


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
    parser.add_argument("--resamples", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=4309)
    args = parser.parse_args()
    summary = run_causal_contextual_router_conditional_permutation_resample_matrix(
        local_audit_dirs=args.audit_dirs,
        out_dir=args.out,
        resamples=args.resamples,
        random_seed=args.random_seed,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
