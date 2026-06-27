"""Conditional token/position-vs-context ablation for teacher-support distillation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_causal_contextual_router_conditional_ablation"
)

EXPECTED_FILES = [
    "summary.json",
    "per_token_supports.csv",
    "null_control_metrics.csv",
    "null_sampling_diagnostics.csv",
]

TARGET_POSITION_CONFOUND = "token_position_support_confound_dominates_context_lookup"
CONTEXT_RETAINED = "causal_history_context_signal_retained_after_target_position_check"
INSUFFICIENT = "insufficient_conditional_ablation_evidence"


def run_causal_contextual_router_conditional_ablation(
    *,
    local_audit_dirs: list[Path] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_target_position_edge: float = 0.05,
) -> dict[str, Any]:
    start = time.time()
    audit_dirs = local_audit_dirs or [
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed2"),
        Path("results/audits/token_larger_causal_contextual_router_distillation_agreement_seed3"),
    ]
    sources = [_load_seed(seed, path) for seed, path in zip([1, 2, 3], audit_dirs, strict=True)]
    source_rows = [source["source_row"] for source in sources]
    token_rows = [row for source in sources for row in source["token_rows"]]
    prediction_rows = [
        row
        for source in sources
        for row in _leave_fold_out_predictions(source["token_rows"])
    ]
    feature_rows = _aggregate_prediction_rows(prediction_rows)
    seed_feature_rows = _aggregate_prediction_rows(prediction_rows, keys=("seed", "feature_family"))
    gain_slice_rows = _gain_slice_rows(prediction_rows)
    comparisons = _key_comparisons(feature_rows)
    criteria = _criteria(
        source_rows=source_rows,
        token_rows=token_rows,
        feature_rows=feature_rows,
        comparisons=comparisons,
    )
    failures = [row for row in criteria if not row["passed"]]
    status = "fail" if failures else "pass"
    if failures:
        decision = "repair_missing_or_invalid_conditional_ablation_sources"
        claim_status = INSUFFICIENT
        selected_next_step = "repair_conditional_ablation_sources"
    elif comparisons["target_position_minus_causal_history_teacher_agreement"] >= min_target_position_edge:
        decision = "teacher_support_predictability_tracks_target_position_more_than_causal_history"
        claim_status = TARGET_POSITION_CONFOUND
        selected_next_step = "conditional_permutation_resample_matrix_before_runpod_repeat"
    else:
        decision = "causal_history_context_lookup_competes_with_target_position"
        claim_status = CONTEXT_RETAINED
        selected_next_step = "same_student_support_intervention_matrix_before_runpod_repeat"

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "experiment_id": "token_larger_causal_contextual_router_conditional_ablation",
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "source_rows": source_rows,
        "feature_rows": feature_rows,
        "seed_feature_rows": seed_feature_rows,
        "gain_slice_rows": gain_slice_rows,
        "key_comparisons": comparisons,
        "gate_status": {"criteria": criteria, "passes_conditional_ablation_gate": not failures},
        "failures": failures,
        "claim_boundaries": {
            "supported": _supported_claim(claim_status),
            "not_supported": [
                "deployable causal-router distillation mechanism",
                "RunPod validation of the stratified-null reversal",
                "functional support usefulness from support-label predictability alone",
            ],
        },
        "rationale": _rationale(claim_status, comparisons, failures),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "feature_ablation_metrics_csv": str(out_dir / "feature_ablation_metrics.csv"),
            "seed_feature_metrics_csv": str(out_dir / "seed_feature_metrics.csv"),
            "token_position_gain_slices_csv": str(out_dir / "token_position_gain_slices.csv"),
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
    _write_csv(out_dir / "feature_ablation_metrics.csv", feature_rows)
    _write_csv(out_dir / "seed_feature_metrics.csv", seed_feature_rows)
    _write_csv(out_dir / "token_position_gain_slices.csv", gain_slice_rows)
    _write_csv(out_dir / "gate_criteria.csv", criteria)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _load_seed(seed: int, path: Path) -> dict[str, Any]:
    packet = _read_json_object(path / "summary.json")
    audit = packet.get("audit", {}) if isinstance(packet.get("audit"), dict) else {}
    expected_present = all((path / name).is_file() for name in EXPECTED_FILES)
    rows = _read_csv_rows(path / "per_token_supports.csv")
    token_rows = [_token_row(seed, row) for row in rows]
    return {
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
            "git_commit": packet.get("git_commit"),
        },
        "token_rows": _attach_target_context(token_rows),
    }


def _token_row(seed: int, row: dict[str, str]) -> dict[str, Any]:
    student_loss = _float(row.get("student_router_support_loss"))
    teacher_loss = _float(row.get("teacher_support_forced_into_student_loss"))
    oracle_loss = _float(row.get("oracle_best_support_for_student_loss"))
    return {
        "seed": seed,
        "fold": int(row["fold"]),
        "flat_position": int(row["flat_position"]),
        "target_token": int(row["target_token"]),
        "teacher_support": _support_key(row.get("teacher_support")),
        "student_support": _support_key(row.get("student_support")),
        "oracle_support": _support_key(row.get("oracle_support")),
        "student_loss": student_loss,
        "teacher_forced_loss": teacher_loss,
        "oracle_loss": oracle_loss,
        "teacher_forced_gain": student_loss - teacher_loss,
        "student_oracle_gap": student_loss - oracle_loss,
        "teacher_student_exact_pair_match": str(
            row.get("teacher_student_exact_pair_match", "")
        ).lower()
        == "true",
    }


def _attach_target_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fold = defaultdict(dict)
    for row in rows:
        by_fold[row["fold"]][row["flat_position"]] = row
    enriched = []
    for row in rows:
        fold_rows = by_fold[row["fold"]]
        pos = row["flat_position"]
        previous_target = fold_rows.get(pos - 1, {}).get("target_token")
        next_target = fold_rows.get(pos + 1, {}).get("target_token")
        enriched.append(
            {
                **row,
                "previous_target_token": previous_target,
                "next_target_token": next_target,
                "position_bin": _position_bin(pos),
            }
        )
    return enriched


def _leave_fold_out_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    folds = sorted({row["fold"] for row in rows})
    for fold in folds:
        train_rows = [row for row in rows if row["fold"] != fold]
        holdout_rows = [row for row in rows if row["fold"] == fold]
        for feature_family in _feature_families():
            lookup = _majority_lookup(train_rows, feature_family)
            global_support = _global_majority(train_rows)
            for row in holdout_rows:
                key = _feature_key(row, feature_family)
                predicted = lookup.get(key, global_support)
                used_fallback = key not in lookup
                predictions.append(
                    {
                        "seed": row["seed"],
                        "fold": fold,
                        "feature_family": feature_family,
                        "flat_position": row["flat_position"],
                        "target_token": row["target_token"],
                        "position_bin": row["position_bin"],
                        "feature_key": key,
                        "predicted_support": predicted,
                        "teacher_support": row["teacher_support"],
                        "student_support": row["student_support"],
                        "oracle_support": row["oracle_support"],
                        "teacher_exact_match": predicted == row["teacher_support"],
                        "student_exact_match": predicted == row["student_support"],
                        "oracle_exact_match": predicted == row["oracle_support"],
                        "used_global_fallback": used_fallback,
                        "teacher_forced_gain": row["teacher_forced_gain"],
                        "student_oracle_gap": row["student_oracle_gap"],
                        "teacher_student_exact_pair_match": row[
                            "teacher_student_exact_pair_match"
                        ],
                    }
                )
    return predictions


def _feature_families() -> list[str]:
    return [
        "global_majority",
        "position_only",
        "position_bin_only",
        "causal_history_position",
        "target_only",
        "target_position",
        "target_history_position",
        "target_local_future_position",
    ]


def _feature_key(row: dict[str, Any], feature_family: str) -> str:
    if feature_family == "global_majority":
        return "global"
    if feature_family == "position_only":
        return f"p={row['flat_position']}"
    if feature_family == "position_bin_only":
        return f"pb={row['position_bin']}"
    if feature_family == "causal_history_position":
        return f"prev={row['previous_target_token']}"
    if feature_family == "target_only":
        return f"t={row['target_token']}"
    if feature_family == "target_position":
        return f"t={row['target_token']}|p={row['flat_position']}"
    if feature_family == "target_history_position":
        return (
            f"prev={row['previous_target_token']}|t={row['target_token']}|"
            f"p={row['flat_position']}"
        )
    if feature_family == "target_local_future_position":
        return (
            f"prev={row['previous_target_token']}|t={row['target_token']}|"
            f"next={row['next_target_token']}|p={row['flat_position']}"
        )
    raise ValueError(f"unknown feature family: {feature_family}")


def _majority_lookup(rows: list[dict[str, Any]], feature_family: str) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[_feature_key(row, feature_family)][row["teacher_support"]] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def _global_majority(rows: list[dict[str, Any]]) -> str:
    return Counter(row["teacher_support"] for row in rows).most_common(1)[0][0]


def _aggregate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ("feature_family",),
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(groups.items()):
        teacher_matches = [bool(row["teacher_exact_match"]) for row in group_rows]
        student_matches = [bool(row["student_exact_match"]) for row in group_rows]
        oracle_matches = [bool(row["oracle_exact_match"]) for row in group_rows]
        fallbacks = [bool(row["used_global_fallback"]) for row in group_rows]
        matched_gains = [
            float(row["teacher_forced_gain"])
            for row in group_rows
            if row["teacher_exact_match"]
        ]
        unmatched_gains = [
            float(row["teacher_forced_gain"])
            for row in group_rows
            if not row["teacher_exact_match"]
        ]
        row = {key: value for key, value in zip(keys, group_key, strict=True)}
        row.update(
            {
                "positions": len(group_rows),
                "teacher_exact_agreement": _mean_bool(teacher_matches),
                "student_exact_agreement": _mean_bool(student_matches),
                "oracle_exact_agreement": _mean_bool(oracle_matches),
                "global_fallback_fraction": _mean_bool(fallbacks),
                "mean_teacher_forced_gain": _mean(
                    [float(item["teacher_forced_gain"]) for item in group_rows]
                ),
                "mean_student_oracle_gap": _mean(
                    [float(item["student_oracle_gap"]) for item in group_rows]
                ),
                "mean_teacher_gain_when_prediction_matches": _mean(matched_gains),
                "mean_teacher_gain_when_prediction_misses": _mean(unmatched_gains),
                "unique_predicted_supports": len(
                    {str(item["predicted_support"]) for item in group_rows}
                ),
            }
        )
        output.append(row)
    return output


def _gain_slice_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {"target_position", "causal_history_position", "target_only"}
    output = []
    for row in _aggregate_prediction_rows(
        [row for row in rows if row["feature_family"] in selected],
        keys=("feature_family", "teacher_exact_match"),
    ):
        output.append(row)
    return output


def _key_comparisons(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_feature = {row["feature_family"]: row for row in feature_rows}
    target_position = by_feature.get("target_position", {})
    causal_history = by_feature.get("causal_history_position", {})
    target_only = by_feature.get("target_only", {})
    local_future = by_feature.get("target_local_future_position", {})
    return {
        "target_position_teacher_agreement": target_position.get(
            "teacher_exact_agreement"
        ),
        "causal_history_teacher_agreement": causal_history.get("teacher_exact_agreement"),
        "target_only_teacher_agreement": target_only.get("teacher_exact_agreement"),
        "target_local_future_teacher_agreement": local_future.get(
            "teacher_exact_agreement"
        ),
        "target_position_minus_causal_history_teacher_agreement": _delta(
            target_position.get("teacher_exact_agreement"),
            causal_history.get("teacher_exact_agreement"),
        ),
        "target_position_minus_target_only_teacher_agreement": _delta(
            target_position.get("teacher_exact_agreement"),
            target_only.get("teacher_exact_agreement"),
        ),
        "local_future_minus_target_position_teacher_agreement": _delta(
            local_future.get("teacher_exact_agreement"),
            target_position.get("teacher_exact_agreement"),
        ),
        "target_position_global_fallback_fraction": target_position.get(
            "global_fallback_fraction"
        ),
        "causal_history_global_fallback_fraction": causal_history.get(
            "global_fallback_fraction"
        ),
    }


def _criteria(
    *,
    source_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> list[dict[str, Any]]:
    required_features = set(_feature_families())
    present_features = {row["feature_family"] for row in feature_rows}
    return [
        _criterion(
            "all_local_seed_artifacts_present",
            all(row["expected_files_present"] for row in source_rows),
            "all expected strengthened-null source files exist",
            [row["expected_files_present"] for row in source_rows],
        ),
        _criterion(
            "all_sources_four_fold",
            all(row.get("fold_count") == 4 for row in source_rows),
            "each source has four folds",
            [(row.get("seed"), row.get("fold_count")) for row in source_rows],
        ),
        _criterion(
            "per_token_rows_present",
            len(token_rows) > 0,
            "per-token support rows are available",
            len(token_rows),
        ),
        _criterion(
            "all_feature_families_evaluated",
            required_features.issubset(present_features),
            sorted(required_features),
            sorted(present_features),
        ),
        _criterion(
            "target_position_and_causal_history_comparable",
            comparisons.get("target_position_teacher_agreement") is not None
            and comparisons.get("causal_history_teacher_agreement") is not None,
            "target-position and causal-history rows are both present",
            comparisons,
        ),
    ]


def _criterion(name: str, passed: bool, threshold: Any, actual: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "threshold": threshold, "actual": actual}


def _supported_claim(claim_status: str) -> str | None:
    if claim_status == TARGET_POSITION_CONFOUND:
        return (
            "teacher support identity is more recoverable from target-token/position "
            "lookup than from causal-history/position lookup in the current artifacts"
        )
    if claim_status == CONTEXT_RETAINED:
        return (
            "causal-history/position lookup remains competitive with target-token/"
            "position lookup as a teacher-support predictor"
        )
    return None


def _rationale(
    claim_status: str,
    comparisons: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    if failures:
        return "Conditional ablation failed closed because required source artifacts were missing or invalid."
    edge = comparisons["target_position_minus_causal_history_teacher_agreement"]
    if claim_status == TARGET_POSITION_CONFOUND:
        return (
            "The target-position lookup has higher leave-fold-out teacher-support "
            f"agreement than causal-history/position lookup (edge {edge:.6f}), so "
            "the stronger-null reversal remains the active interpretation."
        )
    return (
        "The causal-history/position lookup is competitive with target-position "
        f"lookup (edge {edge:.6f}); support usefulness still needs a same-student "
        "intervention matrix before any renewed mechanism claim."
    )


def _position_bin(position: int, bins: int = 4) -> int:
    return int(position % bins)


def _support_key(value: str | None) -> str:
    if value is None:
        return ""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    try:
        return ",".join(str(item) for item in sorted(int(part) for part in parts))
    except ValueError:
        return str(value)


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
        "# Conditional Token/Position-vs-Context Ablation",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        f"Claim status: `{summary['claim_status']}`",
        f"Selected next step: `{summary['selected_next_step']}`",
        "",
        "## Key comparisons",
    ]
    for key, value in summary["key_comparisons"].items():
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


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


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
    summary = run_causal_contextual_router_conditional_ablation(
        local_audit_dirs=args.audit_dirs,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
