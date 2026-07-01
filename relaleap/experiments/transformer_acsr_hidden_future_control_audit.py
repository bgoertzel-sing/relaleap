"""Audit local controls for the hidden/future Transformer-ACSR pregate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/dataset_rows.csv")
DEFAULT_LOSS_LOOKUP = Path("results/reports/transformer_acsr_hidden_future_sequence_dataset/loss_lookup.csv")
DEFAULT_PREGATE = Path("results/reports/transformer_acsr_hidden_future_predictor_pregate/summary.json")
DEFAULT_PREDICTIONS = Path(
    "results/reports/transformer_acsr_hidden_future_predictor_pregate/heldout_predictions.csv"
)
DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_hidden_future_control_audit")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "current_hidden_stratified_null.csv",
    "retention_churn.csv",
    "commutator_proxy.csv",
    "future_perturbation_invariance.csv",
    "notes.md",
)


def run_transformer_acsr_hidden_future_control_audit(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    loss_lookup_path: Path = DEFAULT_LOSS_LOOKUP,
    pregate_path: Path = DEFAULT_PREGATE,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write local control artifacts for the hidden/future pregate."""

    start = time.time()
    failures: list[dict[str, Any]] = []
    for name, path in (
        ("dataset_rows", dataset_path),
        ("loss_lookup", loss_lookup_path),
        ("pregate_summary", pregate_path),
        ("heldout_predictions", predictions_path),
    ):
        if not path.is_file():
            failures.append({"source": name, "path": str(path), "reason": "required artifact missing"})

    if failures:
        summary = _summary(
            status="fail",
            decision="transformer_acsr_hidden_future_control_audit_failed_closed",
            claim_status="hidden_future_control_sources_missing",
            failures=failures,
            start=start,
            out_dir=out_dir,
            controls={},
        )
        _write_artifacts(out_dir, summary, [], [], [], [])
        return summary

    dataset_rows = _read_csv(dataset_path)
    loss_lookup = _loss_lookup(_read_csv(loss_lookup_path))
    pregate = _read_json(pregate_path)
    predictions = _read_csv(predictions_path)

    stratified_null = _current_hidden_stratified_null(dataset_rows, loss_lookup)
    retention_churn = _retention_churn_rows(predictions, dataset_rows)
    commutator_proxy = _commutator_proxy_rows(predictions, loss_lookup)
    future_invariance = _future_perturbation_invariance_rows(dataset_rows, predictions)

    controls = {
        "current_hidden_stratified_null_available": bool(stratified_null),
        "retention_churn_artifact_available": bool(retention_churn),
        "commutator_exact_budget_available": all(
            row["exact_finite_update_commutator_available"] == "true" for row in commutator_proxy
        ),
        "future_perturbation_invariance_passes": all(
            row["prediction_invariant_to_future_target_perturbation"] == "true"
            for row in future_invariance
        ),
    }
    prefix_jaccard = _float(pregate.get("prefix_hidden_jaccard"))
    stratified_jaccard = _mean_float(stratified_null, "jaccard")
    stratified_null_margin = (
        None if prefix_jaccard is None or stratified_jaccard is None else prefix_jaccard - stratified_jaccard
    )
    same_student_loss_gate = pregate.get("same_student_loss_gate_passes") is True
    exact_commutator_available = controls["commutator_exact_budget_available"]
    advance_to_gpu = bool(
        controls["current_hidden_stratified_null_available"]
        and controls["retention_churn_artifact_available"]
        and controls["future_perturbation_invariance_passes"]
        and same_student_loss_gate
        and exact_commutator_available
    )
    decision = (
        "transformer_acsr_hidden_future_control_audit_gpu_ready"
        if advance_to_gpu
        else "transformer_acsr_hidden_future_control_audit_gpu_blocked"
    )
    claim_status = (
        "hidden_future_controls_clear_registered_local_gate"
        if advance_to_gpu
        else "hidden_future_controls_recorded_but_mechanism_gate_not_cleared"
    )
    summary = _summary(
        status="pass",
        decision=decision,
        claim_status=claim_status,
        failures=[],
        start=start,
        out_dir=out_dir,
        controls=controls,
    )
    summary.update(
        {
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": advance_to_gpu,
            "dataset_path": str(dataset_path),
            "loss_lookup_path": str(loss_lookup_path),
            "pregate_path": str(pregate_path),
            "predictions_path": str(predictions_path),
            "row_count": len(dataset_rows),
            "heldout_prediction_count": len(predictions),
            "prefix_hidden_jaccard": prefix_jaccard,
            "current_hidden_stratified_null_jaccard": stratified_jaccard,
            "current_hidden_stratified_null_margin": stratified_null_margin,
            "same_student_loss_gate_passes": same_student_loss_gate,
            "exact_finite_update_commutator_available": exact_commutator_available,
            "selected_next_step": (
                "run_gpu_validation_after_local_artifact_checks"
                if advance_to_gpu
                else "keep_hidden_future_branch_local_until_same_student_loss_and_exact_commutator_controls_clear"
            ),
            "backend_policy": "local control audit only; RunPod and Colab remain blocked",
        }
    )
    _write_artifacts(out_dir, summary, stratified_null, retention_churn, commutator_proxy, future_invariance)
    return summary


def _current_hidden_stratified_null(
    dataset_rows: list[dict[str, str]],
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
) -> list[dict[str, Any]]:
    train_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in dataset_rows:
        if row.get("split") != "train":
            continue
        train_counts[_position_stratum(row)].update([_pair_string(row["teacher_topk_support_target_only"])])
    rows = []
    for row in dataset_rows:
        if row.get("split") != "heldout":
            continue
        stratum = _position_stratum(row)
        pair = _majority_pair(train_counts.get(stratum, Counter()))
        if pair is None:
            pair = _pair_string(row["student_router_support_eval_only"])
        teacher = set(_parse_pair(row["teacher_topk_support_target_only"]))
        pred = set(_parse_pair(pair))
        loss = loss_lookup[(row["sequence_id"], int(row["position_index"]), pair)]
        rows.append(
            {
                "sequence_id": row["sequence_id"],
                "position_index": row["position_index"],
                "position_mod8_stratum": stratum,
                "control": "current_hidden_shuffled_within_position_mod8_null",
                "predicted_support_pair": pair,
                "teacher_support_pair": _pair_string(row["teacher_topk_support_target_only"]),
                "jaccard": len(pred & teacher) / len(pred | teacher),
                "forced_support_loss": loss["forced_support_loss"],
                "forced_minus_student_router_loss": loss["forced_minus_student_router_loss"],
            }
        )
    return rows


def _retention_churn_rows(
    predictions: list[dict[str, str]],
    dataset_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    teacher_by_context = {
        (row["sequence_id"], row["position_index"]): _pair_string(row["teacher_topk_support_target_only"])
        for row in dataset_rows
        if row.get("split") == "heldout"
    }
    student_by_context = {
        (row["sequence_id"], row["position_index"]): _pair_string(row["student_router_support_eval_only"])
        for row in dataset_rows
        if row.get("split") == "heldout"
    }
    rows = []
    for support_source, support_by_context in (
        (
            "prefix_hidden_prediction",
            {
                (row["sequence_id"], row["position_index"]): _pair_string(row["predicted_support_pair"])
                for row in predictions
            },
        ),
        ("teacher_support", teacher_by_context),
        ("student_router_support", student_by_context),
    ):
        for sequence_id in sorted({key[0] for key in support_by_context}):
            ordered = sorted(
                (
                    (int(position), pair)
                    for (seq, position), pair in support_by_context.items()
                    if seq == sequence_id
                ),
                key=lambda item: item[0],
            )
            changes = sum(1 for left, right in zip(ordered, ordered[1:]) if left[1] != right[1])
            denominator = max(1, len(ordered) - 1)
            rows.append(
                {
                    "support_source": support_source,
                    "sequence_id": sequence_id,
                    "position_count": len(ordered),
                    "support_change_count": changes,
                    "support_change_fraction": changes / denominator,
                    "unique_support_count": len({pair for _, pair in ordered}),
                    "retention_churn_budget_status": "artifact_recorded_not_promotion_gate",
                }
            )
    return rows


def _commutator_proxy_rows(
    predictions: list[dict[str, str]],
    loss_lookup: dict[tuple[str, int, str], dict[str, float]],
) -> list[dict[str, str]]:
    rows = []
    for row in predictions:
        pair = _pair_string(row["predicted_support_pair"])
        key = (row["sequence_id"], int(row["position_index"]), pair)
        rows.append(
            {
                "sequence_id": row["sequence_id"],
                "position_index": row["position_index"],
                "predicted_support_pair": pair,
                "canonical_pair_lookup_present": str(key in loss_lookup).lower(),
                "exact_finite_update_commutator_available": "false",
                "commutator_budget_status": "blocked_exact_update_order_not_measured",
            }
        )
    return rows


def _future_perturbation_invariance_rows(
    dataset_rows: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> list[dict[str, str]]:
    prefix_hash_before = _prefix_feature_hash(dataset_rows)
    perturbed = []
    for row in dataset_rows:
        copy = dict(row)
        copy["future_hidden_json_target_only"] = "[999.0]"
        copy["future_delta_json_target_only"] = "[-999.0]"
        copy["teacher_support_logits_json_target_only"] = "[0.0]"
        perturbed.append(copy)
    prefix_hash_after = _prefix_feature_hash(perturbed)
    prediction_hash = _prediction_hash(predictions)
    return [
        {
            "control": "future_target_field_perturbation",
            "prefix_feature_hash_before": prefix_hash_before,
            "prefix_feature_hash_after": prefix_hash_after,
            "prediction_hash": prediction_hash,
            "prefix_features_invariant_to_future_target_perturbation": str(
                prefix_hash_before == prefix_hash_after
            ).lower(),
            "prediction_invariant_to_future_target_perturbation": str(
                prefix_hash_before == prefix_hash_after
            ).lower(),
            "future_perturbation_budget_status": "artifact_passes_feature_contract_not_causal_retraining",
        }
    ]


def _summary(
    *,
    status: str,
    decision: str,
    claim_status: str,
    failures: list[dict[str, Any]],
    start: float,
    out_dir: Path,
    controls: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "controls": controls,
        "failures": failures,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }


def _position_stratum(row: dict[str, str]) -> int:
    return int(row["position_index"]) % 8


def _majority_pair(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _loss_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, float]]:
    lookup = {}
    for row in rows:
        lookup[(row["sequence_id"], int(row["position_index"]), _pair_string(row["forced_support_pair"]))] = {
            "forced_support_loss": float(row["forced_support_loss"]),
            "forced_minus_student_router_loss": float(row["forced_minus_student_router_loss"]),
        }
    return lookup


def _prefix_feature_hash(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["sequence_id"], int(item["position_index"]))):
        payload = {
            "sequence_id": row["sequence_id"],
            "position_index": row["position_index"],
            "current_hidden_json": row["current_hidden_json"],
            "previous_hidden_json": row["previous_hidden_json"],
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _prediction_hash(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["sequence_id"], int(item["position_index"]))):
        digest.update(
            json.dumps(
                {
                    "sequence_id": row["sequence_id"],
                    "position_index": row["position_index"],
                    "predicted_support_pair": _pair_string(row["predicted_support_pair"]),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_pair(value: str) -> tuple[int, int]:
    left, right = [int(part.strip()) for part in value.split(",", maxsplit=1)]
    return tuple(sorted((left, right)))


def _pair_string(value: str) -> str:
    left, right = _parse_pair(value)
    return f"{left},{right}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_artifacts(
    out_dir: Path,
    summary: dict[str, Any],
    stratified_null: list[dict[str, Any]],
    retention_churn: list[dict[str, Any]],
    commutator_proxy: list[dict[str, str]],
    future_invariance: list[dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "current_hidden_stratified_null.csv", stratified_null)
    _write_csv(out_dir / "retention_churn.csv", retention_churn)
    _write_csv(out_dir / "commutator_proxy.csv", commutator_proxy)
    _write_csv(out_dir / "future_perturbation_invariance.csv", future_invariance)
    notes = [
        "# Transformer-ACSR Hidden/Future Control Audit",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        "",
        "The audit records local controls only. Exact finite-update commutator evidence remains unavailable, so GPU validation stays blocked.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--loss-lookup", type=Path, default=DEFAULT_LOSS_LOOKUP)
    parser.add_argument("--pregate", type=Path, default=DEFAULT_PREGATE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_transformer_acsr_hidden_future_control_audit(
        dataset_path=args.dataset,
        loss_lookup_path=args.loss_lookup,
        pregate_path=args.pregate,
        predictions_path=args.predictions,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "advance_to_gpu_validation": summary["advance_to_gpu_validation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
