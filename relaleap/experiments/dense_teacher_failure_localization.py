"""Scaffold the dense-teacher failure-localization audit contract."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT_DIR = Path("results/reports/dense_teacher_columnability_closeout")
DEFAULT_DISTILLATION_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_OUT_DIR = Path("results/reports/dense_teacher_failure_localization")

CONTRACT_RECORDED = "dense_teacher_failure_localization_contract_recorded"
INSUFFICIENT_EVIDENCE = "dense_teacher_failure_localization_contract_failed_closed"
NEXT_STEP = (
    "implement tiny local dense_teacher_failure_localization evaluator that fills "
    "oracle_support_trained_values, retrained_oracle_support, gated_value_pair_composer, "
    "dense/rank/norm controls, and shuffled/random/token-position null rows"
)

REQUIRED_ARMS = (
    "learned_support_sparse_student",
    "oracle_support_trained_values",
    "retrained_oracle_support_values",
    "oracle_support_gated_value_pair_composer",
    "dense_teacher",
    "dense_rank_norm_control",
    "random_support_null",
    "fixed_support_null",
    "token_position_router_null",
    "shuffled_teacher_target_null",
)

REQUIRED_TENSORS = (
    {
        "tensor": "inputs",
        "filename": "inputs.pt",
        "required_for": "fixed validation batch identity and token/position nulls",
    },
    {
        "tensor": "targets",
        "filename": "targets.pt",
        "required_for": "CE guardrail and shuffled-target null checks",
    },
    {
        "tensor": "base_hidden",
        "filename": "base_hidden.pt",
        "required_for": "oracle/retrained sparse value evaluation against frozen hidden states",
    },
    {
        "tensor": "base_logits",
        "filename": "base_logits.pt",
        "required_for": "teacher logit residual and anchor/off-target leakage metrics",
    },
    {
        "tensor": "teacher_logits",
        "filename": "teacher_logits.pt",
        "required_for": "dense-teacher CE, logit residual, and shuffled-target controls",
    },
    {
        "tensor": "teacher_hidden_residual",
        "filename": "teacher_hidden_residual.pt",
        "required_for": "hidden residual MSE/cosine/R2 metrics",
    },
    {
        "tensor": "teacher_logit_residual",
        "filename": "teacher_logit_residual.pt",
        "required_for": "logit residual MSE/cosine/R2 metrics",
    },
    {
        "tensor": "learned_support_indices",
        "filename": "learned_support_indices.pt",
        "required_for": "learned-support baseline and oracle-support regret",
    },
    {
        "tensor": "learned_support_scores",
        "filename": "learned_support_scores.pt",
        "required_for": "router margin, support confidence, and token-position null calibration",
    },
    {
        "tensor": "per_column_hidden_contributions",
        "filename": "per_column_hidden_contributions.pt",
        "required_for": "exhaustive oracle support over trained values",
    },
    {
        "tensor": "per_column_logit_contributions",
        "filename": "per_column_logit_contributions.pt",
        "required_for": "oracle support logit residual and CE evaluation",
    },
    {
        "tensor": "sparse_column_value_state",
        "filename": "sparse_column_value_state.pt",
        "required_for": "retrained-oracle and gated pair-composer initialization/accounting",
    },
)

METRIC_FIELDS = (
    "arm",
    "arm_family",
    "availability",
    "support_source",
    "value_source",
    "composer",
    "train_steps",
    "active_params",
    "stored_params",
    "flops_proxy",
    "ce_loss",
    "teacher_hidden_residual_mse",
    "teacher_hidden_residual_r2",
    "teacher_hidden_residual_cosine",
    "teacher_logit_residual_mse",
    "teacher_logit_residual_r2",
    "teacher_logit_residual_cosine",
    "oracle_support_regret",
    "functional_churn",
    "anchor_kl",
    "offtarget_logit_leakage",
    "residual_norm_ratio",
    "residual_direction_error",
    "pair_synergy",
    "passes_no_gpu_pregate",
    "notes",
)


def run_dense_teacher_failure_localization_contract(
    *,
    closeout_dir: Path = DEFAULT_CLOSEOUT_DIR,
    distillation_dir: Path = DEFAULT_DISTILLATION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed contract for the next local localization evaluator."""

    start = time.time()
    closeout_path = closeout_dir / "summary.json"
    distillation_path = distillation_dir / "summary.json"
    closeout = _read_json(closeout_path)
    distillation = _read_json(distillation_path)
    tensor_inventory = _tensor_inventory(distillation_dir)
    source_rows = [
        _source_row("dense_teacher_columnability_closeout", closeout_path, closeout),
        _source_row("dense_teacher_distillation_comparison", distillation_path, distillation),
    ]
    source_rows.extend(
        _tensor_source_row(row["tensor"], Path(row["path"]), bool(row["present"]))
        for row in tensor_inventory
    )
    contract_rows = _contract_rows()
    pregate = _pregate_rows(closeout, distillation, source_rows, tensor_inventory)
    failures = [row for row in pregate if not row["passed"]]
    status = "fail" if failures else "pass"
    decision = INSUFFICIENT_EVIDENCE if failures else CONTRACT_RECORDED
    next_command = (
        "./.venv-conda/bin/python -m relaleap.experiments.dense_teacher_failure_localization"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": "contract_only_no_scientific_localization_claim",
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "selected_next_step": NEXT_STEP,
        "next_command": next_command,
        "source_rows": source_rows,
        "tensor_inventory": tensor_inventory,
        "pregate_rows": pregate,
        "contract_rows": contract_rows,
        "required_arms": list(REQUIRED_ARMS),
        "metric_fields": list(METRIC_FIELDS),
        "failures": failures,
        "rationale": (
            "This artifact records the local failure-localization audit contract selected by the "
            "dense-teacher closeout. It intentionally makes no new columnability claim. The next "
            "evaluator must separate support prediction, value representability, and pair/value "
            "composition using oracle-support, retrained-oracle, pair-composer, dense/rank/norm, "
            "and null rows before any GPU validation."
        ),
        "backend_policy": "local CPU contract only; RunPod and Colab are not used.",
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "tensor_inventory_csv": str(out_dir / "tensor_inventory.csv"),
            "pregate_rows_csv": str(out_dir / "pregate_rows.csv"),
            "contract_rows_csv": str(out_dir / "contract_rows.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    _write_artifacts(out_dir, summary)
    return summary


def _contract_rows() -> list[dict[str, Any]]:
    return [
        _contract(
            "learned_support_sparse_student",
            "baseline",
            "available_from_distillation_summary",
            "learned contextual/ACSR router support",
            "trained sparse values",
            "independent column sum",
            "anchors the current failed sparse branch",
        ),
        _contract(
            "oracle_support_trained_values",
            "oracle_support",
            "required_pending",
            "exhaustive per-token best top-k support",
            "trained sparse values",
            "independent column sum",
            "tests support-prediction regret while holding values fixed",
        ),
        _contract(
            "retrained_oracle_support_values",
            "oracle_value",
            "required_pending",
            "oracle support fixed during training/eval",
            "values retrained under oracle support",
            "independent column sum",
            "tests value representability after removing router error",
        ),
        _contract(
            "oracle_support_gated_value_pair_composer",
            "composition",
            "required_pending",
            "oracle support",
            "support-conditioned gates plus pair interaction",
            "gated values plus pair composer",
            "tests whether independent column summation is the bottleneck",
        ),
        _contract(
            "dense_teacher",
            "upper_bound",
            "available_from_distillation_summary",
            "dense",
            "dense teacher residual",
            "dense residual adapter",
            "records the target gap and guards against false sparse wins",
        ),
        _contract(
            "dense_rank_norm_control",
            "control",
            "required_pending",
            "dense/rank/norm matched",
            "matched dense or low-rank residual",
            "matched control",
            "prevents pair-composer wins from merely being extra dense capacity",
        ),
        _contract(
            "random_support_null",
            "null",
            "available_from_distillation_summary",
            "random top-k support",
            "trained sparse values",
            "independent column sum",
            "tests support specificity against utilization-matched random routing",
        ),
        _contract(
            "fixed_support_null",
            "null",
            "available_from_distillation_summary",
            "fixed top-k support",
            "trained sparse values",
            "independent column sum",
            "tests whether routing variation matters at all",
        ),
        _contract(
            "token_position_router_null",
            "null",
            "available_from_distillation_summary",
            "token/position-only support predictor",
            "trained sparse values",
            "independent column sum",
            "tests deployable support signal beyond token/position shortcuts",
        ),
        _contract(
            "shuffled_teacher_target_null",
            "null",
            "available_from_distillation_summary",
            "shuffled teacher target support",
            "shuffled residual/logit target",
            "independent column sum",
            "tests target-label leakage and noncausal teacher alignment",
        ),
    ]


def _contract(
    arm: str,
    arm_family: str,
    availability: str,
    support_source: str,
    value_source: str,
    composer: str,
    notes: str,
) -> dict[str, Any]:
    row = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "arm": arm,
            "arm_family": arm_family,
            "availability": availability,
            "support_source": support_source,
            "value_source": value_source,
            "composer": composer,
            "passes_no_gpu_pregate": "pending",
            "notes": notes,
        }
    )
    return row


def _pregate_rows(
    closeout: dict[str, Any],
    distillation: dict[str, Any],
    source_rows: list[dict[str, Any]],
    tensor_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    present = {row["source"]: row["present"] for row in source_rows}
    tensor_present = {row["tensor"]: bool(row["present"]) for row in tensor_inventory}
    variant_rows = distillation.get("variant_rows") if isinstance(distillation.get("variant_rows"), list) else []
    arms = {row.get("arm") for row in variant_rows if isinstance(row, dict)}
    return [
        _pregate(
            "closeout_branch_retired",
            closeout.get("decision") == "dense_teacher_sparse_columnability_branch_closed_for_failure_localization",
            "dense-teacher closeout must have retired the rescue branch",
            closeout.get("decision"),
        ),
        _pregate(
            "distillation_failed_closed",
            distillation.get("status") == "fail",
            "source distillation comparison must be negative evidence, not a promotion",
            distillation.get("status"),
        ),
        _pregate(
            "teacher_residual_tensors_present",
            bool(tensor_present.get("teacher_hidden_residual"))
            and bool(tensor_present.get("teacher_logit_residual")),
            "hidden and logit residual tensors are needed for residual-direction metrics",
            {
                "hidden": tensor_present.get("teacher_hidden_residual"),
                "logit": tensor_present.get("teacher_logit_residual"),
            },
        ),
        _pregate(
            "per_column_evaluator_tensors_present",
            all(bool(row["present"]) for row in tensor_inventory),
            "all required evaluator tensors must be exported before oracle/retrained/composer rows can run",
            {
                row["tensor"]: row["status"]
                for row in tensor_inventory
                if not row["present"]
            },
        ),
        _pregate(
            "baseline_and_null_rows_available",
            {
                "promoted_contextual_topk2_ce_mse_distill",
                "random_support_topk2",
                "fixed_support_topk2",
                "token_position_only_router_topk2",
                "shuffled_teacher_target_topk2",
            }.issubset(arms),
            "existing distillation artifact must expose baseline and null arms",
            sorted(arms),
        ),
        _pregate(
            "oracle_and_composer_rows_pending",
            True,
            "this scaffold records, but does not yet execute, oracle/composer rows",
            "pending evaluator implementation",
        ),
    ]


def _pregate(criterion: str, passed: bool, threshold: str, actual: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else "required source or contract condition missing",
    }


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status", ""),
        "decision": packet.get("decision", ""),
        "claim_status": packet.get("claim_status", ""),
        "git_commit": packet.get("git_commit", ""),
    }


def _tensor_inventory(distillation_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in REQUIRED_TENSORS:
        path = distillation_dir / str(spec["filename"])
        present = path.is_file()
        rows.append(
            {
                "tensor": spec["tensor"],
                "path": str(path),
                "required_for": spec["required_for"],
                "present": present,
                "file_size_bytes": path.stat().st_size if present else 0,
                "status": "present" if present else "missing_required_export",
            }
        )
    return rows


def _tensor_source_row(source: str, path: Path, present: bool) -> dict[str, Any]:
    return {
        "source": f"{source}_tensor",
        "path": str(path),
        "present": present,
        "status": "present" if present else "missing",
        "decision": "",
        "claim_status": "dense_teacher_failure_localization_tensor_source",
        "git_commit": "",
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "tensor_inventory.csv", summary["tensor_inventory"])
    _write_csv(out_dir / "pregate_rows.csv", summary["pregate_rows"])
    _write_csv(out_dir / "contract_rows.csv", summary["contract_rows"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Teacher Failure Localization Contract",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        "",
        "## Required Arms",
        "",
    ]
    lines.extend(f"- `{arm}`" for arm in summary["required_arms"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary["rationale"]),
            "",
            "## Next Step",
            "",
            str(summary["selected_next_step"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout-dir", type=Path, default=DEFAULT_CLOSEOUT_DIR)
    parser.add_argument("--distillation-dir", type=Path, default=DEFAULT_DISTILLATION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_teacher_failure_localization_contract(
        closeout_dir=args.closeout_dir,
        distillation_dir=args.distillation_dir,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
