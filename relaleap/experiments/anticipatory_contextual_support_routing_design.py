"""Design the anticipatory contextual support routing pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_anticipatory_contextual_support_routing_design"
)
DEFAULT_DESIGN_DOC = Path("docs/anticipatory_contextual_support_routing.md")
DEFAULT_PILOT_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_CAUSAL_CONFIG = Path(
    "configs/token_larger_support_wide_causal_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_RETENTION_DECISION = Path(
    "results/reports/token_larger_retention_churn_microtest_decision/decision_report.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")


DESIGN_RECORDED = "anticipatory_contextual_support_routing_design_recorded"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_NEXT_ACTION = "implement_local_acsr_smoke_pilot"


def run_anticipatory_contextual_support_routing_design(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    design_doc_path: Path = DEFAULT_DESIGN_DOC,
    pilot_config_path: Path = DEFAULT_PILOT_CONFIG,
    causal_config_path: Path = DEFAULT_CAUSAL_CONFIG,
    retention_decision_path: Path = DEFAULT_RETENTION_DECISION,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Write a fail-closed design artifact for the first local ACSR pilot."""

    start = time.time()
    design_doc = _read_text(design_doc_path)
    pilot_config = _read_yaml_object(pilot_config_path)
    causal_config = _read_yaml_object(causal_config_path)
    retention_decision = _read_json_object(retention_decision_path)
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = [
        _source_row(
            "design_doc",
            design_doc_path,
            design_doc_path.is_file(),
            "present" if design_doc else None,
            "anticipatory_contextual_support_routing_instruction",
            "branch_instruction",
        ),
        _source_row(
            "pilot_config",
            pilot_config_path,
            pilot_config_path.is_file(),
            _config_status(pilot_config),
            _config_experiment_id(pilot_config),
            "full_context_oracle_local_pilot_scale",
        ),
        _source_row(
            "causal_config",
            causal_config_path,
            causal_config_path.is_file(),
            _config_status(causal_config),
            _config_experiment_id(causal_config),
            "causal_feature_safe_control_scale",
        ),
        _source_row(
            "retention_churn_decision",
            retention_decision_path,
            retention_decision_path.is_file(),
            retention_decision.get("status"),
            retention_decision.get("decision"),
            "non_ce_guardrail_source",
        ),
        _source_row(
            "strategy_review",
            strategy_review_path,
            strategy_review["present"],
            "present" if strategy_review["present"] else "missing_optional",
            strategy_review["recommended_next_action"],
            (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        ),
    ]

    design = _design(pilot_config)
    criteria_rows = _criteria_rows()
    implementation_rows = _implementation_rows()
    failures = _failures(
        design_doc=design_doc,
        pilot_config=pilot_config,
        causal_config=causal_config,
        retention_decision=retention_decision,
        source_rows=source_rows,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_action = None
        next_command = None
        next_step = "repair missing or inconsistent ACSR design sources"
        rationale = (
            "The ACSR design cannot be treated as ready because required "
            "command-generated source artifacts, configs, or branch instructions "
            "are missing or inconsistent."
        )
    else:
        status = "pass"
        decision = DESIGN_RECORDED
        selected_next_action = SELECTED_NEXT_ACTION
        next_command = (
            "./.venv-conda/bin/python -m "
            "relaleap.experiments.anticipatory_contextual_support_routing "
            "--config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml "
            "--out results/audits/token_larger_anticipatory_contextual_support_routing"
        )
        next_step = (
            "implement the smallest local CPU ACSR smoke pilot with MLP/GRU "
            "causal future-feature predictors and fail-closed artifact checks"
        )
        rationale = (
            "The design follows Ben's anticipatory contextual support routing "
            "direction while preserving the non-CE guardrails from the completed "
            "retention/churn gate. It targets the full-context router's future "
            "feature chunks as train-only reconstruction targets, routes from "
            "predicted future chunks at evaluation, and requires shuffled, "
            "token/position-only, same-student, future-perturbation, and "
            "retention/churn controls before any promotion."
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "next_command": next_command,
        "next_step": next_step,
        "claim_statuses": {
            "full_context_contextual_router": "nondeployable_oracle_teacher_only",
            "anticipatory_contextual_support_router": "designed_not_yet_executed",
            "causal_contextual_router": "control_ce_baseline_not_promoted",
            "topk2_causal_cooperation": "not_supported_by_current_evidence",
            "hub_pair_mitigation": "deferred_rejected_diffuse_localization",
            "order_averaging": "diagnostic_only_not_promoted",
            "distillation": "deferred_until_acsr_non_ce_gate",
        },
        "feature_targets": design["feature_targets"],
        "causal_inputs": design["causal_inputs"],
        "pilot": design["pilot"],
        "artifact_schema": design["artifact_schema"],
        "control_rows": design["control_rows"],
        "criteria_rows": criteria_rows,
        "implementation_rows": implementation_rows,
        "source_rows": source_rows,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "control_rows_csv": str(out_dir / "control_rows.csv"),
            "criteria_rows_csv": str(out_dir / "criteria_rows.csv"),
            "implementation_rows_csv": str(out_dir / "implementation_rows.csv"),
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
        out_dir / "control_rows.csv",
        [
            "control",
            "router",
            "top_k",
            "purpose",
            "required_metric",
            "promotion_blocker_if",
        ],
        design["control_rows"],
    )
    _write_csv(
        out_dir / "criteria_rows.csv",
        ["gate", "metric", "pass_condition", "failure_interpretation"],
        criteria_rows,
    )
    _write_csv(
        out_dir / "implementation_rows.csv",
        ["step", "module_or_artifact", "requirement", "test_hook"],
        implementation_rows,
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _design(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    base_cfg = model_cfg.get("base", {}) if isinstance(model_cfg.get("base"), dict) else {}
    column_cfg = (
        model_cfg.get("columns", {}) if isinstance(model_cfg.get("columns"), dict) else {}
    )
    hidden_dim = int(base_cfg.get("hidden_dim", 96))
    seq_len = int(data_cfg.get("seq_len", 64))
    return {
        "feature_targets": [
            {
                "name": "future_hidden",
                "source": "ResidualColumns._contextual_features chunk 3",
                "shape": f"[batch, {seq_len}, {hidden_dim}]",
                "teacher": "next_hidden from full-context contextual_mlp",
                "evaluation_rule": "predicted by causal model; true tensor unavailable to router",
            },
            {
                "name": "future_delta",
                "source": "ResidualColumns._contextual_features chunk 5",
                "shape": f"[batch, {seq_len}, {hidden_dim}]",
                "teacher": "next_hidden - current_hidden from full-context contextual_mlp",
                "evaluation_rule": "predicted by causal model; true tensor unavailable to router",
            },
        ],
        "causal_inputs": [
            "current_hidden",
            "previous_hidden",
            "current_minus_previous_hidden",
            "normalized_position",
            "sin_position",
            "cos_position",
            "optional_past_support_summary",
            "optional_past_residual_norm_entropy_margin",
        ],
        "pilot": {
            "config_path": str(DEFAULT_PILOT_CONFIG),
            "out_dir": "results/audits/token_larger_anticipatory_contextual_support_routing",
            "dataset": data_cfg.get("dataset"),
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_columns": column_cfg.get("num_columns"),
            "atoms_per_column": column_cfg.get("atoms_per_column"),
            "top_k": column_cfg.get("top_k"),
            "support_router_teacher": column_cfg.get("support_router"),
            "causal_control_router": "contextual_mlp_causal",
            "predictor_families": ["small_mlp", "small_gru"],
            "defer_until_local_negative_or_ambiguous": ["causal_transformer", "RunPod"],
            "max_steps": run_cfg.get("max_steps"),
        },
        "artifact_schema": {
            "summary_json": "status, decision, gates, variants, best_predictor, failures",
            "predictor_metrics_csv": "predictor reconstruction loss and heldout feature R2/cosine",
            "router_metrics_csv": "CE, oracle regret, support use, future-invariance by variant",
            "same_student_metrics_csv": "support-forcing deltas through identical learned values",
            "retention_churn_metrics_csv": "A-to-B anchor drift, functional churn, support churn",
            "feature_perturbation_csv": "future-position perturbation deltas for router scores/supports",
            "notes_md": "plain-language interpretation and next-step gate",
        },
        "control_rows": _control_rows(),
    }


def _control_rows() -> list[dict[str, Any]]:
    return [
        {
            "control": "full_context_contextual_topk2_teacher",
            "router": "contextual_mlp",
            "top_k": 2,
            "purpose": "nondeployable upper-bound teacher and feature target source",
            "required_metric": "oracle_gap_to_acsr and CE gap",
            "promotion_blocker_if": "treated as deployable default",
        },
        {
            "control": "causal_feature_safe_contextual_topk2",
            "router": "contextual_mlp_causal",
            "top_k": 2,
            "purpose": "primary deployable baseline",
            "required_metric": "CE, oracle-support regret, functional churn",
            "promotion_blocker_if": "ACSR worsens regret or churn without compensating gain",
        },
        {
            "control": "linear_topk2",
            "router": "linear",
            "top_k": 2,
            "purpose": "support-routing floor at same active width",
            "required_metric": "CE and support utilization",
            "promotion_blocker_if": "ACSR cannot beat linear on CE guardrail",
        },
        {
            "control": "rank_matched_contextual_topk1",
            "router": "contextual_mlp",
            "top_k": 1,
            "purpose": "retention/churn guardrail",
            "required_metric": "anchor drift, support churn, functional churn",
            "promotion_blocker_if": "ACSR underperforms without clear CE/regret compensation",
        },
        {
            "control": "shuffled_predicted_features",
            "router": "ACSR with batch/position-shuffled predictions",
            "top_k": 2,
            "purpose": "reject cheap scale or smoothing effects",
            "required_metric": "ACSR minus shuffled CE/regret/churn deltas",
            "promotion_blocker_if": "shuffled control matches ACSR",
        },
        {
            "control": "token_position_only_predictor",
            "router": "ACSR with no hidden causal summary",
            "top_k": 2,
            "purpose": "reject token/position shortcut",
            "required_metric": "ACSR minus token-position-null deltas",
            "promotion_blocker_if": "token/position-only matches ACSR",
        },
        {
            "control": "random_fixed_topk2",
            "router": "fixed support",
            "top_k": 2,
            "purpose": "support identity null",
            "required_metric": "same-student CE/support-forcing delta",
            "promotion_blocker_if": "random support matches ACSR support",
        },
        {
            "control": "norm_matched_dense_active_rank",
            "router": "dense matched active-rank control",
            "top_k": "",
            "purpose": "active-rank/value capacity deconfounder",
            "required_metric": "CE drift and functional churn",
            "promotion_blocker_if": "dense control dominates all non-CE gates",
        },
    ]


def _criteria_rows() -> list[dict[str, str]]:
    return [
        {
            "gate": "non_cheating",
            "metric": "future_perturbation_score_delta and support_delta",
            "pass_condition": "future-position perturbations leave ACSR scores/supports unchanged except through causal history",
            "failure_interpretation": "ACSR is reading future information and is invalid as deployable evidence",
        },
        {
            "gate": "ce_guardrail",
            "metric": "alpha0_ce_loss",
            "pass_condition": "ACSR improves or matches causal_feature_safe_contextual_topk2",
            "failure_interpretation": "predictor does not recover useful deployable routing signal",
        },
        {
            "gate": "oracle_gap",
            "metric": "router_loss_minus_oracle_loss",
            "pass_condition": "ACSR closes part of the full-context teacher gap",
            "failure_interpretation": "CE changes are not support-selection quality evidence",
        },
        {
            "gate": "support_quality",
            "metric": "oracle-support regret",
            "pass_condition": "ACSR does not worsen regret versus causal_feature_safe_contextual_topk2",
            "failure_interpretation": "record as operational CE router only",
        },
        {
            "gate": "retention_churn",
            "metric": "anchor CE drift, logit KL/MSE, support churn, functional churn",
            "pass_condition": "ACSR lowers churn versus full-context top-k2 or compensates with CE/regret gains",
            "failure_interpretation": "not causal reusable column-selection evidence",
        },
        {
            "gate": "null_controls",
            "metric": "ACSR deltas versus shuffled and token/position-only controls",
            "pass_condition": "ACSR beats both nulls on CE/regret/churn composite",
            "failure_interpretation": "effect is likely shortcut, scale, or smoothing",
        },
    ]


def _implementation_rows() -> list[dict[str, str]]:
    return [
        {
            "step": "feature_extraction",
            "module_or_artifact": "relaleap.smoke.ResidualColumns._contextual_features",
            "requirement": "factor contextual feature construction into named chunks",
            "test_hook": "unit test chunk ordering and dimensions",
        },
        {
            "step": "predictor",
            "module_or_artifact": "relaleap.experiments.anticipatory_contextual_support_routing",
            "requirement": "train small MLP first, then GRU if MLP is non-discriminative",
            "test_hook": "smoke config writes predictor_metrics.csv",
        },
        {
            "step": "router_eval",
            "module_or_artifact": "ACSR support-score path",
            "requirement": "compose true causal chunks with predicted future_hidden/future_delta",
            "test_hook": "future perturbation leaves predictions/support fixed",
        },
        {
            "step": "same_student_controls",
            "module_or_artifact": "same learned residual values with alternate supports",
            "requirement": "compare predicted-feature support to token/position-null support",
            "test_hook": "same_student_metrics.csv has all required controls",
        },
        {
            "step": "retention_churn_gate",
            "module_or_artifact": "retention/churn microtest extension or shared helper",
            "requirement": "A-to-B adaptation metrics for ACSR and controls",
            "test_hook": "decision fails if retention_churn_metrics.csv is missing",
        },
    ]


def _failures(
    *,
    design_doc: str,
    pilot_config: dict[str, Any],
    causal_config: dict[str, Any],
    retention_decision: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:-1]:
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
    required_doc_terms = [
        "anticipatory contextual support routing",
        "shuffled predicted-feature control",
        "retention/churn",
    ]
    lower_doc = design_doc.lower()
    for term in required_doc_terms:
        if term not in lower_doc:
            failures.append(
                {
                    "source": "design_doc",
                    "field": "required_term",
                    "expected": term,
                    "actual": "missing",
                }
            )
    failures.extend(
        _config_failures(
            "pilot_config",
            pilot_config,
            expected_router="contextual_mlp",
            expected_top_k=2,
        )
    )
    failures.extend(
        _config_failures(
            "causal_config",
            causal_config,
            expected_router="contextual_mlp_causal",
            expected_top_k=2,
        )
    )
    if retention_decision.get("status") != "pass":
        failures.append(
            {
                "source": "retention_churn_decision",
                "field": "status",
                "expected": "pass",
                "actual": retention_decision.get("status"),
            }
        )
    if retention_decision.get("colab_replication_warranted") is not False:
        failures.append(
            {
                "source": "retention_churn_decision",
                "field": "colab_replication_warranted",
                "expected": False,
                "actual": retention_decision.get("colab_replication_warranted"),
            }
        )
    return failures


def _config_failures(
    source: str,
    config: dict[str, Any],
    *,
    expected_router: str,
    expected_top_k: int,
) -> list[dict[str, Any]]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    columns = (
        model_cfg.get("columns", {}) if isinstance(model_cfg.get("columns"), dict) else {}
    )
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    failures: list[dict[str, Any]] = []
    expected = {
        "data.dataset": "tiny_shakespeare_word",
        "model.columns.num_columns": 24,
        "model.columns.atoms_per_column": 4,
        "model.columns.top_k": expected_top_k,
        "model.columns.support_router": expected_router,
        "model.columns.support_stress_preset": False,
    }
    actual = {
        "data.dataset": data.get("dataset"),
        "model.columns.num_columns": columns.get("num_columns"),
        "model.columns.atoms_per_column": columns.get("atoms_per_column"),
        "model.columns.top_k": columns.get("top_k"),
        "model.columns.support_router": columns.get("support_router"),
        "model.columns.support_stress_preset": columns.get("support_stress_preset"),
    }
    for field, expected_value in expected.items():
        if actual.get(field) != expected_value:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected_value,
                    "actual": actual.get(field),
                }
            )
    return failures


def _source_row(
    source: str,
    path: Path,
    present: bool,
    status: Any,
    decision: Any,
    claim_status: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": present,
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
    }


def _config_status(config: dict[str, Any]) -> str | None:
    return "present" if config else None


def _config_experiment_id(config: dict[str, Any]) -> str | None:
    run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    value = run.get("experiment_id")
    return str(value) if value is not None else None


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_yaml_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "path": str(path),
            "strategic_change_level": None,
            "notify_ben": None,
            "ben_notification_required": False,
            "recommended_next_action": None,
            "incorporation": "missing optional review; proceeded from command artifacts",
        }
    headers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    notify_ben = headers.get("notify_ben", "").lower() == "true"
    strategic_change_level = headers.get("strategic_change_level")
    return {
        "present": True,
        "path": str(path),
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": headers.get("recommended_next_action"),
        "incorporation": (
            "accepted: no hub-pair/order-averaging mitigation, no distillation "
            "promotion, and no GPU repeat before a local ACSR design/pilot with "
            "retention/churn and null-control gates"
        ),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Anticipatory Contextual Support Routing Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next action: `{summary['selected_next_action']}`",
        "",
        summary["rationale"],
        "",
        "Feature targets:",
    ]
    for target in summary["feature_targets"]:
        lines.append(f"- `{target['name']}`: {target['teacher']}")
    lines.extend(["", f"Next step: {summary['next_step']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_anticipatory_contextual_support_routing_design(out_dir=args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
