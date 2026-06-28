"""Design the non-synthetic core/periphery PC-column pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYNTHESIS = Path("results/reports/core_periphery_pc_column_synthesis/summary.json")
DEFAULT_DESIGN_CONTRACT = Path("results/reports/core_periphery_pc_column_design/summary.json")
DEFAULT_CHAR_CONFIG = Path(
    "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_LARGER_CONFIG = Path(
    "configs/char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_TOKEN_CONFIG = Path(
    "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/core_periphery_pc_column_nonsynthetic_pilot_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_evidence.csv",
    "pilot_arms.csv",
    "hidden_state_contract.csv",
    "gate_criteria.csv",
    "notes.md",
)

REQUIRED_PILOT_ARMS = (
    "core_periphery_pc_contextual_router",
    "current_sparse_acsr_contextual_router",
    "dense_rank_norm_residual",
    "parameter_matched_causal_mlp",
    "random_support_router",
    "frequency_support_router",
    "token_position_only_router",
    "lambda_zero_residual",
    "no_core_ablation",
    "no_periphery_ablation",
    "equal_plasticity_core_periphery",
    "shuffled_core_periphery_assignment",
)

REQUIRED_HIDDEN_STATE_FIELDS = (
    "frozen_base_hidden_train",
    "frozen_base_hidden_anchor",
    "frozen_base_hidden_heldout",
    "frozen_base_logits_anchor",
    "token_ids",
    "positions",
    "support_router_inputs",
    "teacher_hidden_delta_training_only",
)


def run_core_periphery_pc_column_nonsynthetic_pilot_design(
    *,
    synthesis_path: Path = DEFAULT_SYNTHESIS,
    design_contract_path: Path = DEFAULT_DESIGN_CONTRACT,
    char_config_path: Path = DEFAULT_CHAR_CONFIG,
    larger_config_path: Path = DEFAULT_LARGER_CONFIG,
    token_config_path: Path = DEFAULT_TOKEN_CONFIG,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed design artifact for the frozen-hidden-state pilot."""

    start = time.time()
    synthesis = _read_json(synthesis_path)
    design_contract = _read_json(design_contract_path)
    configs = {
        "char_validation": _read_yaml(char_config_path),
        "char_larger": _read_yaml(larger_config_path),
        "token_larger": _read_yaml(token_config_path),
    }
    strategy = _strategy_review(strategy_review_path)
    source_rows = _source_rows(
        synthesis_path=synthesis_path,
        synthesis=synthesis,
        design_contract_path=design_contract_path,
        design_contract=design_contract,
        config_paths={
            "char_validation": char_config_path,
            "char_larger": larger_config_path,
            "token_larger": token_config_path,
        },
        configs=configs,
        strategy_review_path=strategy_review_path,
        strategy=strategy,
    )
    arm_rows = _pilot_arms(configs)
    hidden_rows = _hidden_state_contract(configs)
    gate_rows = _gate_rows(synthesis, design_contract, configs, arm_rows, hidden_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "core_periphery_pc_column_nonsynthetic_pilot_design_recorded"
            if status == "pass"
            else "core_periphery_pc_column_nonsynthetic_pilot_design_failed_closed"
        ),
        "scientific_gate": "ready_for_local_nonsynthetic_pilot_implementation"
        if status == "pass"
        else "blocked",
        "claim_status": "design_only_not_training_gpu_or_promotion_evidence",
        "requires_gpu_now": False,
        "backend_policy": (
            "local CPU implementation only; RunPod/Colab validation remains blocked "
            "until the non-synthetic pilot exists and passes its local artifact gates"
        ),
        "selected_next_step": (
            "implement the command-driven local frozen-hidden-state pilot from this design"
            if status == "pass"
            else "repair missing synthesis, contract, config, arm, or hidden-state design fields"
        ),
        "source_evidence": source_rows,
        "pilot_arms": arm_rows,
        "hidden_state_contract": hidden_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "interpretation": _interpretation(status),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "generated_from_head": _git_commit(),
        "dirty_diff_hash": _dirty_diff_hash(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    *,
    synthesis_path: Path,
    synthesis: dict[str, Any],
    design_contract_path: Path,
    design_contract: dict[str, Any],
    config_paths: dict[str, Path],
    configs: dict[str, dict[str, Any]],
    strategy_review_path: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "source": "core_periphery_synthesis",
            "path": str(synthesis_path),
            "present": bool(synthesis),
            "status": synthesis.get("status", "missing"),
            "decision": synthesis.get("decision", ""),
            "claim_status": synthesis.get("claim_status", ""),
            "scientific_gate": synthesis.get("scientific_gate", ""),
        },
        {
            "source": "core_periphery_design_contract",
            "path": str(design_contract_path),
            "present": bool(design_contract),
            "status": design_contract.get("status", "missing"),
            "decision": design_contract.get("decision", ""),
            "claim_status": design_contract.get("claim_status", ""),
            "scientific_gate": design_contract.get("scientific_gate", ""),
        },
    ]
    for name, path in config_paths.items():
        cfg = configs[name]
        rows.append(
            {
                "source": f"{name}_config",
                "path": str(path),
                "present": bool(cfg),
                "status": "present" if cfg else "missing",
                "decision": _experiment_id(cfg),
                "claim_status": _config_summary(cfg),
                "scientific_gate": "pilot_scale_source",
            }
        )
    rows.append(
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
            "scientific_gate": "external_review_context",
        }
    )
    return rows


def _pilot_arms(configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    char_cfg = configs.get("char_validation", {})
    return [
        _arm("core_periphery_pc_contextual_router", "candidate", char_cfg, "split core/periphery PC residual columns"),
        _arm("current_sparse_acsr_contextual_router", "sparse_control", char_cfg, "current promoted contextual support router without split core/periphery units"),
        _arm("dense_rank_norm_residual", "dense_control", char_cfg, "matched active rank and residual-norm comparator"),
        _arm("parameter_matched_causal_mlp", "mlp_control", char_cfg, "causal-input MLP with matched parameter and compute budget"),
        _arm("random_support_router", "support_null", char_cfg, "uniform random support under the same top-k"),
        _arm("frequency_support_router", "support_null", char_cfg, "train-frequency support null with no heldout labels"),
        _arm("token_position_only_router", "router_null", char_cfg, "causal token/position-only router"),
        _arm("lambda_zero_residual", "residual_null", char_cfg, "frozen base and artifact-contract baseline"),
        _arm("no_core_ablation", "mechanism_ablation", char_cfg, "periphery-only split with matched residual budget"),
        _arm("no_periphery_ablation", "mechanism_ablation", char_cfg, "protected-core-only split with matched residual budget"),
        _arm("equal_plasticity_core_periphery", "mechanism_ablation", char_cfg, "same learning rate and consolidation for core and periphery"),
        _arm("shuffled_core_periphery_assignment", "mechanism_null", char_cfg, "post-training core/periphery assignment shuffle"),
    ]


def _arm(name: str, family: str, config: dict[str, Any], role: str) -> dict[str, Any]:
    columns = ((config.get("model") or {}).get("columns") or {}) if isinstance(config, dict) else {}
    return {
        "arm": name,
        "family": family,
        "top_k": columns.get("top_k", ""),
        "num_columns": columns.get("num_columns", ""),
        "support_router": columns.get("support_router", ""),
        "matched_budget": "params, active_compute, residual_l2, train_tokens, seed, frozen_base_checkpoint",
        "role": role,
        "required_outputs": (
            "heldout_ce; anchor_kl; flip_churn; functional_churn; residual_stream_churn; "
            "finite_update_commutator; intervention_fingerprints; pruning_metrics"
        ),
    }


def _hidden_state_contract(configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    char_cfg = configs.get("char_validation", {})
    data = char_cfg.get("data", {}) if isinstance(char_cfg.get("data"), dict) else {}
    model = char_cfg.get("model", {}) if isinstance(char_cfg.get("model"), dict) else {}
    base = model.get("base", {}) if isinstance(model.get("base"), dict) else {}
    seq_len = data.get("seq_len", 64)
    hidden_dim = base.get("hidden_dim", 64)
    return [
        _hidden("frozen_base_hidden_train", f"[train_batches, {seq_len}, {hidden_dim}]", "train-only local PC updates", "base model frozen; no labels needed"),
        _hidden("frozen_base_hidden_anchor", f"[anchor_batches, {seq_len}, {hidden_dim}]", "retention/churn baseline", "must be captured before candidate updates"),
        _hidden("frozen_base_hidden_heldout", f"[heldout_batches, {seq_len}, {hidden_dim}]", "heldout CE and intervention evaluation", "not used for training updates"),
        _hidden("frozen_base_logits_anchor", "[anchor_batches, seq, vocab]", "anchor KL, flip churn, CE guardrail", "base logits frozen for drift comparison"),
        _hidden("token_ids", "[split_batches, seq]", "token controls and CE labels where allowed", "labels only enter supervised guardrail/objective terms, not routing"),
        _hidden("positions", "[split_batches, seq]", "position features and token/position-only null", "causal feature only"),
        _hidden("support_router_inputs", "[split_batches, seq, causal_features]", "contextual/ACSR router inputs", "no future hidden states or heldout labels"),
        _hidden("teacher_hidden_delta_training_only", f"[train_batches, {seq_len}, {hidden_dim}]", "optional dense-teacher/local hidden-delta PC target", "training-only; forbidden at evaluation"),
    ]


def _hidden(field: str, shape: str, purpose: str, leakage_rule: str) -> dict[str, str]:
    return {
        "field": field,
        "shape": shape,
        "purpose": purpose,
        "leakage_rule": leakage_rule,
        "required": "true",
    }


def _gate_rows(
    synthesis: dict[str, Any],
    design_contract: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arms = {row["arm"] for row in arm_rows}
    hidden_fields = {row["field"] for row in hidden_rows if row.get("required") == "true"}
    return [
        _criterion(
            "synthesis_ready_for_nonsynthetic_design",
            synthesis.get("status") == "pass"
            and synthesis.get("scientific_gate") == "ready_for_non_synthetic_pilot_design",
            "hard",
            "completed local repeat synthesis allows only non-synthetic pilot design",
            synthesis.get("scientific_gate", "missing"),
            "rerun or repair core_periphery_pc_column_synthesis",
        ),
        _criterion(
            "design_contract_ready",
            design_contract.get("status") == "pass"
            and design_contract.get("scientific_gate") == "ready_for_tiny_pilot",
            "hard",
            "base design contract is present and ready for tiny/local pilots",
            design_contract.get("scientific_gate", "missing"),
            "rerun or repair core_periphery_pc_column_design",
        ),
        _criterion(
            "all_reference_configs_present",
            all(bool(cfg) for cfg in configs.values()),
            "hard",
            "char validation, larger char, and token larger support-wide configs are readable",
            {name: bool(cfg) for name, cfg in configs.items()},
            "restore missing config sources before designing the pilot",
        ),
        _criterion(
            "pilot_arms_complete",
            set(REQUIRED_PILOT_ARMS).issubset(arms),
            "hard",
            "candidate plus dense/MLP/null/ablation controls are preregistered",
            sorted(arms),
            "add missing mandatory pilot arms",
        ),
        _criterion(
            "hidden_state_contract_complete",
            set(REQUIRED_HIDDEN_STATE_FIELDS).issubset(hidden_fields),
            "hard",
            "frozen hidden/logit/input tensors and leakage rules are preregistered",
            sorted(hidden_fields),
            "add missing frozen-hidden-state contract fields",
        ),
        _criterion(
            "local_only_before_gpu",
            True,
            "hard",
            "RunPod/Colab validation blocked until local non-synthetic pilot artifacts exist",
            "requires_gpu_now=false",
            "do not run GPU validation from this design step",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    expected: Any,
    actual: Any,
    failure_action: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "failure_action": failure_action,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _experiment_id(config: dict[str, Any]) -> str:
    run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    return str(run.get("experiment_id", ""))


def _config_summary(config: dict[str, Any]) -> str:
    if not config:
        return ""
    run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    columns = model.get("columns", {}) if isinstance(model.get("columns"), dict) else {}
    return (
        f"seed={run.get('seed')}; steps={run.get('max_steps')}; "
        f"seq_len={data.get('seq_len')}; columns={columns.get('num_columns')}; "
        f"top_k={columns.get('top_k')}; router={columns.get('support_router')}"
    )


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": None,
        "notify_ben": None,
        "recommended_next_action": None,
        "verdict": None,
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in fields:
                fields[key] = value.strip()
    fields["ben_notification_required"] = (
        str(fields.get("notify_ben")).lower() == "true"
        or fields.get("strategic_change_level") == "major"
    )
    return fields


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if strategy.get("ben_notification_required"):
        return "accepted direction but recorded Ben notification requirement from review header"
    return (
        "accepted: continues the local fail-closed core/periphery path; no recommendation "
        "was deferred or rejected in this bounded design step"
    )


def _interpretation(status: str) -> str:
    if status == "pass":
        return (
            "The synthetic repeat evidence is sufficient only to design a local "
            "non-synthetic pilot. The next implementation must consume frozen base "
            "hidden states from the command-driven harness and preserve the dense, "
            "MLP, null, ablation, retention/churn, commutator, pruning, intervention, "
            "and CE guardrails before any GPU validation."
        )
    return "The non-synthetic pilot design is incomplete; implementation and GPU validation remain blocked."


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_evidence.csv", summary["source_evidence"])
    _write_csv(out_dir / "pilot_arms.csv", summary["pilot_arms"])
    _write_csv(out_dir / "hidden_state_contract.csv", summary["hidden_state_contract"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Core/Periphery PC Column Non-Synthetic Pilot Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        summary["interpretation"],
        "",
        f"Next step: {summary['selected_next_step']}",
    ]
    if summary["strategy_review"].get("ben_notification_required"):
        lines.extend(["", "Ben notification is required by the strategy review header."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff"], text=True)
    except Exception:
        return "unknown"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--design-contract", type=Path, default=DEFAULT_DESIGN_CONTRACT)
    parser.add_argument("--char-config", type=Path, default=DEFAULT_CHAR_CONFIG)
    parser.add_argument("--larger-config", type=Path, default=DEFAULT_LARGER_CONFIG)
    parser.add_argument("--token-config", type=Path, default=DEFAULT_TOKEN_CONFIG)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_core_periphery_pc_column_nonsynthetic_pilot_design(
        synthesis_path=args.synthesis,
        design_contract_path=args.design_contract,
        char_config_path=args.char_config,
        larger_config_path=args.larger_config,
        token_config_path=args.token_config,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scientific_gate": summary["scientific_gate"],
                "selected_next_step": summary["selected_next_step"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
