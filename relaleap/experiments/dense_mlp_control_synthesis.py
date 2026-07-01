"""Synthesize dense/MLP dominance and select the next local sparse-interference pregate."""

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


DEFAULT_PAIR_CLOSEOUT = Path("results/reports/dense_teacher_pair_composer_closeout/summary.json")
DEFAULT_CONTEXT_CORE_CLOSEOUT = Path("results/reports/context_contrastive_core_periphery_closeout/summary.json")
DEFAULT_REGRET_CLOSEOUT = Path("results/reports/regret_soft_utility_head_closeout/summary.json")
DEFAULT_VALUE_DICTIONARY_CLOSEOUT = Path(
    "results/reports/low_churn_mlp_value_dictionary_capacity_rescue_closeout/summary.json"
)
DEFAULT_NORM_CHURN_PILOT = Path("results/reports/norm_budgeted_churn_regularized_residual_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/dense_mlp_control_synthesis")

ORTHOGONALIZED_SPARSE_PREGATE_ACTION = "design_orthogonalized_sparse_additive_core_periphery_interference_pregate"
REPAIR_ACTION = "repair_dense_mlp_control_synthesis_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_matrix.csv",
    "pregate_spec.csv",
    "candidate_actions.csv",
    "notes.md",
)


def run_dense_mlp_control_synthesis(
    *,
    pair_closeout_path: Path = DEFAULT_PAIR_CLOSEOUT,
    context_core_closeout_path: Path = DEFAULT_CONTEXT_CORE_CLOSEOUT,
    regret_closeout_path: Path = DEFAULT_REGRET_CLOSEOUT,
    value_dictionary_closeout_path: Path = DEFAULT_VALUE_DICTIONARY_CLOSEOUT,
    norm_churn_pilot_path: Path = DEFAULT_NORM_CHURN_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local branch synthesis from command-generated source artifacts."""

    start = time.time()
    pair = _read_json(pair_closeout_path)
    context_core = _read_json(context_core_closeout_path)
    regret = _read_json(regret_closeout_path)
    value_dictionary = _read_json(value_dictionary_closeout_path)
    norm_churn = _read_json(norm_churn_pilot_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("dense_teacher_pair_composer_closeout", pair_closeout_path, pair),
        _source_row("context_contrastive_core_periphery_closeout", context_core_closeout_path, context_core),
        _source_row("regret_soft_utility_head_closeout", regret_closeout_path, regret),
        _source_row("low_churn_mlp_value_dictionary_capacity_rescue_closeout", value_dictionary_closeout_path, value_dictionary),
        _source_row("norm_budgeted_churn_regularized_residual_pilot", norm_churn_pilot_path, norm_churn),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "sha256": _file_sha256(strategy_review_path),
            "mtime": _mtime(strategy_review_path),
            "status": "read" if strategy["present"] else "missing_required",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
        },
    ]
    decision_matrix = _decision_matrix(pair, context_core, regret, value_dictionary, norm_churn, strategy)
    pregate_spec = _pregate_spec()
    source_failures = [
        {"source": row["source"], "reason": f"{row['path']} is missing or unreadable"}
        for row in source_rows
        if not row["present"]
    ]
    gate_failures = [row for row in decision_matrix if row["required"] and not row["passed"]]
    selected = not source_failures and not gate_failures
    selected_next_action = ORTHOGONALIZED_SPARSE_PREGATE_ACTION if selected else REPAIR_ACTION
    status = "pass" if selected else "fail"
    summary = {
        "status": status,
        "decision": (
            "dense_mlp_dominance_synthesized_sparse_interference_pregate_selected"
            if selected
            else "dense_mlp_control_synthesis_failed_closed"
        ),
        "claim_status": (
            "pair_composer_closed_dense_mlp_dominance_sparse_interference_pregate_selected_no_gpu"
            if selected
            else "dense_mlp_control_synthesis_sources_or_gates_incomplete"
        ),
        "selected_next_action": selected_next_action,
        "selected_next_step": (
            "implement a local pregate for orthogonalized additive sparse columns with norm controllers, update masks, and protected-core/plastic-periphery value modules"
            if selected
            else "repair or regenerate missing source artifacts before selecting a mechanism branch"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local synthesis and pregate design only; RunPod and Colab remain blocked",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "direction_shift_recorded": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "decision_matrix": decision_matrix,
        "pregate_spec": pregate_spec,
        "candidate_actions": _candidate_actions(selected_next_action),
        "failures": source_failures + gate_failures,
        "rationale": _rationale(pair, selected),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _decision_matrix(
    pair: dict[str, Any],
    context_core: dict[str, Any],
    regret: dict[str, Any],
    value_dictionary: dict[str, Any],
    norm_churn: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    dense_signal = _decision_signal(pair, "dense_mlp_controls_dominate_pair_composer")
    dense_actual = dense_signal.get("actual") if isinstance(dense_signal.get("actual"), dict) else {}
    oracle_ce = _float(dense_actual.get("oracle_pair_holdout_ce"))
    best_control_ce = _float(dense_actual.get("best_matched_control_ce"))
    return [
        {
            "signal": "pair_composer_branch_closed_by_dense_mlp_controls",
            "required": True,
            "passed": (
                pair.get("status") == "pass"
                and pair.get("selected_next_action") == "redirect_from_pair_composer_to_dense_mlp_control_synthesis"
                and dense_signal.get("passed") is True
                and oracle_ce is not None
                and best_control_ce is not None
                and best_control_ce < oracle_ce
            ),
            "actual": {
                "pair_decision": pair.get("decision"),
                "pair_claim_status": pair.get("claim_status"),
                "oracle_pair_holdout_ce": oracle_ce,
                "best_matched_control": dense_actual.get("best_matched_control"),
                "best_matched_control_ce": best_control_ce,
            },
            "expected": "pair composer is closed because matched dense/MLP controls beat even oracle pair composition",
        },
        {
            "signal": "prior_sparse_or_support_branches_are_not_gpu_candidates",
            "required": True,
            "passed": all(
                source.get("status") == "pass"
                and source.get("requires_gpu_now") is False
                and source.get("advance_to_gpu_validation", False) is False
                for source in (context_core, regret, value_dictionary)
            ),
            "actual": {
                "context_core": context_core.get("selected_next_action"),
                "regret_soft": regret.get("selected_next_action"),
                "value_dictionary": value_dictionary.get("selected_next_action"),
            },
            "expected": "recent local sparse/support branches must be closed, redirected, or fail-closed before a new branch is selected",
        },
        {
            "signal": "norm_churn_control_context_blocks_ce_only_promotion",
            "required": True,
            "passed": (
                norm_churn.get("status") == "pass"
                and norm_churn.get("requires_gpu_now") is False
                and norm_churn.get("promotion_allowed") is False
                and norm_churn.get("scientific_gate") == "blocked"
            ),
            "actual": {
                "decision": norm_churn.get("decision"),
                "claim_status": norm_churn.get("claim_status"),
                "scientific_gate": norm_churn.get("scientific_gate"),
                "selected_next_step": norm_churn.get("selected_next_step"),
            },
            "expected": "dense/MLP strength must redirect to interference/retention/commutator criteria, not CE-only promotion",
        },
        {
            "signal": "major_strategy_review_accepted_and_ben_notification_recorded",
            "required": True,
            "passed": (
                strategy["present"]
                and strategy["ben_notification_required"]
                and str(strategy["verdict"]).upper() == "PIVOT"
                and "dense/MLP" in strategy["recommended_next_action"]
            ),
            "actual": {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "verdict": strategy["verdict"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "expected": "latest GPT-5.5-Pro major pivot is accepted; Ben should be notified",
        },
        {
            "signal": "next_branch_is_local_mechanism_pregate_not_gpu",
            "required": True,
            "passed": True,
            "actual": {
                "selected_action": ORTHOGONALIZED_SPARSE_PREGATE_ACTION,
                "requires_gpu_now": False,
                "promotion_allowed": False,
            },
            "expected": "the synthesis selects one local sparse-interference pregate and keeps backend validation blocked",
        },
    ]


def _pregate_spec() -> list[dict[str, Any]]:
    return [
        {
            "component": "mechanism_arm",
            "requirement": "orthogonalized additive sparse columns with learned scalar norm controller",
            "promotion_gate": "match dense/MLP task quality band while improving retention, functional churn, and commutator metrics",
        },
        {
            "component": "within_column_structure",
            "requirement": "protected lower-plasticity core plus plastic/pruneable periphery and explicit per-column update masks",
            "promotion_gate": "periphery-first pruning preserves task-generic retention better than matched dense/MLP and flat-value controls",
        },
        {
            "component": "matched_controls",
            "requirement": "dense ridge, random-feature MLP, low-rank residual, flat-value same-router control, random sparse columns",
            "promotion_gate": "all controls matched on active params, stored params, residual norm, and CE band",
        },
        {
            "component": "nulls_and_leakage",
            "requirement": "token/position-only, shuffled/delayed teacher residuals, random supports, majority support, feature-schema hash",
            "promotion_gate": "deployable router inputs exclude future hidden/delta, teacher logits, oracle supports, and teacher residual labels",
        },
        {
            "component": "observables",
            "requirement": "CE guardrail, residual norm, functional churn, retention, finite-update commutator, intervention selectivity, reuse across contexts",
            "promotion_gate": "sparse mechanism must beat or match dense/MLP on interference at comparable quality before any GPU validation",
        },
    ]


def _candidate_actions(selected_next_action: str) -> list[dict[str, str]]:
    return [
        {
            "candidate_action": ORTHOGONALIZED_SPARSE_PREGATE_ACTION,
            "disposition": "selected" if selected_next_action == ORTHOGONALIZED_SPARSE_PREGATE_ACTION else "blocked",
            "claim_status": "mechanism_factorized_sparse_interference_pregate_selected_no_gpu",
            "reason": "Dense/MLP controls dominate pair-composer CE, so the next branch must be judged by interference and reuse, not another support-label tuning loop.",
            "next_step": "design the local pregate and fail-closed measurement table before training or GPU validation",
        },
        {
            "candidate_action": "reopen_pair_composer_tuning_or_runpod_validation",
            "disposition": "rejected",
            "claim_status": "pair_composer_gpu_blocked_by_dense_mlp_dominance",
            "reason": "Matched dense/MLP controls beat even the oracle pair composer on holdout CE.",
            "next_step": "only reopen with a new low-interference mechanism that beats matched controls on non-CE metrics",
        },
        {
            "candidate_action": "continue_ce_only_dense_teacher_distillation",
            "disposition": "rejected",
            "claim_status": "ce_only_distillation_not_aligned_with_current_research_question",
            "reason": "The project pivot treats CE as a guardrail; dense-control dominance should force interference/retention/commutator testing.",
            "next_step": "use dense/MLP rows as matched controls in the sparse-interference pregate",
        },
    ]


def _rationale(pair: dict[str, Any], selected: bool) -> str:
    if not selected:
        return "The synthesis could not establish all required local source and strategy gates, so it fails closed."
    dense_signal = _decision_signal(pair, "dense_mlp_controls_dominate_pair_composer")
    actual = dense_signal.get("actual") if isinstance(dense_signal.get("actual"), dict) else {}
    return (
        "The pair-composer signal remains real relative to independent and majority/null baselines, but it is closed "
        "as sparse-column evidence because matched dense/MLP residual controls dominate the oracle pair-composer CE "
        f"({actual.get('best_matched_control_ce')} versus {actual.get('oracle_pair_holdout_ce')}). The accepted major "
        "strategy review redirects the loop to a local mechanism-factorized sparse-interference pregate and records "
        "that Ben should be notified."
    )


def _decision_signal(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary.get("decision_matrix", []):
        if isinstance(row, dict) and row.get("signal") == name:
            return row
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    strategy = {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": _header_value(text, "notify_ben") or "unknown",
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }
    strategy["ben_notification_required"] = (
        str(strategy["notify_ben"]).lower() == "true"
        or str(strategy["strategic_change_level"]).lower() == "major"
    )
    return strategy


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "Strategy review was missing; synthesis fails closed until the latest review is available."
    if strategy["ben_notification_required"]:
        return "Accepted the GPT-5.5-Pro major pivot; Ben should be notified, and GPU remains blocked."
    return "Accepted the GPT-5.5-Pro local-control recommendation; GPU remains blocked."


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "sha256": _file_sha256(path),
        "mtime": _mtime(path),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(path.stat().st_mtime)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_matrix.csv", summary["decision_matrix"])
    _write_csv(out_dir / "pregate_spec.csv", summary["pregate_spec"])
    _write_csv(out_dir / "candidate_actions.csv", summary["candidate_actions"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], lineterminator="\n")
        writer.writeheader()
        for row in rows or [{"status": "missing"}]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames or []})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense/MLP Control Synthesis",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
            f"- Direction shift recorded: `{summary['direction_shift_recorded']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked. The selected branch is a local mechanism pregate with dense/MLP/null controls.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-closeout", type=Path, default=DEFAULT_PAIR_CLOSEOUT)
    parser.add_argument("--context-core-closeout", type=Path, default=DEFAULT_CONTEXT_CORE_CLOSEOUT)
    parser.add_argument("--regret-closeout", type=Path, default=DEFAULT_REGRET_CLOSEOUT)
    parser.add_argument("--value-dictionary-closeout", type=Path, default=DEFAULT_VALUE_DICTIONARY_CLOSEOUT)
    parser.add_argument("--norm-churn-pilot", type=Path, default=DEFAULT_NORM_CHURN_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_dense_mlp_control_synthesis(
        pair_closeout_path=args.pair_closeout,
        context_core_closeout_path=args.context_core_closeout,
        regret_closeout_path=args.regret_closeout,
        value_dictionary_closeout_path=args.value_dictionary_closeout,
        norm_churn_pilot_path=args.norm_churn_pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "ben_should_be_notified": summary["ben_should_be_notified"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
