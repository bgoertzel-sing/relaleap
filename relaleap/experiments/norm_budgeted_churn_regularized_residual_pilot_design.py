"""Design a local norm-budgeted churn-regularized residual pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_MATCHED_DECISION_DIR = Path("results/reports/sparse_dense_mlp_matched_intervention_decision")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/norm_budgeted_churn_regularized_residual_pilot_design")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_evidence.csv",
    "pilot_arms.csv",
    "objective_terms.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_norm_budgeted_churn_regularized_residual_pilot_design(
    *,
    matched_decision_dir: Path = DEFAULT_MATCHED_DECISION_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed design spec for the next local residual pilot."""

    start = time.time()
    matched_summary = _read_json(matched_decision_dir / "summary.json")
    available_rows = _read_csv(matched_decision_dir / "available_arms.csv")
    pareto_rows = _read_csv(matched_decision_dir / "pareto_frontier.csv")
    domination_rows = _read_csv(matched_decision_dir / "domination_cases.csv")
    strategy = _strategy_review(strategy_review_path)
    dense24 = _available_arm(available_rows, "dense_rank24_best_norm")
    residual_budget = _float(dense24.get("heldout_residual_update_l2"))
    if residual_budget is None:
        residual_budget = _best_dense_budget(pareto_rows)

    source_rows = _source_rows(
        matched_decision_dir=matched_decision_dir,
        matched_summary=matched_summary,
        available_rows=available_rows,
        pareto_rows=pareto_rows,
        domination_rows=domination_rows,
        strategy_review_path=strategy_review_path,
        strategy=strategy,
        residual_budget=residual_budget,
    )
    arm_rows = _pilot_arms(residual_budget)
    objective_rows = _objective_terms(residual_budget)
    gate_rows = _gate_rows(matched_summary, available_rows, domination_rows, strategy, residual_budget)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "norm_budgeted_churn_regularized_residual_pilot_design_recorded"
            if status == "pass"
            else "norm_budgeted_churn_regularized_residual_pilot_design_failed_closed"
        ),
        "claim_status": "design_only_no_candidate_promoted",
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "selected_next_step": (
            "implement the local low-step pilot from pilot_arms.csv and objective_terms.csv"
            if status == "pass"
            else "repair matched sparse/dense/MLP decision evidence before designing the pilot"
        ),
        "residual_l2_budget_source": "dense_rank24_best_norm_heldout_mean_or_dense_pareto_fallback",
        "residual_l2_budget": residual_budget if residual_budget is not None else "",
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_evidence": source_rows,
        "pilot_arms": arm_rows,
        "objective_terms": objective_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "claim_boundaries": {
            "supported": [
                "a local CPU pilot design can be specified from the matched sparse/dense/MLP artifacts",
                "dense rank24 is the hard comparator for local advancement",
                "MLP remains a high-capacity control until it wins under the same norm and churn budget",
            ],
            "not_supported": [
                "GPU validation before a local budgeted pilot exists",
                "promotion of sparse ACSR or MLP from post-hoc lambda scaling",
                "interpreting artifact pass as a scientific mechanism pass",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    *,
    matched_decision_dir: Path,
    matched_summary: dict[str, Any],
    available_rows: list[dict[str, str]],
    pareto_rows: list[dict[str, str]],
    domination_rows: list[dict[str, str]],
    strategy_review_path: Path,
    strategy: dict[str, Any],
    residual_budget: float | None,
) -> list[dict[str, Any]]:
    return [
        {
            "source": "matched_sparse_dense_mlp_summary",
            "path": str(matched_decision_dir / "summary.json"),
            "present": bool(matched_summary),
            "status": matched_summary.get("status", "missing"),
            "decision": matched_summary.get("decision", ""),
            "metric": "advancement_row_count",
            "value": matched_summary.get("advancement_row_count", ""),
        },
        {
            "source": "matched_available_arms",
            "path": str(matched_decision_dir / "available_arms.csv"),
            "present": bool(available_rows),
            "status": "present" if available_rows else "missing",
            "decision": f"rows={len(available_rows)}",
            "metric": "dense_rank24_residual_l2_budget",
            "value": residual_budget if residual_budget is not None else "",
        },
        {
            "source": "matched_dense_pareto_frontier",
            "path": str(matched_decision_dir / "pareto_frontier.csv"),
            "present": bool(pareto_rows),
            "status": "present" if pareto_rows else "missing",
            "decision": f"rows={len(pareto_rows)}",
            "metric": "row_count",
            "value": len(pareto_rows),
        },
        {
            "source": "matched_domination_cases",
            "path": str(matched_decision_dir / "domination_cases.csv"),
            "present": bool(domination_rows),
            "status": "present" if domination_rows else "missing",
            "decision": f"rows={len(domination_rows)}",
            "metric": "advancing_cases",
            "value": sum(_truthy(row.get("challenger_advances")) for row in domination_rows),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "metric": "review_header",
            "value": f"strategic_change_level={strategy['strategic_change_level']}; notify_ben={strategy['notify_ben']}",
        },
    ]


def _pilot_arms(residual_budget: float | None) -> list[dict[str, Any]]:
    budget = "" if residual_budget is None else residual_budget
    return [
        _arm("dense_rank24_norm_budgeted", "dense_control", budget, "hard comparator and Pareto anchor"),
        _arm("dense_rank16_norm_budgeted", "dense_control", budget, "lower-rank dense control"),
        _arm("sparse_contextual_topk2_norm_budgeted", "sparse_acsr", budget, "current sparse top-k2 mechanism candidate"),
        _arm("sparse_rank_matched_topk1_norm_budgeted", "sparse_acsr", budget, "rank-matched sparse support-width control"),
        _arm("sparse_frequency_matched_random_topk1_norm_budgeted", "sparse_null", budget, "frequency-preserving support null"),
        _arm("sparse_random_residual_same_l2_null", "residual_null", budget, "random residual vector scaled to the same L2 budget"),
        _arm("token_position_router_topk1_null", "router_null", budget, "token/position-only sparse router null"),
        _arm("bottleneck_gated_mlp_norm_budgeted", "mlp_control", budget, "MLP control trained under the same L2 and churn budget"),
    ]


def _arm(name: str, family: str, residual_budget: Any, role: str) -> dict[str, Any]:
    return {
        "arm": name,
        "family": family,
        "residual_l2_budget": residual_budget,
        "trainable": family not in {"residual_null"},
        "role": role,
        "required_outputs": "heldout_ce; anchor_kl; logit_mse_vs_base; prediction_flip_rate; residual_update_l2",
    }


def _objective_terms(residual_budget: float | None) -> list[dict[str, Any]]:
    budget = "" if residual_budget is None else residual_budget
    return [
        {
            "term": "supervised_target_ce",
            "weight": 1.0,
            "scope": "target tokens",
            "purpose": "preserve the original residual-adaptation task",
            "gate": "must improve or match dense rank24 only under matched interference metrics",
        },
        {
            "term": "residual_l2_budget_penalty",
            "weight": 1.0,
            "scope": "all residual updates",
            "purpose": "penalize updates above the dense rank24 residual L2 budget",
            "gate": f"mean heldout residual_update_l2 <= {budget}",
        },
        {
            "term": "residual_l2_budget_floor_penalty",
            "weight": 2.0,
            "scope": "all trainable residual updates",
            "purpose": "force sparse and MLP challengers to test a nontrivial dense24-budget operating point instead of winning only by tiny residuals",
            "gate": f"mean heldout residual_update_l2 >= 0.5 * {budget} for scientific advancement",
        },
        {
            "term": "anchor_kl_or_logit_mse_penalty",
            "weight": 0.25,
            "scope": "anchor/off-target rows",
            "purpose": "discourage drift hidden by CE-only reporting",
            "gate": "no worse than dense rank24 at matched CE or matched residual L2",
        },
        {
            "term": "prediction_flip_churn_penalty",
            "weight": 0.25,
            "scope": "anchor and heldout rows",
            "purpose": "control functional churn proxy",
            "gate": "flip rate no worse than dense rank24 unless CE is strictly matched and churn is lower",
        },
        {
            "term": "support_value_diagnostics",
            "weight": 0.0,
            "scope": "sparse arms only",
            "purpose": "separate router support churn from learned value quality",
            "gate": "required diagnostic output, not optimized directly in the first pilot",
        },
    ]


def _gate_rows(
    matched_summary: dict[str, Any],
    available_rows: list[dict[str, str]],
    domination_rows: list[dict[str, str]],
    strategy: dict[str, Any],
    residual_budget: float | None,
) -> list[dict[str, Any]]:
    arms = {row.get("arm") for row in available_rows}
    required = {
        "dense_rank24_best_norm",
        "dense_rank16_best_norm",
        "sparse_contextual_topk2",
        "sparse_rank_matched_topk1",
        "sparse_frequency_matched_random_topk1",
        "parameter_matched_causal_mlp_control",
    }
    return [
        _criterion(
            "matched_decision_passed_artifacts",
            matched_summary.get("status") == "pass",
            "matched sparse/dense/MLP report has passing artifacts",
            matched_summary.get("status", "missing"),
            "matched decision report is missing or failed",
        ),
        _criterion(
            "matched_scientific_gate_blocked",
            matched_summary.get("scientific_gate") == "blocked"
            and int(matched_summary.get("advancement_row_count") or 0) == 0,
            "post-hoc challengers do not clear best-dense Pareto guardrail",
            {
                "scientific_gate": matched_summary.get("scientific_gate", ""),
                "advancement_row_count": matched_summary.get("advancement_row_count", ""),
            },
            "pilot should not be designed from an unresolved or already-promoted matched decision",
        ),
        _criterion(
            "required_source_arms_present",
            required.issubset(arms),
            sorted(required),
            sorted(arms),
            "one or more required sparse/dense/MLP source arms is missing",
        ),
        _criterion(
            "dense_rank24_budget_available",
            residual_budget is not None and residual_budget > 0.0,
            "positive residual-L2 budget from dense rank24 or dense Pareto frontier",
            residual_budget if residual_budget is not None else "",
            "cannot set a norm budget without dense rank24/Pareto residual L2",
        ),
        _criterion(
            "domination_cases_written",
            bool(domination_rows),
            "best-dense domination table exists",
            len(domination_rows),
            "domination cases are missing",
        ),
        _criterion(
            "strategy_review_consumed",
            strategy["present"],
            "latest GPT-5.5-Pro review is read",
            strategy["recommended_next_action"],
            "strategy review missing",
        ),
    ]


def _criterion(criterion: str, passed: bool, threshold: Any, actual: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_evidence.csv", summary["source_evidence"])
    _write_csv(out_dir / "pilot_arms.csv", summary["pilot_arms"])
    _write_csv(out_dir / "objective_terms.csv", summary["objective_terms"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    lines = [
        "# Norm-Budgeted Churn-Regularized Residual Pilot Design",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Residual L2 budget: `{summary['residual_l2_budget']}`",
        f"- Pilot arms: `{len(summary['pilot_arms'])}`",
        f"- Objective terms: `{len(summary['objective_terms'])}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This is a design-only local source-of-truth report. It accepts the latest review's recommendation to train future challengers under an explicit dense-rank24 residual-L2 and churn budget before any GPU validation or mechanism promotion.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _available_arm(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    return next((row for row in rows if row.get("arm") == arm), {})


def _best_dense_budget(rows: list[dict[str, str]]) -> float | None:
    dense = [
        row for row in rows
        if row.get("arm") == "dense_rank24_best_norm" and _float(row.get("residual_update_l2")) is not None
    ]
    if not dense:
        return None
    best = min(dense, key=lambda row: _float(row.get("ce_loss")) or float("inf"))
    return _float(best.get("residual_update_l2"))


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "path": str(path),
        "present": bool(text),
        "strategic_change_level": _header_value(text, "strategic_change_level") or "unknown",
        "notify_ben": (_header_value(text, "notify_ben") or "unknown"),
        "recommended_next_action": _header_value(text, "recommended_next_action") or "",
        "verdict": _header_value(text, "verdict") or "",
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; the design falls back to the matched-decision selected next step."
    return (
        "Accepted the GPT-5.5-Pro recommendation to keep work local, avoid GPU validation, "
        "and define a norm-budgeted, churn-regularized pilot after best-dense Pareto blocking."
    )


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-decision-dir", type=Path, default=DEFAULT_MATCHED_DECISION_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_norm_budgeted_churn_regularized_residual_pilot_design(
        matched_decision_dir=args.matched_decision_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
