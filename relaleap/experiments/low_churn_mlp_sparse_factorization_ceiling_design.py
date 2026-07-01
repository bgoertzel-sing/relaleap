"""Design the local low-churn-MLP sparse-factorization ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CLOSEOUT = Path("results/reports/context_contrastive_core_periphery_closeout/summary.json")
DEFAULT_LOW_CHURN_PILOT = Path("results/reports/low_churn_mlp_residual_control_pilot/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/low_churn_mlp_sparse_factorization_ceiling_design")

IMPLEMENT_ACTION = "implement_low_churn_mlp_sparse_factorization_ceiling_extractor"
REPAIR_ACTION = "repair_sparse_factorization_ceiling_design_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "support_arms.csv",
    "observable_rows.csv",
    "gate_criteria.csv",
    "null_controls.csv",
    "notes.md",
)


def run_low_churn_mlp_sparse_factorization_ceiling_design(
    *,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    low_churn_pilot_path: Path = DEFAULT_LOW_CHURN_PILOT,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a design contract for the sparse factorization ceiling."""

    start = time.time()
    closeout = _read_json(closeout_path)
    low_churn = _read_json(low_churn_pilot_path)
    strategy = _strategy_review(strategy_review_path)
    source_rows = [
        _source("context_contrastive_core_periphery_closeout", closeout_path, closeout),
        _source("low_churn_mlp_residual_control_pilot", low_churn_pilot_path, low_churn),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
        },
    ]
    support_arms = _support_arms()
    observables = _observable_rows()
    null_controls = _null_controls()
    criteria = _gate_criteria(closeout, low_churn, strategy, support_arms, observables, null_controls)
    failures = [row for row in criteria if not row["passed"]]
    source_failures = [
        {"criterion": f"{row['source']}_present", "passed": False, "actual": row["path"], "threshold": "required source must exist"}
        for row in source_rows
        if row["source"] != "strategy_review" and not row["present"]
    ]
    failures.extend(source_failures)
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "low_churn_mlp_sparse_factorization_ceiling_design_recorded"
            if status == "pass"
            else "low_churn_mlp_sparse_factorization_ceiling_design_failed_closed"
        ),
        "claim_status": "design_only_no_sparse_factorization_claim",
        "selected_next_action": IMPLEMENT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement read-only extraction of low-churn teacher residual rows and sparse-ceiling artifact schema before any training"
            if status == "pass"
            else "repair missing closeout or low-churn source artifacts before implementation"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local design and extraction first; GPU validation blocked until ceiling rows and gates exist",
        "ben_should_be_notified": strategy["ben_notification_required"],
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "support_arms": support_arms,
        "observable_rows": observables,
        "null_controls": null_controls,
        "gate_criteria": criteria,
        "failures": failures,
        "rationale": _rationale(),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _support_arms() -> list[dict[str, Any]]:
    return [
        _arm("oracle_support_sparse_ceiling", "oracle", True, "upper bound using teacher residual loss or oracle support utility"),
        _arm("learned_router_sparse_factorization", "learned", True, "deployable prefix-safe support predictor"),
        _arm("token_position_router_sparse_factorization", "token_position", False, "strong shortcut null"),
        _arm("frequency_support_router_sparse_factorization", "frequency", False, "support-frequency null"),
        _arm("random_fixed_support_sparse_factorization", "random", False, "random fixed-support null"),
        _arm("route_scrambled_same_values", "route_scrambled", False, "same values with support mapping scrambled"),
        _arm("shuffled_teacher_residual_sparse_factorization", "shuffled_teacher", False, "misaligned teacher residual null"),
    ]


def _arm(name: str, support_type: str, trainable: bool, role: str) -> dict[str, Any]:
    return {
        "arm": name,
        "support_type": support_type,
        "trainable": trainable,
        "budget_match": "match low-churn MLP residual norm, active params, stored params, and column count where applicable",
        "required_splits": "sequence-heldout and latent-rule-combo-heldout",
        "role": role,
    }


def _observable_rows() -> list[dict[str, str]]:
    return [
        _observable("teacher_residual_reconstruction_mse", "quality", "lower than nulls; report R2 against low-churn teacher residual"),
        _observable("teacher_gap_closure_fraction", "quality", "meaningful closure of low-churn teacher residual CE/reconstruction gap"),
        _observable("heldout_ce_transfer", "quality", "CE guardrail, not sole promotion criterion"),
        _observable("oracle_support_regret", "support", "oracle-vs-learned gap must be explicit"),
        _observable("support_entropy_and_load", "support", "detect collapse and shortcut routing"),
        _observable("functional_churn_kl_and_flip_rate", "interference", "must beat dense/flat/null controls at matched budget"),
        _observable("anchor_kl", "interference", "anchor drift must stay bounded"),
        _observable("finite_update_commutator", "interference", "must beat dense/flat controls"),
        _observable("intervention_fingerprint_specificity", "causal", "necessity, sufficiency, off-target KL, pruning-order deltas"),
    ]


def _observable(metric: str, family: str, gate: str) -> dict[str, str]:
    return {"metric": metric, "family": family, "gate": gate}


def _null_controls() -> list[dict[str, str]]:
    return [
        {"control": "shuffled_or_misaligned_teacher_residual", "reason": "tests whether sparse values simply absorb target scale"},
        {"control": "route_scrambled_same_values", "reason": "tests value dictionary without coherent support"},
        {"control": "frequency_preserving_support_permutation", "reason": "tests support-frequency confounding"},
        {"control": "random_fixed_support", "reason": "tests support choice against random columns"},
        {"control": "token_position_only_router", "reason": "tests shortcut position/token features"},
        {"control": "dense_or_flat_same_budget_residual", "reason": "tests sparse factorization against generic capacity"},
    ]


def _gate_criteria(
    closeout: dict[str, Any],
    low_churn: dict[str, Any],
    strategy: dict[str, Any],
    support_arms: list[dict[str, Any]],
    observables: list[dict[str, Any]],
    null_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_arms = {
        "oracle_support_sparse_ceiling",
        "learned_router_sparse_factorization",
        "token_position_router_sparse_factorization",
        "frequency_support_router_sparse_factorization",
        "random_fixed_support_sparse_factorization",
        "route_scrambled_same_values",
        "shuffled_teacher_residual_sparse_factorization",
    }
    arm_names = {row["arm"] for row in support_arms}
    metric_names = {row["metric"] for row in observables}
    return [
        _criterion(
            "closeout_selected_sparse_factorization_ceiling",
            closeout.get("status") == "pass"
            and closeout.get("selected_next_action") == "design_low_churn_mlp_sparse_factorization_ceiling",
            closeout.get("selected_next_action"),
            "context closeout must select this design",
        ),
        _criterion(
            "low_churn_teacher_artifact_available",
            low_churn.get("status") == "pass"
            and low_churn.get("decision") == "low_churn_mlp_residual_control_pilot_completed",
            low_churn.get("decision"),
            "low-churn MLP pilot must exist as teacher/control source",
        ),
        _criterion(
            "support_arm_coverage_complete",
            required_arms.issubset(arm_names),
            sorted(required_arms - arm_names),
            "oracle, learned, shortcut, random, scrambled, and shuffled-teacher arms are required",
        ),
        _criterion(
            "observable_coverage_complete",
            {
                "teacher_residual_reconstruction_mse",
                "teacher_gap_closure_fraction",
                "functional_churn_kl_and_flip_rate",
                "finite_update_commutator",
                "intervention_fingerprint_specificity",
            }.issubset(metric_names),
            sorted(metric_names),
            "quality, support, interference, and causal observables must be specified",
        ),
        _criterion(
            "null_control_coverage_complete",
            len(null_controls) >= 6,
            len(null_controls),
            "strong null controls must include shuffled, scrambled, frequency, random, shortcut, and dense/flat controls",
        ),
        _criterion(
            "major_strategy_review_recorded",
            strategy["present"] and strategy["ben_notification_required"],
            {
                "strategic_change_level": strategy["strategic_change_level"],
                "notify_ben": strategy["notify_ben"],
                "recommended_next_action": strategy["recommended_next_action"],
            },
            "major/notify-Ben strategy shift must be recorded before implementation",
        ),
    ]


def _criterion(criterion: str, passed: bool, actual: Any, threshold: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "failure_reason": "" if passed else threshold,
    }


def _rationale() -> str:
    return (
        "This is a ceiling/kill-test design, not a promotion claim. Sparse columns must factorize the "
        "already-strong low-churn MLP residual under matched budget, then beat route, teacher, shortcut, "
        "random, dense/flat, churn, commutator, and intervention-fingerprint controls before any GPU path opens."
    )


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
        return "No strategy review was present; design fails closed through source criteria."
    if strategy["ben_notification_required"]:
        return "Accepted the major GPT-5.5-Pro pivot to a local sparse-factorization ceiling; Ben should be notified."
    return "Accepted the external review and preserved local no-GPU design-first handling."


def _source(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "support_arms.csv", summary["support_arms"])
    _write_csv(out_dir / "observable_rows.csv", summary["observable_rows"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_csv(out_dir / "null_controls.csv", summary["null_controls"])
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
            "# Low-Churn MLP Sparse-Factorization Ceiling Design",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Ben should be notified: `{summary['ben_should_be_notified']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "This design keeps GPU validation blocked until sparse factorization rows, strong nulls, churn, commutator, and intervention-fingerprint gates exist locally.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument("--low-churn-pilot", type=Path, default=DEFAULT_LOW_CHURN_PILOT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_low_churn_mlp_sparse_factorization_ceiling_design(
        closeout_path=args.closeout,
        low_churn_pilot_path=args.low_churn_pilot,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
