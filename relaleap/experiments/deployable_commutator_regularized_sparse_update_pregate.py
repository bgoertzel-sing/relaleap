"""Record a deployable commutator-regularized sparse update pregate."""

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


DEFAULT_SELECTOR = Path("results/reports/post_order_averaging_deployable_mechanism_selector/summary.json")
DEFAULT_ORDER_PROBE = Path("results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json")
DEFAULT_VALUE_PENALTY = Path("results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/deployable_commutator_regularized_sparse_update_pregate")

IMPLEMENT_ACTION = "implement_local_deployable_commutator_regularized_sparse_update_probe"
REPAIR_ACTION = "repair_deployable_commutator_regularized_sparse_update_pregate_sources"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "update_contract.csv",
    "control_arms.csv",
    "observable_gates.csv",
    "kill_thresholds.csv",
    "notes.md",
)


def run_deployable_commutator_regularized_sparse_update_pregate(
    *,
    selector_path: Path = DEFAULT_SELECTOR,
    order_probe_path: Path = DEFAULT_ORDER_PROBE,
    value_penalty_path: Path = DEFAULT_VALUE_PENALTY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a local-only design contract for the next deployable probe."""

    start = time.time()
    selector = _read_json(selector_path)
    order_probe = _read_json(order_probe_path)
    value_penalty = _read_json(value_penalty_path)
    strategy = _strategy_review(strategy_review_path)
    update_contract = _update_contract()
    control_arms = _control_arms()
    observable_gates = _observable_gates()
    kill_thresholds = _kill_thresholds()
    source_rows = [
        _source_row("post_order_averaging_selector", selector_path, selector),
        _source_row("explicit_order_averaging_probe", order_probe_path, order_probe),
        _source_row("commutator_value_penalty_probe", value_penalty_path, value_penalty),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "sha256": _file_sha256(strategy_review_path),
            "status": "read" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}; verdict={strategy['verdict']}"
            ),
            "selected_next_action": "",
        },
    ]
    gate_rows = _gate_rows(selector, order_probe, value_penalty, update_contract, control_arms, observable_gates, kill_thresholds)
    failures = [row for row in gate_rows if not row["passed"]]
    failures.extend(
        {
            "criterion": f"{row['source']}_present",
            "passed": False,
            "actual": row["path"],
            "threshold": "required source artifact must exist and parse",
            "failure_reason": "required source artifact must exist and parse",
        }
        for row in source_rows[:3]
        if not row["present"]
    )
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "deployable_commutator_regularized_sparse_update_pregate_recorded"
            if status == "pass"
            else "deployable_commutator_regularized_sparse_update_pregate_failed_closed"
        ),
        "claim_status": "pregate_only_no_training_or_gpu_claim",
        "selected_next_action": IMPLEMENT_ACTION if status == "pass" else REPAIR_ACTION,
        "selected_next_step": (
            "implement the local CPU probe from update_contract.csv with dense/flat/random-support/no-update controls and order averaging as an upper-bound control"
            if status == "pass"
            else "repair missing or inconsistent pregate source artifacts before implementing the probe"
        ),
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "backend_policy": "local CPU pregate/probe only; RunPod and Colab remain blocked",
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "source_rows": source_rows,
        "gate_rows": gate_rows,
        "update_contract": update_contract,
        "control_arms": control_arms,
        "observable_gates": observable_gates,
        "kill_thresholds": kill_thresholds,
        "failures": failures,
        "rationale": _rationale(status, selector, order_probe, value_penalty),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _update_contract() -> list[dict[str, str]]:
    return [
        {
            "component": "candidate_update_rule",
            "deployability": "deployable_forward_only",
            "contract": "apply task A then task B with a forward-only sparse-column update; add a local commutator surrogate penalty computed from current-batch support/value Jacobian proxies, without evaluating the reversed update at deployment",
            "required_outputs": "ce; residual_norm; parameter_commutator; logit_mse_commutator; logit_kl_commutator; forgetting; anchor_kl; support_overlap_bin",
        },
        {
            "component": "nondeployable_upper_bound",
            "deployability": "diagnostic_only",
            "contract": "retain explicit forward/reverse order averaging only as an upper-bound control, never as promotion evidence",
            "required_outputs": "same metrics plus order_averaging_ratio",
        },
        {
            "component": "budget_matching",
            "deployability": "required_for_claim",
            "contract": "match active columns, stored values, top-k, residual L2 mean/p95, active/stored parameter budgets, splits, and seeds across sparse, dense, flat, random-support, and no-update arms",
            "required_outputs": "active_params; stored_params; active_rank_proxy; residual_norm_mean; residual_norm_p95",
        },
        {
            "component": "support_overlap_strata",
            "deployability": "required_for_claim",
            "contract": "report low/medium/high support-overlap bins so commutator gains cannot hide in easy disjoint-support cases",
            "required_outputs": "support_overlap_bin; support_churn; load_entropy; per_bin_gate_status",
        },
    ]


def _control_arms() -> list[dict[str, str]]:
    return [
        _control("sparse_commutator_regularized_update", "candidate", "deployable forward-only sparse update with commutator surrogate"),
        _control("sparse_unregularized_update", "sparse baseline", "same support/value budget without commutator regularization"),
        _control("explicit_order_averaged_sparse_update", "upper-bound control", "nondeployable forward/reverse averaged update"),
        _control("dense_active_matched_update", "dense control", "matched active parameters and residual norm"),
        _control("dense_stored_matched_update", "dense control", "matched stored parameters and residual norm"),
        _control("same_router_flat_value_update", "flat value control", "same support policy with noncolumnar value capacity"),
        _control("random_support_sparse_update", "sparse null", "random supports with matched top-k and value budget"),
        _control("no_update", "null", "zero learning/reference commutator floor"),
    ]


def _control(arm: str, family: str, purpose: str) -> dict[str, str]:
    return {"arm": arm, "family": family, "purpose": purpose, "required": "true"}


def _observable_gates() -> list[dict[str, str]]:
    return [
        _observable("ce_guardrail", "candidate must not collapse CE versus unregularized sparse and dense/flat controls"),
        _observable("residual_norm_parity", "candidate residual norm mean and p95 must stay within matched control band"),
        _observable("parameter_commutator_norm", "candidate must reduce normalized parameter commutator beyond dense/flat/random-support controls"),
        _observable("behavioral_logit_commutator_mse", "candidate must reduce logit MSE commutator beyond dense/flat/random-support controls"),
        _observable("behavioral_logit_commutator_kl", "candidate must reduce symmetric logit KL commutator beyond dense/flat/random-support controls"),
        _observable("old_task_forgetting", "candidate must improve or match old-task forgetting at CE/norm parity"),
        _observable("anchor_kl_drift", "candidate must not increase anchor KL drift versus matched controls"),
        _observable("support_overlap_bin_coverage", "low, medium, and high overlap bins must all be populated or the report fails closed"),
    ]


def _observable(metric: str, gate: str) -> dict[str, str]:
    return {"metric": metric, "promotion_gate": gate, "required": "true"}


def _kill_thresholds() -> list[dict[str, str]]:
    return [
        _threshold("generic_smoothing_null_wins", "kill", "dense/flat/random-support regularized controls match candidate within tolerance"),
        _threshold("ce_collapse", "kill", "candidate CE worsens beyond guardrail"),
        _threshold("norm_shrinkage_explains_gain", "repair", "commutator gain coincides with materially lower residual norm or active budget"),
        _threshold("no_update_floor_confusion", "repair", "no-update has lowest commutator but no learning; report separates diagnostic floor from mechanism evidence"),
        _threshold("missing_support_overlap_bins", "repair", "support-overlap strata are absent or underpopulated"),
        _threshold("order_averaging_mislabeled_deployable", "kill", "explicit order averaging is used as candidate promotion evidence"),
    ]


def _threshold(name: str, disposition: str, condition: str) -> dict[str, str]:
    return {"threshold": name, "disposition": disposition, "condition": condition, "required": "true"}


def _gate_rows(
    selector: dict[str, Any],
    order_probe: dict[str, Any],
    value_penalty: dict[str, Any],
    update_contract: list[dict[str, str]],
    control_arms: list[dict[str, str]],
    observable_gates: list[dict[str, str]],
    kill_thresholds: list[dict[str, str]],
) -> list[dict[str, Any]]:
    arms = {row["arm"] for row in control_arms}
    metrics = {row["metric"] for row in observable_gates}
    components = {row["component"] for row in update_contract}
    return [
        _criterion(
            "selector_selected_this_pregate",
            selector.get("status") == "pass"
            and selector.get("selected_next_action") == "design_deployable_commutator_regularized_sparse_update_pregate"
            and selector.get("requires_gpu_now") is False
            and selector.get("advance_to_gpu_validation") is False,
            selector.get("selected_next_action"),
            "post-order-averaging selector must select this local pregate with GPU blocked",
        ),
        _criterion(
            "order_averaging_available_but_nondeployable",
            order_probe.get("status") == "pass"
            and order_probe.get("decision") == "explicit_order_averaging_diagnostic_candidate_not_promoted",
            order_probe.get("decision"),
            "explicit order averaging must remain diagnostic-only source evidence",
        ),
        _criterion(
            "simple_value_penalty_not_duplicated",
            value_penalty.get("decision") == "commutator_value_penalty_not_established",
            value_penalty.get("decision"),
            "simple value penalty branch must be closed before designing a different update-rule pregate",
        ),
        _criterion(
            "deployable_update_contract_complete",
            {"candidate_update_rule", "nondeployable_upper_bound", "budget_matching", "support_overlap_strata"}.issubset(components),
            sorted(components),
            "candidate, upper bound, budget, and support-overlap contract rows are required",
        ),
        _criterion(
            "strong_control_coverage_complete",
            {
                "sparse_unregularized_update",
                "explicit_order_averaged_sparse_update",
                "dense_active_matched_update",
                "dense_stored_matched_update",
                "same_router_flat_value_update",
                "random_support_sparse_update",
                "no_update",
            }.issubset(arms),
            sorted(arms),
            "dense, flat, random-support, no-update, sparse baseline, and upper-bound controls are required",
        ),
        _criterion(
            "observable_coverage_complete",
            {
                "ce_guardrail",
                "residual_norm_parity",
                "parameter_commutator_norm",
                "behavioral_logit_commutator_mse",
                "behavioral_logit_commutator_kl",
                "old_task_forgetting",
                "anchor_kl_drift",
                "support_overlap_bin_coverage",
            }.issubset(metrics),
            sorted(metrics),
            "CE, norm, commutator, forgetting, drift, and support-overlap observables are required",
        ),
        _criterion(
            "kill_thresholds_fail_closed",
            len(kill_thresholds) >= 6 and all(row["disposition"] in {"kill", "repair"} for row in kill_thresholds),
            len(kill_thresholds),
            "pregate must specify fail-closed kill/repair thresholds",
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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file() and bool(payload),
        "sha256": _file_sha256(path),
        "status": payload.get("status", "missing" if not path.is_file() else ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
        "selected_next_action": payload.get("selected_next_action", ""),
    }


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
    if strategy["ben_notification_required"]:
        return "Read the external review and preserved the Ben-notification requirement; GPU remains blocked."
    return "Read the external review; accepted the local commutator-mechanism direction while avoiding duplicate order-averaging/value-penalty work."


def _header_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines()[:20]:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _rationale(status: str, selector: dict[str, Any], order_probe: dict[str, Any], value_penalty: dict[str, Any]) -> str:
    if status != "pass":
        return "The pregate could not verify its selector/probe sources or required control contract, so it fails closed."
    return (
        "The selector picked a deployable commutator-regularized sparse update after explicit order averaging was "
        f"closed as diagnostic-only ({order_probe.get('decision', '')}) and simple value penalties failed "
        f"({value_penalty.get('decision', '')}). This artifact records the local CPU probe contract selected by "
        f"{selector.get('decision', '')}; it adds no training evidence and permits no GPU promotion."
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "update_contract.csv", summary["update_contract"])
    _write_csv(out_dir / "control_arms.csv", summary["control_arms"])
    _write_csv(out_dir / "observable_gates.csv", summary["observable_gates"])
    _write_csv(out_dir / "kill_thresholds.csv", summary["kill_thresholds"])
    (out_dir / "notes.md").write_text(_notes(summary), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
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
            "# Deployable Commutator-Regularized Sparse Update Pregate",
            "",
            f"- Status: `{summary['status']}`",
            f"- Decision: `{summary['decision']}`",
            f"- Selected action: `{summary['selected_next_action']}`",
            f"- Requires GPU now: `{summary['requires_gpu_now']}`",
            f"- Next step: {summary['selected_next_step']}",
            "",
            "GPU validation remains blocked. This artifact is a local design contract, not training or promotion evidence.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_deployable_commutator_regularized_sparse_update_pregate(out_dir=args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_next_action": summary["selected_next_action"],
                "requires_gpu_now": summary["requires_gpu_now"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
