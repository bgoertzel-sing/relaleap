"""Close out ACSR dense-control retention/churn evidence after commutator assay."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_SUPPORT_HEAD_CLOSEOUT = Path(
    "results/reports/acsr_support_head_closeout_redirect/summary.json"
)
DEFAULT_COMMUTATOR_ASSAY = Path(
    "results/reports/acsr_finite_update_commutator_assay/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_dense_control_retention_churn_closeout")

SUPPORT_DISCOVERY_FROZEN = "acsr_dense_control_retention_churn_closeout_selected"
INSUFFICIENT_EVIDENCE = "acsr_dense_control_retention_churn_closeout_failed_closed"
NEXT_PATH = "dense_residual_control_baseline_retention_churn_synthesis"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_criteria.csv",
    "metrics.csv",
    "notes.md",
)


def run_acsr_dense_control_retention_churn_closeout_report(
    *,
    support_head_closeout_path: Path = DEFAULT_SUPPORT_HEAD_CLOSEOUT,
    commutator_assay_path: Path = DEFAULT_COMMUTATOR_ASSAY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Consume local ACSR closeout artifacts and select the next non-CE path."""

    start = time.time()
    support_closeout = _read_json_object(support_head_closeout_path)
    commutator = _read_json_object(commutator_assay_path)
    strategy = _strategy_review(strategy_review_path)
    metrics = _metrics(support_closeout, commutator)
    criteria = _criteria(metrics)
    hard_failures = [
        row for row in criteria if not row["passed"] and row["severity"] == "hard"
    ]
    claim_blockers = [
        row for row in criteria if not row["passed"] and row["severity"] == "claim_blocker"
    ]

    if hard_failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_next_step = None
        rationale = (
            "The dense-control retention/churn closeout cannot be interpreted because "
            "one or more required source reports are missing or not in the expected "
            "local closeout state."
        )
    else:
        status = "pass"
        decision = SUPPORT_DISCOVERY_FROZEN
        selected_next_step = NEXT_PATH
        rationale = (
            "Deployable ACSR support discovery remains frozen: the null-complete "
            "support-head closeout already blocked the claim, and the finite-update "
            "commutator assay found only a tiny absolute sparse commutator. Sparse "
            "ACSR is lower than the dense causal control on logit-MSE order "
            "sensitivity, but the magnitude is too small to promote a mechanism "
            "claim. Dense residual controls stay as the active comparison baseline."
        )

    source_rows = [
        _source_row("support_head_closeout", support_head_closeout_path, support_closeout),
        _source_row("finite_update_commutator_assay", commutator_assay_path, commutator),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "present" if strategy["present"] else "missing_optional",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]
    summary = {
        "status": status,
        "decision": decision,
        "selected_next_step": selected_next_step,
        "next_step": (
            "write a command-driven dense-residual-control retention/churn synthesis "
            "that compares existing sparse ACSR, rank-matched sparse, and dense "
            "control evidence before any new GPU validation"
            if status == "pass"
            else "repair or rerun the missing local source reports"
        ),
        "claim_statuses": {
            "deployable_support_discovery": (
                "frozen_negative_tiny_headroom_sequence_holdout_and_tiny_commutator"
                if status == "pass"
                else "not_interpretable"
            ),
            "sparse_support_identity": (
                "retired_by_upstream_gate" if status == "pass" else "not_interpretable"
            ),
            "finite_update_commutator": (
                "tiny_absolute_signal_blocks_mechanism_claim"
                if status == "pass"
                else "not_interpretable"
            ),
            "runpod_validation": "deferred_no_gpu_target",
            "ben_notification": "required" if strategy["ben_notification_required"] else "not_required",
        },
        "source_rows": source_rows,
        "closeout_criteria": criteria,
        "failures": hard_failures,
        "claim_blockers_preserved": claim_blockers,
        "metrics": metrics,
        "strategy_review": strategy,
        "strategy_review_handling": _strategy_review_handling(strategy),
        "rationale": rationale,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _metrics(support_closeout: dict[str, Any], commutator: dict[str, Any]) -> dict[str, Any]:
    support_metrics = _as_dict(support_closeout.get("metrics"))
    commutator_metrics = _as_dict(commutator.get("metrics"))
    return {
        "support_closeout_status": support_closeout.get("status"),
        "support_closeout_decision": support_closeout.get("decision"),
        "support_closeout_selected_next_action": support_closeout.get("selected_next_action"),
        "deployable_support_discovery_status": _as_dict(
            support_closeout.get("claim_statuses")
        ).get("deployable_support_discovery"),
        "learned_head_delta_vs_router": _float_or_none(
            support_metrics.get("learned_head_holdout_delta_vs_router")
        ),
        "upstream_oracle_ce_headroom": _float_or_none(
            support_metrics.get("upstream_oracle_ce_headroom")
        ),
        "sequence_head_delta_vs_router": _float_or_none(
            support_metrics.get("sequence_head_delta_vs_router")
        ),
        "commutator_status": commutator.get("status"),
        "commutator_decision": commutator.get("decision"),
        "commutator_claim_status": commutator.get("claim_status"),
        "sparse_mean_logit_mse": _float_or_none(commutator_metrics.get("sparse_mean_logit_mse")),
        "dense_mean_logit_mse": _float_or_none(commutator_metrics.get("dense_mean_logit_mse")),
        "sparse_minus_dense_logit_mse": _float_or_none(
            commutator_metrics.get("sparse_minus_dense_logit_mse")
        ),
        "sparse_support_churn_fraction": _float_or_none(
            commutator_metrics.get("sparse_support_churn_fraction")
        ),
        "sparse_mean_ce_abs_delta": _float_or_none(
            commutator_metrics.get("sparse_mean_ce_abs_delta")
        ),
    }


def _criteria(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "support_head_closeout_present",
            metrics["support_closeout_status"] == "pass",
            "hard",
            "support-head closeout report passed",
            metrics["support_closeout_status"],
            "support-head closeout is missing or failed",
        ),
        _criterion(
            "support_discovery_frozen",
            metrics["support_closeout_decision"]
            == "acsr_support_head_negative_closeout_redirect_selected",
            "hard",
            "support-head closeout selected a negative redirect",
            metrics["support_closeout_decision"],
            "support-head closeout is not the expected negative redirect",
        ),
        _criterion(
            "commutator_assay_interpretable",
            metrics["commutator_claim_status"]
            == "finite_update_commutator_too_small_for_sparse_mechanism_claim",
            "hard",
            "finite-update assay completed and blocked on tiny commutator magnitude",
            metrics["commutator_claim_status"],
            "finite-update commutator assay is missing, failed closed, or positive",
        ),
        _criterion(
            "oracle_headroom_still_tiny",
            _greater_than(metrics["upstream_oracle_ce_headroom"], -0.01),
            "claim_blocker",
            "oracle support CE headroom remains below the material threshold",
            metrics["upstream_oracle_ce_headroom"],
            "oracle support headroom is no longer tiny; reroute through strategy review",
        ),
        _criterion(
            "sequence_holdout_still_unfavorable",
            _greater_than(metrics["sequence_head_delta_vs_router"], 0.0),
            "claim_blocker",
            "sequence-heldout support head remains worse than router",
            metrics["sequence_head_delta_vs_router"],
            "sequence-heldout support head is not unfavorable",
        ),
        _criterion(
            "sparse_commutator_absolute_signal_tiny",
            _less_than(metrics["sparse_mean_logit_mse"], 0.01),
            "claim_blocker",
            "sparse finite-update logit-MSE commutator is below material threshold",
            metrics["sparse_mean_logit_mse"],
            "sparse commutator is material enough to require a different interpretation",
        ),
        _criterion(
            "dense_control_remains_active_baseline",
            _less_than(metrics["sparse_minus_dense_logit_mse"], 0.0),
            "claim_blocker",
            "sparse has lower order-sensitivity than dense but not enough absolute signal to promote",
            metrics["sparse_minus_dense_logit_mse"],
            "sparse did not beat the dense causal control on logit-MSE order sensitivity",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    severity: str,
    requirement: str,
    observed: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "severity": severity,
        "requirement": requirement,
        "observed": observed,
        "failure_reason": "" if passed else failure_reason,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": "",
            "notify_ben": "",
            "recommended_next_action": "",
            "verdict": "",
            "ben_notification_required": False,
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "ben_notification_required": (
            header.get("notify_ben", "").lower() == "true"
            or header.get("strategic_change_level", "").lower() == "major"
        ),
    }


def _strategy_review_handling(strategy: dict[str, Any]) -> str:
    if not strategy["present"]:
        return "No strategy review was present; closeout relies on command artifacts only."
    if strategy["ben_notification_required"]:
        return (
            "Latest GPT-5.5-Pro review required Ben notification or marked a major "
            "shift; this report records that requirement while keeping the run local."
        )
    return (
        "Latest GPT-5.5-Pro review was read. Its local null-completion recommendation "
        "has already been incorporated by the support-head gate; this closeout defers "
        "additional support-discovery work because the subsequent commutator signal is tiny."
    )


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", ""),
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_criteria.csv", summary["closeout_criteria"])
    _write_csv(out_dir / "metrics.csv", [summary["metrics"]])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# ACSR Dense-Control Retention/Churn Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Oracle CE headroom: `{metrics['upstream_oracle_ce_headroom']}`",
        f"- Sequence-head delta vs router: `{metrics['sequence_head_delta_vs_router']}`",
        f"- Sparse commutator logit MSE: `{metrics['sparse_mean_logit_mse']}`",
        f"- Sparse minus dense logit MSE: `{metrics['sparse_minus_dense_logit_mse']}`",
        f"- Sparse support churn fraction: `{metrics['sparse_support_churn_fraction']}`",
        "",
        summary["rationale"],
        "",
        summary["strategy_review_handling"],
        "",
        "RunPod remains deferred because this closeout is a local artifact synthesis and does not define a GPU validation target.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['criterion']}`: {failure['failure_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _greater_than(value: Any, threshold: float) -> bool:
    number = _float_or_none(value)
    return number is not None and number > threshold


def _less_than(value: Any, threshold: float) -> bool:
    number = _float_or_none(value)
    return number is not None and number < threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-head-closeout", type=Path, default=DEFAULT_SUPPORT_HEAD_CLOSEOUT)
    parser.add_argument("--commutator-assay", type=Path, default=DEFAULT_COMMUTATOR_ASSAY)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_dense_control_retention_churn_closeout_report(
        support_head_closeout_path=args.support_head_closeout,
        commutator_assay_path=args.commutator_assay,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
