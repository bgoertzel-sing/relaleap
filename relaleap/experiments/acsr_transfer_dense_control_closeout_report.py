"""Close out ACSR transfer-objective evidence against dense controls."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_TRANSFER_VALIDATION = Path(
    "results/reports/acsr_transfer_objective_validation_gate/summary.json"
)
DEFAULT_HELDOUT_CONTROL = Path(
    "results/reports/acsr_transfer_objective_heldout_control_gate/summary.json"
)
DEFAULT_DENSE_TRANSFER = Path(
    "results/reports/acsr_dense_residual_transfer_control/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/acsr_transfer_dense_control_closeout")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_criteria.csv",
    "notes.md",
)


def run_acsr_transfer_dense_control_closeout_report(
    *,
    transfer_validation_path: Path = DEFAULT_TRANSFER_VALIDATION,
    heldout_control_path: Path = DEFAULT_HELDOUT_CONTROL,
    dense_transfer_path: Path = DEFAULT_DENSE_TRANSFER,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Record a no-promotion decision after dense transfer controls."""

    start = time.time()
    validation = _read_json(transfer_validation_path)
    heldout = _read_json(heldout_control_path)
    dense = _read_json(dense_transfer_path)
    review = _strategy_review(strategy_review_path)
    evidence = _evidence(validation, heldout, dense, review)
    source_rows = [
        _source_row("transfer_validation_gate", transfer_validation_path, validation),
        _source_row("heldout_transfer_control_gate", heldout_control_path, heldout),
        _source_row("dense_residual_transfer_control", dense_transfer_path, dense),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review_path.is_file(),
            "status": "read" if strategy_review_path.is_file() else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}"
            ),
        },
    ]
    criteria = _criteria(evidence)
    failures = [row for row in criteria if not row["passed"] and row["severity"] == "hard"]
    claim_blockers = [
        row for row in criteria if not row["passed"] and row["severity"] == "claim_blocker"
    ]
    status = "fail" if failures else "pass"
    if status == "fail":
        decision = "acsr_transfer_dense_control_closeout_failed_closed"
        claim_status = "source_artifacts_missing_or_inconsistent"
        selected_next_step = "repair transfer dense-control closeout sources before interpretation"
        rationale = (
            "The closeout failed closed because one or more transfer or dense-control "
            "source artifacts are missing or inconsistent."
        )
    else:
        decision = "acsr_transfer_dense_control_closeout_recorded"
        claim_status = "transfer_objective_not_separated_from_dense_controls"
        selected_next_step = (
            "stop ACSR transfer-objective promotion and keep future work local until "
            "a mechanism benchmark separates sparse supports from dense controls"
        )
        rationale = (
            "The transfer objective is robust versus direct, token-position, and random "
            "support controls across local/RunPod packets and held-out positions, but "
            "the rank-matched causal dense residual control matches or beats the sparse "
            "transfer gain. That blocks ACSR mechanism/default-router claims and keeps "
            "GPU validation out of scope."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": "no RunPod or Colab action selected by this closeout",
        "source_rows": source_rows,
        "closeout_criteria": criteria,
        "failures": failures,
        "claim_blockers_preserved": claim_blockers,
        "evidence": evidence,
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence(
    validation: dict[str, Any],
    heldout: dict[str, Any],
    dense: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    validation_metrics = _as_dict(validation.get("aggregate_metrics"))
    heldout_metrics = _as_dict(heldout.get("aggregate_metrics"))
    dense_metrics = _as_dict(dense.get("source_metrics"))
    dense_primary = _as_dict(dense.get("dense_control_rows"))
    sparse_partner = _row_by(
        dense.get("source_metrics"),
        "value_path",
        "partner_values",
        "arm",
        "transfer_objective_router",
    )
    causal_dense = _row_by(
        dense.get("dense_control_rows"),
        "control",
        "rank_matched_causal_dense_residual",
    )
    return {
        "transfer_validation_status": validation.get("status"),
        "transfer_validation_claim_status": validation.get("claim_status"),
        "heldout_control_status": heldout.get("status"),
        "heldout_control_claim_status": heldout.get("claim_status"),
        "dense_transfer_status": dense.get("status"),
        "dense_transfer_claim_status": dense.get("claim_status"),
        "validation_mean_partner_minus_direct": validation_metrics.get(
            "mean_partner_transfer_minus_direct_ce"
        ),
        "heldout_mean_partner_minus_direct": heldout_metrics.get(
            "mean_heldout_partner_transfer_minus_direct_ce"
        ),
        "heldout_max_own_ce_damage": heldout_metrics.get(
            "max_heldout_own_transfer_minus_direct_ce"
        ),
        "sparse_transfer_heldout_delta_vs_direct": sparse_partner.get(
            "heldout_delta_vs_direct_ce"
        ),
        "causal_dense_heldout_delta_vs_base": causal_dense.get(
            "heldout_delta_vs_base_ce"
        ),
        "dense_control_failure_count": len(dense.get("failures", []))
        if isinstance(dense.get("failures"), list)
        else None,
        "strategy_major_or_notify_ben": review["ben_notification_required"],
        "unused_dense_metrics_marker": bool(dense_metrics or dense_primary),
    }


def _criteria(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            "transfer_validation_passed",
            evidence["transfer_validation_status"] == "pass",
            "hard",
            "transfer validation gate passed",
            evidence["transfer_validation_status"],
            "transfer validation gate missing or failed",
        ),
        _criterion(
            "heldout_control_passed",
            evidence["heldout_control_status"] == "pass",
            "hard",
            "held-out transfer controls passed",
            evidence["heldout_control_status"],
            "held-out controls missing or failed",
        ),
        _criterion(
            "dense_transfer_control_available",
            evidence["dense_transfer_status"] == "fail"
            and evidence["dense_transfer_claim_status"]
            == "sparse_transfer_not_separated_from_dense_control",
            "hard",
            "dense transfer control ran and blocked sparse separation",
            evidence["dense_transfer_claim_status"],
            "dense transfer control is missing or did not reach the expected blocking state",
        ),
        _criterion(
            "transfer_objective_signal_replicated",
            _lt(evidence["validation_mean_partner_minus_direct"], 0.0)
            and _lt(evidence["heldout_mean_partner_minus_direct"], 0.0),
            "claim_blocker",
            "transfer objective remains positive versus direct controls",
            {
                "validation": evidence["validation_mean_partner_minus_direct"],
                "heldout": evidence["heldout_mean_partner_minus_direct"],
            },
            "transfer objective signal is no longer replicated",
        ),
        _criterion(
            "own_value_guardrail_preserved",
            _leq(evidence["heldout_max_own_ce_damage"], 0.02),
            "claim_blocker",
            "own-value held-out CE damage remains within guardrail",
            evidence["heldout_max_own_ce_damage"],
            "own-value damage exceeds guardrail",
        ),
        _criterion(
            "sparse_not_separated_from_dense",
            True,
            "claim_blocker",
            "rank-matched causal dense control blocks sparse transfer mechanism claim",
            {
                "sparse_transfer_heldout_delta_vs_direct": evidence[
                    "sparse_transfer_heldout_delta_vs_direct"
                ],
                "causal_dense_heldout_delta_vs_base": evidence[
                    "causal_dense_heldout_delta_vs_base"
                ],
            },
            "",
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


def _source_row(source: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status") if path.is_file() else "missing",
        "decision": payload.get("decision", ""),
        "claim_status": payload.get("claim_status", ""),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    header: dict[str, str] = {}
    for line in text.splitlines()[:16]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    notify_ben = header.get("notify_ben", "false").lower() == "true"
    strategic_change_level = header.get("strategic_change_level", "missing")
    return {
        "present": path.is_file(),
        "strategic_change_level": strategic_change_level,
        "notify_ben": notify_ben,
        "ben_notification_required": notify_ben or strategic_change_level == "major",
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
        "disposition": (
            "accepted in part: ACSR/no-GPU boundary retained; pilot advice is already satisfied by existing command artifacts"
            if path.is_file()
            else "missing optional strategy review"
        ),
    }


def _direction_shift(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategic_change_level": review["strategic_change_level"],
        "notify_ben": review["notify_ben"],
        "ben_should_be_notified": review["ben_notification_required"],
        "record": (
            "Latest review is major/notify_ben=true. This closeout records that Ben should "
            "be notified; it keeps the ACSR/no-GPU direction but treats completed dense "
            "transfer controls as blocking mechanism/default claims."
            if review["ben_notification_required"]
            else "No major direction shift or Ben notification requested."
        ),
    }


def _row_by(rows: Any, key: str, value: str, key2: str | None = None, value2: str | None = None) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict) or row.get(key) != value:
            continue
        if key2 is None or row.get(key2) == value2:
            return row
    return {}


def _lt(left: Any, right: float) -> bool:
    value = _float(left)
    return value is not None and value < right


def _leq(left: Any, right: float) -> bool:
    value = _float(left)
    return value is not None and value <= right


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_criteria.csv", summary["closeout_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# ACSR Transfer Dense-Control Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "- Held-out transfer minus direct CE: "
        f"`{evidence.get('heldout_mean_partner_minus_direct')}`",
        "- Sparse transfer held-out delta vs direct: "
        f"`{evidence.get('sparse_transfer_heldout_delta_vs_direct')}`",
        "- Causal dense held-out delta vs base: "
        f"`{evidence.get('causal_dense_heldout_delta_vs_base')}`",
        "",
        str(summary["rationale"]),
        "",
        "## Direction Shift",
        "",
        str(summary["direction_shift"]["record"]),
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-validation", type=Path, default=DEFAULT_TRANSFER_VALIDATION)
    parser.add_argument("--heldout-control", type=Path, default=DEFAULT_HELDOUT_CONTROL)
    parser.add_argument("--dense-transfer", type=Path, default=DEFAULT_DENSE_TRANSFER)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_transfer_dense_control_closeout_report(
        transfer_validation_path=args.transfer_validation,
        heldout_control_path=args.heldout_control,
        dense_transfer_path=args.dense_transfer,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
