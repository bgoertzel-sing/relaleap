"""Close out the current sparse target-adaptation rescue branch."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("results/reports/sparse_target_adaptation_rescue_closeout")
DEFAULT_MECHANISM_PROBE = Path(
    "results/reports/mechanism_factorized_continual_learning_probe/summary.json"
)
DEFAULT_RESCUE_PROBE = Path(
    "results/reports/sparse_target_adaptation_rescue_probe/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "closeout_rows.csv",
    "notes.md",
)

RETIRE_CURRENT_RESCUE = "current_topk2_rescue_path_retired"
INSUFFICIENT_EVIDENCE = "sparse_rescue_closeout_insufficient_evidence"


def run_sparse_target_adaptation_rescue_closeout_report(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    mechanism_probe_path: Path = DEFAULT_MECHANISM_PROBE,
    rescue_probe_path: Path = DEFAULT_RESCUE_PROBE,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
) -> dict[str, Any]:
    """Record a bounded no-promotion decision for the current sparse rescue path."""

    start = time.time()
    mechanism = _read_json_object(mechanism_probe_path)
    rescue = _read_json_object(rescue_probe_path)
    review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("mechanism_factorized_cl_probe", mechanism_probe_path, mechanism),
        _source_row("sparse_target_adaptation_rescue_probe", rescue_probe_path, rescue),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": review["present"],
            "status": "present" if review["present"] else "missing_optional",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence_snapshot(mechanism, rescue, review)
    closeout_rows = _closeout_rows(evidence)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "source_artifacts_missing_or_inconsistent"
        selected_next_step = "repair sparse rescue closeout source artifacts before interpretation"
        rationale = "Required local source artifacts are missing or do not contain the expected negative-claim statuses."
    else:
        status = "pass"
        decision = RETIRE_CURRENT_RESCUE
        claim_status = "topk2_value_lr_or_focal_rescue_not_established"
        selected_next_step = (
            "design a mechanistically different sparse-retention objective with "
            "off-target anchors before any RunPod or Colab validation"
        )
        rationale = (
            "The mechanism-factorized CL probe showed sparse top-k1 protected "
            "off-target CE/KL and forgetting but trailed dense target adaptation. "
            "The top-k2 value-LR/focal rescue then improved target adaptation yet "
            "did not preserve the original sparse off-target-KL advantage. The "
            "current rescue family should therefore be retired rather than repeated."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": "local closeout only; no RunPod/Colab spend for retired negative branch",
        "source_rows": source_rows,
        "closeout_rows": closeout_rows,
        "evidence": evidence,
        "strategy_review": review,
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _evidence_snapshot(
    mechanism: dict[str, Any],
    rescue: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    mechanism_result = _as_dict(mechanism.get("primary_result"))
    rescue_result = _as_dict(rescue.get("primary_result"))
    return {
        "mechanism_status": mechanism.get("status"),
        "mechanism_decision": mechanism.get("decision"),
        "mechanism_claim_status": mechanism.get("claim_status"),
        "mechanism_topk1_minus_dense_target_ce_delta": mechanism_result.get(
            "topk1_minus_dense_mean_target_ce_delta"
        ),
        "mechanism_topk1_minus_dense_off_target_kl": mechanism_result.get(
            "topk1_minus_dense_mean_off_target_kl"
        ),
        "mechanism_topk1_minus_dense_final_forgetting": mechanism_result.get(
            "topk1_minus_dense_mean_final_forgetting"
        ),
        "rescue_status": rescue.get("status"),
        "rescue_decision": rescue.get("decision"),
        "rescue_claim_status": rescue.get("claim_status"),
        "rescue_best_arm": rescue_result.get("best_rescue_arm"),
        "rescue_best_minus_dense_target_ce_delta": rescue_result.get(
            "best_rescue_minus_dense_target_ce_delta"
        ),
        "rescue_best_minus_topk2_off_target_kl": rescue_result.get(
            "best_rescue_minus_topk2_off_target_kl"
        ),
        "strategy_recommendation_incorporated": review.get("present")
        and review.get("strategic_change_level") != "major"
        and not review.get("notify_ben"),
        "ben_notification_required": review.get("ben_notification_required", False),
    }


def _closeout_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch": "mechanism_factorized_sparse_retention",
            "source_decision": evidence["mechanism_decision"],
            "key_metric": "topk1_minus_dense_target_ce_delta",
            "key_value": evidence["mechanism_topk1_minus_dense_target_ce_delta"],
            "disposition": "blocked_by_dense_target_adaptation_gap",
        },
        {
            "branch": "topk2_value_lr_or_focal_rescue",
            "source_decision": evidence["rescue_decision"],
            "key_metric": "best_rescue_minus_topk2_off_target_kl",
            "key_value": evidence["rescue_best_minus_topk2_off_target_kl"],
            "disposition": "retired_off_target_kl_advantage_not_preserved",
        },
        {
            "branch": "gpu_validation",
            "source_decision": "local_closeout",
            "key_metric": "requires_gpu_now",
            "key_value": False,
            "disposition": "blocked_until_local_mechanistic_rescue_exists",
        },
    ]


def _failures(source_rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows:
        if row["source"] == "strategy_review":
            continue
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "source_artifact",
                    "reason": "required source artifact missing",
                }
            )
        elif row["status"] != "pass":
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "reason": f"expected pass status, observed {row['status']}",
                }
            )
    expected_claims = {
        "mechanism_claim_status": "mechanism_factorized_sparse_retention_not_established",
        "rescue_claim_status": "sparse_target_adaptation_rescue_not_established",
    }
    for field, expected in expected_claims.items():
        if evidence.get(field) != expected:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "reason": f"expected {expected}, observed {evidence.get(field)}",
                }
            )
    if evidence.get("ben_notification_required"):
        failures.append(
            {
                "source": "strategy_review",
                "field": "ben_notification_required",
                "reason": "strategy review requests Ben notification before treating this as routine closeout",
            }
        )
    return failures


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "closeout_rows.csv", summary["closeout_rows"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    evidence = summary["evidence"]
    lines = [
        "# Sparse Target-Adaptation Rescue Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Best rescue arm: `{evidence.get('rescue_best_arm')}`",
        "- Mechanism top-k1 minus dense target CE delta: "
        f"`{evidence.get('mechanism_topk1_minus_dense_target_ce_delta')}`",
        "- Mechanism top-k1 minus dense off-target KL: "
        f"`{evidence.get('mechanism_topk1_minus_dense_off_target_kl')}`",
        "- Rescue best minus top-k2 off-target KL: "
        f"`{evidence.get('rescue_best_minus_topk2_off_target_kl')}`",
        "",
        str(summary["rationale"]),
        "",
        "## Next Step",
        "",
        str(summary["selected_next_step"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status", "missing"),
        "decision": packet.get("decision", ""),
        "claim_status": packet.get("claim_status", ""),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    header: dict[str, str] = {}
    for line in text.splitlines()[:12]:
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
        "recommendation_disposition": (
            "incorporated: local closeout preserves confounded-slice label and continues mechanism-controlled work"
            if path.is_file()
            else "missing optional strategy review"
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mechanism-probe", type=Path, default=DEFAULT_MECHANISM_PROBE)
    parser.add_argument("--rescue-probe", type=Path, default=DEFAULT_RESCUE_PROBE)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    args = parser.parse_args(argv)
    summary = run_sparse_target_adaptation_rescue_closeout_report(
        out_dir=args.out,
        mechanism_probe_path=args.mechanism_probe,
        rescue_probe_path=args.rescue_probe,
        strategy_review_path=args.strategy_review,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
