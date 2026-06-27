"""Backend closeout for contextual-router sequence K-fold ablation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_SUMMARY = Path(
    "results/reports/token_larger_contextual_router_sequence_kfold_ablation/summary.json"
)
DEFAULT_RUNPOD_SUMMARY = Path(
    "results/runpod_fetch/reports/runpod_token_larger_contextual_router_sequence_kfold_ablation/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_contextual_router_sequence_kfold_backend_closeout"
)

SEQUENCE_KFOLD_BACKEND_VALIDATED = "sequence_kfold_backend_validated"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
EXPECTED_DECISION = "causal_contextual_router_sequence_holdout_candidate"


def run_contextual_router_sequence_kfold_backend_closeout_report(
    *,
    local_summary_path: Path = DEFAULT_LOCAL_SUMMARY,
    runpod_summary_path: Path = DEFAULT_RUNPOD_SUMMARY,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Compare local and fetched RunPod K-fold sequence-heldout reports."""

    start = time.time()
    local = _read_json_object(local_summary_path)
    runpod = _read_json_object(runpod_summary_path)
    strategy_review = _strategy_review(strategy_review_path)
    source_rows = [
        _source_row("local_sequence_kfold", local_summary_path, local),
        _source_row("runpod_sequence_kfold", runpod_summary_path, runpod),
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy_review["present"],
            "status": "present" if strategy_review["present"] else "missing_optional",
            "decision": strategy_review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy_review['strategic_change_level']}; "
                f"notify_ben={strategy_review['notify_ben']}"
            ),
        },
    ]
    evidence = _evidence(local, runpod)
    failures = _failures(source_rows, evidence)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = "sequence_kfold_backend_parity_uninterpretable"
        next_step = "repair missing or inconsistent sequence K-fold backend artifacts"
        rationale = (
            "The sequence K-fold backend closeout cannot be interpreted because "
            "the local or fetched RunPod source artifact is missing, failing, or "
            "not making the expected sequence-heldout decision."
        )
    else:
        status = "pass"
        decision = SEQUENCE_KFOLD_BACKEND_VALIDATED
        claim_status = "causal_feature_safe_router_sequence_holdout_backend_validated"
        next_step = (
            "run a bounded post-sequence decision report that chooses whether to "
            "promote the causal-feature-safe contextual router candidate, add "
            "non-CE causal separability controls, or stop this branch"
        )
        rationale = (
            "Local and RunPod sequence-heldout K-fold reports agree that the "
            "causal-feature-safe contextual top-k-2 router is a candidate: it "
            "beats the linear top-k-2 control on all four folds in both backends. "
            "The full-context router remains better on all folds, so this is "
            "backend-validated sequence generalization evidence, not yet a "
            "deployable causal-router promotion."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "source_rows": source_rows,
        "evidence": evidence,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "backend_comparisons_csv": str(out_dir / "backend_comparisons.csv"),
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
        out_dir / "backend_comparisons.csv",
        [
            "comparison",
            "local_mean_loss_delta",
            "local_left_wins",
            "local_right_wins",
            "runpod_mean_loss_delta",
            "runpod_left_wins",
            "runpod_right_wins",
        ],
        evidence["comparison_rows"],
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _evidence(local: dict[str, Any], runpod: dict[str, Any]) -> dict[str, Any]:
    local_ablation = local.get("ablation", {}) if isinstance(local.get("ablation"), dict) else {}
    runpod_ablation = (
        runpod.get("ablation", {}) if isinstance(runpod.get("ablation"), dict) else {}
    )
    rows = []
    for name in (
        "causal_contextual_vs_linear",
        "causal_contextual_vs_full_context_oracle_baseline",
        "full_context_oracle_baseline_vs_linear",
    ):
        local_row = _comparison(local_ablation, name)
        runpod_row = _comparison(runpod_ablation, name)
        rows.append(
            {
                "comparison": name,
                "local_mean_loss_delta": local_row.get("mean_loss_delta"),
                "local_left_wins": local_row.get("left_wins"),
                "local_right_wins": local_row.get("right_wins"),
                "runpod_mean_loss_delta": runpod_row.get("mean_loss_delta"),
                "runpod_left_wins": runpod_row.get("left_wins"),
                "runpod_right_wins": runpod_row.get("right_wins"),
            }
        )
    causal_local = rows[0]
    full_context_local = rows[1]
    return {
        "local_status": local.get("status"),
        "local_decision": local.get("decision"),
        "local_claim_status": local.get("claim_status"),
        "local_cuda_available": local.get("cuda_available"),
        "runpod_status": runpod.get("status"),
        "runpod_decision": runpod.get("decision"),
        "runpod_claim_status": runpod.get("claim_status"),
        "runpod_cuda_available": runpod.get("cuda_available"),
        "fold_count_match": local_ablation.get("fold_count") == runpod_ablation.get("fold_count"),
        "local_fold_count": local_ablation.get("fold_count"),
        "runpod_fold_count": runpod_ablation.get("fold_count"),
        "causal_contextual_beats_linear_both_backends": (
            _lt(causal_local.get("local_mean_loss_delta"), 0.0)
            and _lt(causal_local.get("runpod_mean_loss_delta"), 0.0)
            and causal_local.get("local_left_wins") == 4
            and causal_local.get("runpod_left_wins") == 4
        ),
        "full_context_beats_causal_contextual_both_backends": (
            _gt(full_context_local.get("local_mean_loss_delta"), 0.0)
            and _gt(full_context_local.get("runpod_mean_loss_delta"), 0.0)
            and full_context_local.get("local_right_wins") == 4
            and full_context_local.get("runpod_right_wins") == 4
        ),
        "comparison_rows": rows,
    }


def _comparison(ablation: dict[str, Any], name: str) -> dict[str, Any]:
    comparisons = ablation.get("key_comparisons", {})
    if not isinstance(comparisons, dict):
        return {}
    row = comparisons.get(name, {})
    return row if isinstance(row, dict) else {}


def _failures(
    source_rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:2]:
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
    expected = {
        "local_status": "ok",
        "local_decision": EXPECTED_DECISION,
        "runpod_status": "ok",
        "runpod_decision": EXPECTED_DECISION,
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": expected_value,
                    "actual": evidence.get(field),
                }
            )
    if evidence.get("runpod_cuda_available") is not True:
        failures.append(
            {
                "source": "runpod_sequence_kfold",
                "field": "cuda_available",
                "expected": True,
                "actual": evidence.get("runpod_cuda_available"),
            }
        )
    for field in (
        "fold_count_match",
        "causal_contextual_beats_linear_both_backends",
        "full_context_beats_causal_contextual_both_backends",
    ):
        if evidence.get(field) is not True:
            failures.append(
                {
                    "source": "evidence",
                    "field": field,
                    "expected": True,
                    "actual": evidence.get(field),
                }
            )
    return failures


def _source_row(source: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path),
        "present": path.is_file(),
        "status": packet.get("status"),
        "decision": packet.get("decision"),
        "claim_status": packet.get("claim_status"),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "ben_notification_required": False,
            "incorporation": "optional review not present",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "ben_notification_required": bool(notify_ben) or major,
        "incorporation": (
            "accepted: the review requested sequence-heldout causal-feature "
            "evidence before GPU or distillation work; this closeout records "
            "the local K-fold result and its RunPod repeat"
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) < threshold


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) > threshold


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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Contextual Router Sequence K-fold Backend Closeout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Next Step",
        "",
        summary["next_step"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-summary", type=Path, default=DEFAULT_LOCAL_SUMMARY)
    parser.add_argument("--runpod-summary", type=Path, default=DEFAULT_RUNPOD_SUMMARY)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_contextual_router_sequence_kfold_backend_closeout_report(
        local_summary_path=args.local_summary,
        runpod_summary_path=args.runpod_summary,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "next_step": summary["next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
