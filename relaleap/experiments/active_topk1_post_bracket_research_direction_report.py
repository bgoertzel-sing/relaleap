"""Post-bracket research-direction report for the active top-k-1 packet."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


DEFAULT_BACKEND_PROVENANCE_DIR = Path(
    "results/reports/token_larger_active_topk1_backend_provenance_manifest"
)
DEFAULT_FUNCTIONAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_SINGLETON_RECONCILIATION_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_post_bracket_research_direction"
)

POST_BRACKET_DIRECTION_SELECTED = "post_bracket_direction_selected"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SELECTED_EXPERIMENT = "context_conditioned_singleton_interference_decomposition"
BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED = (
    "broad_reusable_singleton_claim_excluded"
)

_SELECTED_EXPERIMENT_ROWS = (
    {
        "component": "routed_baseline",
        "purpose": "anchor the promoted contextual-router prediction and CE guardrail",
    },
    {
        "component": "no_residual_baseline",
        "purpose": "measure residual-column net gain relative to the frozen base path",
    },
    {
        "component": "own_context_forced_active_singleton",
        "purpose": "estimate selected in-context singleton gain",
    },
    {
        "component": "off_context_forced_singleton_matched",
        "purpose": "estimate singleton reuse harm under token/position/context matching where available",
    },
    {
        "component": "context_gated_singleton_predictor_holdout",
        "purpose": "test whether a learned context gate preserves singleton gains while suppressing off-context harm",
    },
    {
        "component": "matched_topk2_reference",
        "purpose": "retain promoted top-k-2 as a guardrail/reference condition only",
    },
    {
        "component": "random_support_control",
        "purpose": "separate singleton efficacy from chance support selection",
    },
    {
        "component": "dense_rank_matched_control",
        "purpose": "check whether retention comes from low active rank rather than columnar context gating",
    },
)

_ESTIMANDS = (
    "within_context_singleton_gain",
    "off_context_singleton_harm",
    "context_gated_net_gain",
    "retention_at_matched_transfer_ce_improvement",
    "anchor_task_drift",
    "logit_churn",
    "functional_churn",
    "support_jaccard_distance",
    "per_column_marginal_churn",
    "topk2_at_least_one_original_column_retained_rate",
)


def run_active_topk1_post_bracket_research_direction_report(
    *,
    backend_provenance_dir: Path = DEFAULT_BACKEND_PROVENANCE_DIR,
    functional_retention_dir: Path = DEFAULT_FUNCTIONAL_RETENTION_DIR,
    singleton_reconciliation_dir: Path = DEFAULT_SINGLETON_RECONCILIATION_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Select the next bounded causal-retention experiment from existing packets."""

    start = time.time()
    backend = _read_json_object(backend_provenance_dir / "summary.json")
    retention = _read_json_object(functional_retention_dir / "summary.json")
    singleton = _read_json_object(singleton_reconciliation_dir / "summary.json")
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = _source_rows(
        backend_provenance_dir=backend_provenance_dir,
        functional_retention_dir=functional_retention_dir,
        singleton_reconciliation_dir=singleton_reconciliation_dir,
        backend=backend,
        retention=retention,
        singleton=singleton,
        strategy_review=strategy_review,
    )
    failures = _failures(
        backend=backend,
        retention=retention,
        singleton=singleton,
        source_rows=source_rows,
    )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        selected_experiment = None
        claim_policy = INSUFFICIENT_EVIDENCE
        rationale = (
            "The post-bracket direction report could not select the next experiment "
            "because one or more prerequisite active top-k-1 packets are missing, "
            "failing, or inconsistent with the bracket-only/off-context-interference "
            "interpretation."
        )
        next_step = (
            "repair the failing source packet before selecting a new causal-retention "
            "experiment"
        )
    else:
        status = "pass"
        decision = POST_BRACKET_DIRECTION_SELECTED
        selected_experiment = SELECTED_EXPERIMENT
        claim_policy = BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
        rationale = (
            "The provenance-anchored active top-k-1 packet supports a contextual "
            "low-churn functional-retention bracket, but the singleton reconciliation "
            "shows positive selected in-context singleton gain together with harmful "
            "forced off-context singleton reuse. The next scientifically coherent "
            "step is therefore an interference decomposition that treats column plus "
            "context gate as the live hypothesis and explicitly excludes a broad "
            "reusable singleton claim."
        )
        next_step = (
            "implement and run the bounded context-conditioned singleton interference "
            "decomposition with matched top-k-2, random-support, and dense/rank-matched "
            "controls"
        )

    summary = {
        "status": status,
        "decision": decision,
        "selected_experiment": selected_experiment,
        "claim_policy": claim_policy,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "strategy_review": strategy_review,
        "experiment_design": {
            "components": list(_SELECTED_EXPERIMENT_ROWS),
            "estimands": list(_ESTIMANDS),
            "requires_gpu_now": False,
            "new_training_required_for_this_report": False,
            "backend_policy": (
                "No GPU validation is required for this report. Spend backend time "
                "only after the local decomposition implementation produces a "
                "claim-changing result."
            ),
        },
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "selected_experiment_csv": str(out_dir / "selected_experiment.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_selected_experiment(out_dir / "selected_experiment.csv")
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    *,
    backend_provenance_dir: Path,
    functional_retention_dir: Path,
    singleton_reconciliation_dir: Path,
    backend: dict[str, Any],
    retention: dict[str, Any],
    singleton: dict[str, Any],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "backend_provenance_manifest",
            "path": str(backend_provenance_dir / "summary.json"),
            "present": (backend_provenance_dir / "summary.json").is_file(),
            "status": backend.get("status"),
            "decision": backend.get("decision"),
            "claim_status": None,
        },
        {
            "source": "functional_retention_audit",
            "path": str(functional_retention_dir / "summary.json"),
            "present": (functional_retention_dir / "summary.json").is_file(),
            "status": retention.get("status"),
            "decision": retention.get("decision"),
            "claim_status": retention.get("claim_status"),
        },
        {
            "source": "singleton_reconciliation_audit",
            "path": str(singleton_reconciliation_dir / "summary.json"),
            "present": (singleton_reconciliation_dir / "summary.json").is_file(),
            "status": singleton.get("status"),
            "decision": singleton.get("decision"),
            "claim_status": singleton.get("decision"),
        },
        {
            "source": "strategy_review",
            "path": strategy_review.get("path"),
            "present": strategy_review.get("present"),
            "status": "present" if strategy_review.get("present") else "missing_optional",
            "decision": strategy_review.get("recommended_next_action"),
            "claim_status": (
                f"strategic_change_level={strategy_review.get('strategic_change_level')}; "
                f"notify_ben={strategy_review.get('notify_ben')}"
            ),
        },
    ]


def _failures(
    *,
    backend: dict[str, Any],
    retention: dict[str, Any],
    singleton: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in source_rows[:3]:
        if not row["present"]:
            failures.append(
                {
                    "source": row["source"],
                    "field": "summary_json",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": row["path"],
                }
            )
    expectations = (
        (
            "backend_provenance_manifest",
            backend,
            "decision",
            ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
        ),
        (
            "functional_retention_audit",
            retention,
            "decision",
            FUNCTIONAL_RETENTION_BRACKET_ONLY,
        ),
        (
            "functional_retention_audit",
            retention,
            "claim_status",
            CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        ),
        (
            "singleton_reconciliation_audit",
            singleton,
            "decision",
            CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        ),
    )
    for source, packet, field, expected in expectations:
        if packet.get(field) != expected:
            failures.append(
                {
                    "source": source,
                    "field": field,
                    "expected": expected,
                    "actual": packet.get(field),
                }
            )
    for source, packet in (
        ("backend_provenance_manifest", backend),
        ("functional_retention_audit", retention),
        ("singleton_reconciliation_audit", singleton),
    ):
        if packet.get("status") != "pass":
            failures.append(
                {
                    "source": source,
                    "field": "status",
                    "expected": "pass",
                    "actual": packet.get("status"),
                }
            )
    return failures


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
        }
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {}
    for line in lines[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"strategic_change_level", "notify_ben", "recommended_next_action"}:
            header[key] = value.strip()
    notify_ben = _bool_or_none(header.get("notify_ben"))
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "followed: selected the context-conditioned singleton interference "
            "decomposition and deferred GPU work until local claim-changing evidence"
        ),
        "ben_notification_required": bool(notify_ben)
        or header.get("strategic_change_level") == "major",
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


def _write_selected_experiment(path: Path) -> None:
    fieldnames = ["component", "purpose"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_SELECTED_EXPERIMENT_ROWS)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Active Top-k-1 Post-Bracket Research Direction",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Selected experiment: `{summary['selected_experiment']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Source Packets",
    ]
    for row in summary["source_rows"]:
        lines.append(
            f"- `{row['source']}`: present=`{row['present']}`, status=`{row['status']}`, "
            f"decision=`{row['decision']}`"
        )
    lines.extend(
        [
            "",
            "## Direction",
            "",
            summary["rationale"],
            "",
            "## Selected Experiment Components",
        ]
    )
    for row in _SELECTED_EXPERIMENT_ROWS:
        lines.append(f"- `{row['component']}`: {row['purpose']}")
    lines.extend(
        [
            "",
            "## Estimands",
        ]
    )
    for estimand in _ESTIMANDS:
        lines.append(f"- `{estimand}`")
    lines.extend(
        [
            "",
            "## Strategy Review",
            "",
            f"- Present: `{summary['strategy_review']['present']}`",
            f"- Strategic change level: `{summary['strategy_review']['strategic_change_level']}`",
            f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
            f"- Incorporation: {summary['strategy_review']['incorporation']}",
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-provenance-dir", type=Path, default=DEFAULT_BACKEND_PROVENANCE_DIR
    )
    parser.add_argument(
        "--functional-retention-dir", type=Path, default=DEFAULT_FUNCTIONAL_RETENTION_DIR
    )
    parser.add_argument(
        "--singleton-reconciliation-dir",
        type=Path,
        default=DEFAULT_SINGLETON_RECONCILIATION_DIR,
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_post_bracket_research_direction_report(
        backend_provenance_dir=args.backend_provenance_dir,
        functional_retention_dir=args.functional_retention_dir,
        singleton_reconciliation_dir=args.singleton_reconciliation_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "selected_experiment": summary["selected_experiment"],
                "claim_policy": summary["claim_policy"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
