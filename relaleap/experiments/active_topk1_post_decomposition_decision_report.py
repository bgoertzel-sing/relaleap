"""Post-decomposition decision report for active top-k-1 singleton gating."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)


DEFAULT_INTERFERENCE_DIR = Path(
    "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference"
)
DEFAULT_BACKEND_PROVENANCE_DIR = Path(
    "results/reports/token_larger_active_topk1_backend_provenance_manifest"
)
DEFAULT_FUNCTIONAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_post_decomposition_decision"
)

POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED = (
    "post_decomposition_runpod_validation_recommended"
)
POST_DECOMPOSITION_BACKEND_VALIDATION_RECOMMENDED = (
    "post_decomposition_backend_validation_recommended"
)
POST_DECOMPOSITION_LOCAL_ONLY = "post_decomposition_local_only"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED = "broad_reusable_singleton_claim_excluded"
COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS = "column_plus_context_gate_hypothesis"


def run_active_topk1_post_decomposition_decision_report(
    *,
    interference_dir: Path = DEFAULT_INTERFERENCE_DIR,
    backend_provenance_dir: Path = DEFAULT_BACKEND_PROVENANCE_DIR,
    functional_retention_dir: Path = DEFAULT_FUNCTIONAL_RETENTION_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
    gpu_backend: str | None = None,
) -> dict[str, Any]:
    """Decide whether the local singleton-gating packet warrants backend validation."""

    start = time.time()
    gpu_backend = (gpu_backend or os.environ.get("RELALEAP_GPU_BACKEND") or "unset").strip()
    interference = _read_json_object(interference_dir / "summary.json")
    provenance = _read_json_object(backend_provenance_dir / "summary.json")
    retention = _read_json_object(functional_retention_dir / "summary.json")
    strategy_review = _strategy_review(strategy_review_path)

    source_rows = _source_rows(
        interference_dir=interference_dir,
        backend_provenance_dir=backend_provenance_dir,
        functional_retention_dir=functional_retention_dir,
        interference=interference,
        provenance=provenance,
        retention=retention,
        strategy_review=strategy_review,
    )
    metrics = interference.get("evidence", {}).get("metrics", {})
    signals = interference.get("evidence", {}).get("signals", {})
    failures = _failures(
        source_rows=source_rows,
        interference=interference,
        provenance=provenance,
        retention=retention,
        metrics=metrics,
        signals=signals,
    )
    validation_warrant = _validation_warrant(metrics, signals)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        selected_next_step = "repair_missing_or_inconsistent_post_decomposition_sources"
        backend_plan = _backend_plan(gpu_backend, selected=False)
        rationale = (
            "The post-decomposition decision report could not select a backend "
            "validation step because one or more required local source packets are "
            "missing, failing, or do not carry the context-gated interference result."
        )
    elif validation_warrant:
        status = "pass"
        decision = (
            POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED
            if gpu_backend == "runpod"
            else POST_DECOMPOSITION_BACKEND_VALIDATION_RECOMMENDED
        )
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_next_step = "bounded_backend_validation_of_context_gated_singleton_packet"
        backend_plan = _backend_plan(gpu_backend, selected=True)
        rationale = (
            "The local no-training decomposition is claim-changing enough to warrant "
            "one bounded backend validation packet: own-context selected singletons "
            "have positive gain, off-context forced singleton reuse is harmful, and "
            "the simple context gate preserves positive holdout gain while improving "
            "over ungated forced reuse. The claim remains column plus context gate; "
            "a broad reusable singleton claim is still excluded."
        )
    else:
        status = "pass"
        decision = POST_DECOMPOSITION_LOCAL_ONLY
        claim_status = COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS
        selected_next_step = "keep_context_gated_singleton_packet_local_only"
        backend_plan = _backend_plan(gpu_backend, selected=False)
        rationale = (
            "The local decomposition packet is present, but it is not strong enough "
            "under the configured gate to spend backend validation time. Keep it as "
            "local source-artifact evidence and refresh local controls before any "
            "GPU validation."
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        "selected_next_step": selected_next_step,
        "gpu_backend": gpu_backend,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "decision_gate": {
            "validation_warrant": validation_warrant,
            "required_signals": [
                "own_context_singleton_gain_positive",
                "offcontext_singleton_interference_present",
                "context_gate_holdout_net_gain_positive",
                "context_gate_improves_over_ungated_holdout",
                "matched_topk2_reference_present",
                "random_control_present",
                "exhaustive_control_present",
            ],
            "metrics": metrics,
            "signals": signals,
        },
        "backend_plan": backend_plan,
        "strategy_review": strategy_review,
        "failures": failures,
        "rationale": rationale,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "decision_sources_csv": str(out_dir / "decision_sources.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_decision_sources(out_dir / "decision_sources.csv", source_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _source_rows(
    *,
    interference_dir: Path,
    backend_provenance_dir: Path,
    functional_retention_dir: Path,
    interference: dict[str, Any],
    provenance: dict[str, Any],
    retention: dict[str, Any],
    strategy_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "context_conditioned_singleton_interference_audit",
            "path": str(interference_dir / "summary.json"),
            "present": (interference_dir / "summary.json").is_file(),
            "status": interference.get("status"),
            "decision": interference.get("decision"),
            "claim_status": interference.get("claim_policy"),
        },
        {
            "source": "backend_provenance_manifest",
            "path": str(backend_provenance_dir / "summary.json"),
            "present": (backend_provenance_dir / "summary.json").is_file(),
            "status": provenance.get("status"),
            "decision": provenance.get("decision"),
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
    source_rows: list[dict[str, Any]],
    interference: dict[str, Any],
    provenance: dict[str, Any],
    retention: dict[str, Any],
    metrics: dict[str, Any],
    signals: dict[str, Any],
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
            "context_conditioned_singleton_interference_audit",
            interference,
            "status",
            "pass",
        ),
        (
            "context_conditioned_singleton_interference_audit",
            interference,
            "decision",
            CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
        ),
        (
            "backend_provenance_manifest",
            provenance,
            "status",
            "pass",
        ),
        (
            "backend_provenance_manifest",
            provenance,
            "decision",
            ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
        ),
        (
            "functional_retention_audit",
            retention,
            "status",
            "pass",
        ),
        (
            "functional_retention_audit",
            retention,
            "decision",
            FUNCTIONAL_RETENTION_BRACKET_ONLY,
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
    if not metrics:
        failures.append(
            {
                "source": "context_conditioned_singleton_interference_audit",
                "field": "evidence.metrics",
                "expected": "nonempty metrics",
                "actual": metrics,
            }
        )
    if not signals:
        failures.append(
            {
                "source": "context_conditioned_singleton_interference_audit",
                "field": "evidence.signals",
                "expected": "nonempty signals",
                "actual": signals,
            }
        )
    return failures


def _validation_warrant(metrics: dict[str, Any], signals: dict[str, Any]) -> bool:
    required_signals = (
        "own_context_singleton_gain_positive",
        "offcontext_singleton_interference_present",
        "context_gate_holdout_net_gain_positive",
        "context_gate_improves_over_ungated_holdout",
        "matched_topk2_reference_present",
        "random_control_present",
        "exhaustive_control_present",
    )
    if not all(bool(signals.get(signal)) for signal in required_signals):
        return False
    return (
        _gt(metrics.get("own_context_singleton_gain_mean"), 0.0)
        and _lt(metrics.get("off_context_singleton_gain_mean"), 0.0)
        and _gt(metrics.get("context_gated_net_gain_holdout_mean"), 0.0)
        and _gt(metrics.get("context_gate_gain_minus_ungated_holdout_mean"), 0.0)
    )


def _backend_plan(gpu_backend: str, *, selected: bool) -> dict[str, Any]:
    if not selected:
        return {
            "requires_backend_validation": False,
            "backend": gpu_backend,
            "commands": [],
            "blocker": None,
        }
    if gpu_backend == "runpod":
        return {
            "requires_backend_validation": True,
            "backend": "runpod",
            "commands": [
                "./.venv-conda/bin/python tools/runpod_ssh_runner.py bootstrap",
                (
                    "./.venv-conda/bin/python tools/runpod_ssh_runner.py run --command "
                    "'python -m relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit "
                    "--out results/audits/runpod_token_larger_active_topk1_context_conditioned_singleton_interference && "
                    "python -m relaleap.experiments.active_topk1_post_decomposition_decision_report "
                    "--interference-dir results/audits/runpod_token_larger_active_topk1_context_conditioned_singleton_interference "
                    "--out results/reports/runpod_token_larger_active_topk1_post_decomposition_decision'"
                ),
                "./.venv-conda/bin/python tools/runpod_ssh_runner.py fetch",
                (
                    "./.venv-conda/bin/python -m relaleap.experiments.active_topk1_post_decomposition_decision_report "
                    "--interference-dir results/runpod_fetch/audits/runpod_token_larger_active_topk1_context_conditioned_singleton_interference "
                    "--out results/reports/local_checked_runpod_active_topk1_post_decomposition_decision"
                ),
            ],
            "blocker": None,
        }
    if gpu_backend == "colab":
        return {
            "requires_backend_validation": True,
            "backend": "colab",
            "commands": [
                (
                    "./.venv-conda/bin/python tools/colab_playwright_runner.py "
                    "--cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut "
                    "--wait-completion --debug-snapshot"
                )
            ],
            "blocker": None,
        }
    return {
        "requires_backend_validation": True,
        "backend": gpu_backend,
        "commands": [],
        "blocker": (
            "backend validation is scientifically warranted, but RELALEAP_GPU_BACKEND "
            "is local-only or unset; stop after recording this report"
        ),
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": None,
            "notify_ben": None,
            "recommended_next_action": None,
            "incorporation": "optional review not present",
            "ben_notification_required": False,
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
    major = header.get("strategic_change_level") == "major"
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level"),
        "notify_ben": notify_ben,
        "recommended_next_action": header.get("recommended_next_action"),
        "incorporation": (
            "followed: the recommended context-conditioned singleton interference "
            "decomposition was run locally; this report now treats the positive "
            "context-gated packet as warranting one bounded backend validation while "
            "still excluding broad reusable singleton claims"
        ),
        "ben_notification_required": bool(notify_ben) or major,
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


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) > threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (float, int)) and float(value) < threshold


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


def _write_decision_sources(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source", "path", "present", "status", "decision", "claim_status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["decision_gate"].get("metrics", {})
    backend_plan = summary["backend_plan"]
    lines = [
        "# Active Top-k-1 Post-Decomposition Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- GPU backend: `{summary['gpu_backend']}`",
        f"- Backend validation required: `{backend_plan['requires_backend_validation']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Key Metrics",
        "",
        f"- Own-context singleton gain mean: `{metrics.get('own_context_singleton_gain_mean')}`",
        f"- Off-context singleton gain mean: `{metrics.get('off_context_singleton_gain_mean')}`",
        f"- Context-gated holdout net gain: `{metrics.get('context_gated_net_gain_holdout_mean')}`",
        f"- Context gate minus ungated holdout: `{metrics.get('context_gate_gain_minus_ungated_holdout_mean')}`",
        f"- Top-k-2 reference gain mean: `{metrics.get('topk2_reference_gain_mean')}`",
        f"- Random singleton gain mean: `{metrics.get('random_singleton_gain_mean')}`",
        f"- Exhaustive singleton gain mean: `{metrics.get('exhaustive_singleton_gain_mean')}`",
        "",
        "## Rationale",
        "",
        summary["rationale"],
        "",
        "## Backend Plan",
        "",
    ]
    if backend_plan["commands"]:
        for command in backend_plan["commands"]:
            lines.append(f"- `{command}`")
    else:
        lines.append(f"- Blocker: `{backend_plan['blocker']}`")
    lines.extend(
        [
            "",
            "## Strategy Review",
            "",
            f"- Present: `{summary['strategy_review']['present']}`",
            f"- Strategic change level: `{summary['strategy_review']['strategic_change_level']}`",
            f"- Notify Ben: `{summary['strategy_review']['notify_ben']}`",
            f"- Ben notification required: `{summary['strategy_review']['ben_notification_required']}`",
            f"- Incorporation: {summary['strategy_review']['incorporation']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interference-dir", type=Path, default=DEFAULT_INTERFERENCE_DIR)
    parser.add_argument(
        "--backend-provenance-dir", type=Path, default=DEFAULT_BACKEND_PROVENANCE_DIR
    )
    parser.add_argument(
        "--functional-retention-dir", type=Path, default=DEFAULT_FUNCTIONAL_RETENTION_DIR
    )
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gpu-backend", default=None)
    args = parser.parse_args(argv)
    summary = run_active_topk1_post_decomposition_decision_report(
        interference_dir=args.interference_dir,
        backend_provenance_dir=args.backend_provenance_dir,
        functional_retention_dir=args.functional_retention_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
        gpu_backend=args.gpu_backend,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "claim_status": summary["claim_status"],
                "gpu_backend": summary["gpu_backend"],
                "selected_next_step": summary["selected_next_step"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
