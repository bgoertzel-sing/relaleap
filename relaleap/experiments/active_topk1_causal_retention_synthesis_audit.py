"""Causal-retention synthesis audit for the active top-k-1 bracket."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
)
from relaleap.experiments.active_topk1_context_gate_suppression_calibration_audit import (
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED,
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_retention_functional_churn_followup_report import (
    RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


DEFAULT_RETENTION_FOLLOWUP_DIR = Path(
    "results/reports/token_larger_active_topk1_retention_functional_churn_followup"
)
DEFAULT_FUNCTIONAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_SINGLETON_RECONCILIATION_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit"
)
DEFAULT_INTERFERENCE_DIR = Path(
    "results/audits/token_larger_active_topk1_context_conditioned_singleton_interference"
)
DEFAULT_GATE_CALIBRATION_DIR = Path(
    "results/audits/token_larger_active_topk1_context_gate_suppression_calibration"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_causal_retention_synthesis"
)

CAUSAL_RETENTION_CLAIM_BLOCKED_DEPLOYABLE_GATE = (
    "causal_retention_claim_blocked_by_deployable_gate"
)
CAUSAL_RETENTION_CLAIM_SUPPORTED = "causal_retention_claim_supported"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_causal_retention_synthesis_audit(
    *,
    retention_followup_dir: Path = DEFAULT_RETENTION_FOLLOWUP_DIR,
    functional_retention_dir: Path = DEFAULT_FUNCTIONAL_RETENTION_DIR,
    singleton_reconciliation_dir: Path = DEFAULT_SINGLETON_RECONCILIATION_DIR,
    interference_dir: Path = DEFAULT_INTERFERENCE_DIR,
    gate_calibration_dir: Path = DEFAULT_GATE_CALIBRATION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Synthesize retention/churn and singleton-causal packets without retraining."""

    start = time.time()
    packets = {
        "retention_followup": _packet(
            retention_followup_dir,
            RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED,
        ),
        "functional_retention": _packet(
            functional_retention_dir,
            FUNCTIONAL_RETENTION_BRACKET_ONLY,
        ),
        "singleton_reconciliation": _packet(
            singleton_reconciliation_dir,
            CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        ),
        "context_conditioned_interference": _packet(
            interference_dir,
            CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
        ),
        "context_gate_calibration": _packet(
            gate_calibration_dir,
            (CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED, CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED),
        ),
    }
    source_rows = [_source_row(name, packet) for name, packet in packets.items()]
    failures = _failures(source_rows)
    evidence = _evidence(packets)
    signals = _signals(evidence, packets)

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        claim_status = INSUFFICIENT_EVIDENCE
        rationale = (
            "The causal-retention synthesis cannot be interpreted because one or "
            "more required source packets is missing, failing, or has an unexpected "
            "decision label."
        )
        next_step = "repair_missing_or_inconsistent_active_topk1_causal_retention_sources"
    elif signals["causal_retention_claim_supported"]:
        status = "pass"
        decision = CAUSAL_RETENTION_CLAIM_SUPPORTED
        claim_status = CAUSAL_RETENTION_CLAIM_SUPPORTED
        rationale = (
            "The active rank-matched contextual top-k-1 bracket satisfies the "
            "retention/churn controls, context-gated singleton efficacy evidence, "
            "and deployable context-gate calibration criteria. This supports a "
            "bounded causal-retention claim under matched local contexts."
        )
        next_step = (
            "run one backend-stable repeat before promoting the causal-retention "
            "claim beyond the local command-generated packet"
        )
    else:
        status = "pass"
        decision = CAUSAL_RETENTION_CLAIM_BLOCKED_DEPLOYABLE_GATE
        claim_status = "local_retention_bracket_with_context_gated_singleton_efficacy_only"
        rationale = (
            "The active top-k-1 bracket has strong local retention/churn evidence "
            "and the singleton evidence is positive when conditioned on its own "
            "router context. However, the no-training deployable context gate did "
            "not beat its pre-registered suppression and random-control criteria, "
            "so the broad reusable causal-retention claim remains blocked. The "
            "scientifically valid interpretation is a local low-churn retention "
            "bracket plus a context-gated singleton-efficacy diagnostic."
        )
        next_step = (
            "return the main architecture loop to contextual top-k-2 support "
            "routing, keeping active top-k-1 as a retention/churn control and "
            "diagnostic causal-retention bracket"
        )

    summary = {
        "status": status,
        "decision": decision,
        "claim_status": claim_status,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "source_rows": source_rows,
        "evidence": evidence,
        "signals": signals,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "source_rows_csv": str(out_dir / "source_rows.csv"),
            "evidence_rows_csv": str(out_dir / "evidence_rows.csv"),
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
        ["source", "path", "present", "status", "decision", "expected_decision"],
        source_rows,
    )
    _write_csv(
        out_dir / "evidence_rows.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in sorted(evidence["metrics"].items())],
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _packet(path: Path, expected_decision: str | tuple[str, ...]) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json_object(summary_path)
    expected = (expected_decision,) if isinstance(expected_decision, str) else expected_decision
    return {
        "dir": path,
        "summary_path": summary_path,
        "summary": summary,
        "expected_decision": expected,
    }


def _source_row(name: str, packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet["summary"]
    return {
        "source": name,
        "path": str(packet["summary_path"]),
        "present": packet["summary_path"].is_file(),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "expected_decision": ",".join(packet["expected_decision"]),
    }


def _failures(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in source_rows:
        expected = set(str(row["expected_decision"]).split(","))
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
            continue
        if row["status"] != "pass":
            failures.append(
                {
                    "source": row["source"],
                    "field": "status",
                    "expected": "pass",
                    "actual": row["status"],
                }
            )
        if row["decision"] not in expected:
            failures.append(
                {
                    "source": row["source"],
                    "field": "decision",
                    "expected": sorted(expected),
                    "actual": row["decision"],
                }
            )
    return failures


def _evidence(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    retention = packets["retention_followup"]["summary"]
    functional = packets["functional_retention"]["summary"]
    reconciliation = packets["singleton_reconciliation"]["summary"]
    interference = packets["context_conditioned_interference"]["summary"]
    calibration = packets["context_gate_calibration"]["summary"]

    retention_aggregates = retention.get("aggregates", {})
    functional_evidence = functional.get("evidence", {})
    functional_aggregates = functional_evidence.get("aggregates", {})
    functional_signals = functional_evidence.get("claim_signals", {})
    reconciliation_metrics = reconciliation.get("evidence", {}).get("metrics", {})
    reconciliation_signals = reconciliation.get("evidence", {}).get("signals", {})
    interference_metrics = interference.get("evidence", {}).get("metrics", {})
    interference_signals = interference.get("evidence", {}).get("signals", {})
    calibration_metrics = calibration.get("evidence", {}).get("metrics", {})
    calibration_signals = calibration.get("evidence", {}).get("signals", {})

    metrics = {
        "min_support_churn_advantage_topk1_vs_topk2": retention_aggregates.get(
            "min_support_churn_advantage_topk1_vs_topk2"
        ),
        "min_commutator_anchor_advantage_topk1_vs_topk2": retention_aggregates.get(
            "min_commutator_anchor_advantage_topk1_vs_topk2"
        ),
        "min_transfer_advantage_topk1_vs_dense": retention_aggregates.get(
            "min_transfer_advantage_topk1_vs_dense"
        ),
        "mean_topk1_anchor_support_churn_after_transfer": retention_aggregates.get(
            "mean_topk1_anchor_support_churn_after_transfer"
        ),
        "mean_topk2_anchor_support_churn_after_transfer": retention_aggregates.get(
            "mean_topk2_anchor_support_churn_after_transfer"
        ),
        "functional_mean_commutator_anchor_advantage_topk1_vs_dense": functional_aggregates.get(
            "mean_commutator_anchor_logit_mse_advantage_topk1_vs_dense"
        ),
        "selected_singleton_gain_mean": reconciliation_metrics.get(
            "selected_singleton_gain_mean"
        ),
        "offcontext_fixed_dominant_singleton_gain_mean": reconciliation_metrics.get(
            "offcontext_fixed_dominant_singleton_gain_mean"
        ),
        "context_gated_net_gain_holdout_mean": interference_metrics.get(
            "context_gated_net_gain_holdout_mean"
        ),
        "context_gate_gain_minus_ungated_holdout_mean": interference_metrics.get(
            "context_gate_gain_minus_ungated_holdout_mean"
        ),
        "deployable_holdout_net_gain": calibration_metrics.get(
            "deployable_holdout_net_gain"
        ),
        "deployable_gain_minus_ungated": calibration_metrics.get(
            "deployable_gain_minus_ungated"
        ),
        "deployable_gain_minus_coverage_matched_random": calibration_metrics.get(
            "deployable_gain_minus_coverage_matched_random"
        ),
        "deployable_offcontext_harm_suppression_fraction": calibration_metrics.get(
            "deployable_offcontext_harm_suppression_fraction"
        ),
    }
    return {
        "metrics": metrics,
        "source_signals": {
            "retention_branch_supported": retention.get("signals", {}).get(
                "branch_supported"
            ),
            "functional_retention_claim_supported": functional_signals.get(
                "claim_supported"
            ),
            "functional_offcontext_interference_present": functional_signals.get(
                "offcontext_singleton_interference_present"
            ),
            "selected_incontext_singleton_gain_positive": reconciliation_signals.get(
                "selected_incontext_positive"
            ),
            "offcontext_fixed_dominant_negative": reconciliation_signals.get(
                "offcontext_fixed_dominant_negative"
            ),
            "context_gate_holdout_net_gain_positive": interference_signals.get(
                "context_gate_holdout_net_gain_positive"
            ),
            "context_gate_improves_over_ungated_holdout": interference_signals.get(
                "context_gate_improves_over_ungated_holdout"
            ),
            "deployable_gate_passes_pre_registered_criteria": calibration_signals.get(
                "deployable_gate_passes_pre_registered_criteria"
            ),
            "deployable_improves_over_ungated": calibration_signals.get(
                "deployable_improves_over_ungated"
            ),
            "deployable_suppresses_offcontext_harm": calibration_signals.get(
                "deployable_suppresses_offcontext_harm"
            ),
        },
    }


def _signals(evidence: dict[str, Any], packets: dict[str, dict[str, Any]]) -> dict[str, bool]:
    source = evidence["source_signals"]
    retention_ready = all(
        source.get(field) is True
        for field in (
            "retention_branch_supported",
            "selected_incontext_singleton_gain_positive",
            "offcontext_fixed_dominant_negative",
            "context_gate_holdout_net_gain_positive",
            "context_gate_improves_over_ungated_holdout",
        )
    )
    deployable_gate_passed = (
        packets["context_gate_calibration"]["summary"].get("decision")
        == CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED
        and source.get("deployable_gate_passes_pre_registered_criteria") is True
    )
    return {
        "local_retention_churn_bracket_supported": bool(
            source.get("retention_branch_supported")
        ),
        "context_gated_singleton_efficacy_supported": all(
            source.get(field) is True
            for field in (
                "selected_incontext_singleton_gain_positive",
                "offcontext_fixed_dominant_negative",
                "context_gate_holdout_net_gain_positive",
            )
        ),
        "deployable_context_gate_passed": deployable_gate_passed,
        "deployable_context_gate_failed": (
            packets["context_gate_calibration"]["summary"].get("decision")
            == CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED
        ),
        "broad_reusable_singleton_claim_excluded": True,
        "causal_retention_claim_supported": retention_ready and deployable_gate_passed,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence"]["metrics"]
    lines = [
        "# Active Top-k-1 Causal-Retention Synthesis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        "- Minimum support-churn advantage top-k-1 vs top-k-2: "
        f"`{metrics.get('min_support_churn_advantage_topk1_vs_topk2')}`",
        "- Minimum commutator advantage top-k-1 vs top-k-2: "
        f"`{metrics.get('min_commutator_anchor_advantage_topk1_vs_topk2')}`",
        "- Selected singleton gain mean: "
        f"`{metrics.get('selected_singleton_gain_mean')}`",
        "- Off-context singleton gain mean: "
        f"`{metrics.get('offcontext_fixed_dominant_singleton_gain_mean')}`",
        "- Deployable gate gain minus ungated: "
        f"`{metrics.get('deployable_gain_minus_ungated')}`",
        "- Deployable off-context harm suppression fraction: "
        f"`{metrics.get('deployable_offcontext_harm_suppression_fraction')}`",
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


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-followup-dir", type=Path, default=DEFAULT_RETENTION_FOLLOWUP_DIR)
    parser.add_argument("--functional-retention-dir", type=Path, default=DEFAULT_FUNCTIONAL_RETENTION_DIR)
    parser.add_argument("--singleton-reconciliation-dir", type=Path, default=DEFAULT_SINGLETON_RECONCILIATION_DIR)
    parser.add_argument("--interference-dir", type=Path, default=DEFAULT_INTERFERENCE_DIR)
    parser.add_argument("--gate-calibration-dir", type=Path, default=DEFAULT_GATE_CALIBRATION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_active_topk1_causal_retention_synthesis_audit(
        retention_followup_dir=args.retention_followup_dir,
        functional_retention_dir=args.functional_retention_dir,
        singleton_reconciliation_dir=args.singleton_reconciliation_dir,
        interference_dir=args.interference_dir,
        gate_calibration_dir=args.gate_calibration_dir,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
