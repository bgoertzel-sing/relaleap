"""Active top-k-1 retention/churn probe from the separability packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_causal_separability_audit import (
    ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED,
    DEFAULT_OUT_DIR as DEFAULT_SEPARABILITY_DIR,
)
from relaleap.experiments.retention_churn_microtest import DEFAULT_CONFIG
from relaleap.experiments.retention_churn_microtest import run_retention_churn_microtest


DEFAULT_OUT_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"
)
ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED = (
    "active_topk1_retention_churn_probe_established"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_retention_churn_probe(
    *,
    config_path: Path = DEFAULT_CONFIG,
    separability_dir: Path = DEFAULT_SEPARABILITY_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Run and interpret a bounded retention/churn probe for active top-k-1."""

    failures = _separability_failures(separability_dir)
    microtest = run_retention_churn_microtest(config_path, out_dir)
    evidence = _build_evidence(separability_dir, microtest, failures)
    signals = evidence["signals"]
    if (
        not failures
        and microtest.get("status") == "ok"
        and signals["required_variants_present"]
        and signals["topk1_support_churn_lower_than_topk2"]
        and signals["topk1_logit_churn_not_higher_than_topk2"]
        and signals["topk1_transfer_improvement_at_least_topk2"]
    ):
        status = "pass"
        decision = ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED
        rationale = (
            "The active rank-matched contextual top-k-1 bracket is tied to the "
            "separability packet and shows much lower support churn than the "
            "promoted top-k-2 reference after transfer training, with no worse "
            "logit churn and at least comparable transfer improvement. This "
            "supports using top-k-1 as the local retention/churn bracket, but it "
            "does not establish singleton causal separability because the source "
            "separability packet still had negative average singleton gain."
        )
        next_step = (
            "run a local seed-2 active top-k-1 retention/churn repeat before "
            "treating the low-churn signal as stable"
        )
    else:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The active top-k-1 retention/churn probe could not be established "
            "from the current separability and microtest artifacts."
        )
        next_step = (
            "repair the active top-k-1 separability packet or retention/churn "
            "microtest artifacts before interpreting retention evidence"
        )

    summary = {
        **microtest,
        "status": status,
        "decision": decision,
        "separability_dir": str(separability_dir),
        "evidence": evidence,
        "rationale": rationale,
        "next_step": next_step,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _separability_failures(separability_dir: Path) -> list[dict[str, Any]]:
    path = separability_dir / "summary.json"
    if not path.is_file():
        return [
            {
                "field": "separability_summary_json",
                "expected": "file exists",
                "actual": "missing",
                "path": str(path),
            }
        ]
    summary = _read_json_object(path)
    failures = []
    if summary.get("status") != "pass":
        failures.append(
            {
                "field": "separability.status",
                "expected": "pass",
                "actual": summary.get("status"),
                "path": str(path),
            }
        )
    if summary.get("decision") != ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED:
        failures.append(
            {
                "field": "separability.decision",
                "expected": ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED,
                "actual": summary.get("decision"),
                "path": str(path),
            }
        )
    return failures


def _build_evidence(
    separability_dir: Path,
    microtest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    separability = _read_json_object(separability_dir / "summary.json")
    variants = {
        str(row.get("variant")): row
        for row in microtest.get("audit", {}).get("variants", [])
        if isinstance(row, dict)
    }
    topk1 = variants.get("rank_matched_contextual_topk1", {})
    topk2 = variants.get("promoted_contextual_topk2", {})
    dense = variants.get("norm_matched_dense_active_rank", {})
    metrics = {
        "source_separability_decision": separability.get("decision"),
        "source_topk1_singleton_gain_mean": _nested_get(
            separability,
            "evidence",
            "metrics",
            "topk1_singleton_gain_mean",
        ),
        "source_context_level_topk1_singleton_gain_mean": _nested_get(
            separability,
            "evidence",
            "metrics",
            "context_level_topk1_singleton_gain_mean",
        ),
        "topk1_anchor_support_churn_after_transfer": _float_or_none(
            topk1.get("anchor_support_churn_after_transfer")
        ),
        "topk2_anchor_support_churn_after_transfer": _float_or_none(
            topk2.get("anchor_support_churn_after_transfer")
        ),
        "topk1_anchor_logit_mse_drift": _float_or_none(
            topk1.get("anchor_logit_mse_drift")
        ),
        "topk2_anchor_logit_mse_drift": _float_or_none(
            topk2.get("anchor_logit_mse_drift")
        ),
        "topk1_anchor_residual_stream_l2_drift": _float_or_none(
            topk1.get("anchor_residual_stream_l2_drift")
        ),
        "topk2_anchor_residual_stream_l2_drift": _float_or_none(
            topk2.get("anchor_residual_stream_l2_drift")
        ),
        "topk1_anchor_ce_drift": _float_or_none(topk1.get("anchor_ce_drift")),
        "topk2_anchor_ce_drift": _float_or_none(topk2.get("anchor_ce_drift")),
        "dense_anchor_ce_drift": _float_or_none(dense.get("anchor_ce_drift")),
        "topk1_transfer_ce_improvement": _float_or_none(
            topk1.get("transfer_ce_improvement")
        ),
        "topk2_transfer_ce_improvement": _float_or_none(
            topk2.get("transfer_ce_improvement")
        ),
        "dense_transfer_ce_improvement": _float_or_none(
            dense.get("transfer_ce_improvement")
        ),
    }
    topk1_churn = metrics["topk1_anchor_support_churn_after_transfer"]
    topk2_churn = metrics["topk2_anchor_support_churn_after_transfer"]
    topk1_logit = metrics["topk1_anchor_logit_mse_drift"]
    topk2_logit = metrics["topk2_anchor_logit_mse_drift"]
    topk1_transfer = metrics["topk1_transfer_ce_improvement"]
    topk2_transfer = metrics["topk2_transfer_ce_improvement"]
    dense_transfer = metrics["dense_transfer_ce_improvement"]
    return {
        "separability_dir": str(separability_dir),
        "microtest_out_dir": microtest.get("out_dir"),
        "metrics": metrics,
        "signals": {
            "required_variants_present": all(
                key in variants
                for key in (
                    "rank_matched_contextual_topk1",
                    "promoted_contextual_topk2",
                    "norm_matched_dense_active_rank",
                )
            ),
            "topk1_support_churn_lower_than_topk2": (
                topk1_churn is not None
                and topk2_churn is not None
                and topk1_churn < topk2_churn
            ),
            "topk1_logit_churn_not_higher_than_topk2": (
                topk1_logit is not None
                and topk2_logit is not None
                and topk1_logit <= topk2_logit
            ),
            "topk1_transfer_improvement_at_least_topk2": (
                topk1_transfer is not None
                and topk2_transfer is not None
                and topk1_transfer >= topk2_transfer
            ),
            "topk1_transfer_improvement_beats_dense": (
                topk1_transfer is not None
                and dense_transfer is not None
                and topk1_transfer > dense_transfer
            ),
            "source_singleton_gain_still_negative": (
                _float_or_none(metrics["source_topk1_singleton_gain_mean"])
                is not None
                and float(metrics["source_topk1_singleton_gain_mean"]) < 0.0
            ),
        },
        "failures": failures,
    }


def _nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["evidence"]["metrics"]
    lines = [
        "# Active Top-k-1 Retention/Churn Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Source separability packet: `{summary['separability_dir']}`",
        "- Top-k-1 support churn after transfer: "
        f"`{metrics['topk1_anchor_support_churn_after_transfer']}`",
        "- Top-k-2 reference support churn after transfer: "
        f"`{metrics['topk2_anchor_support_churn_after_transfer']}`",
        f"- Top-k-1 transfer CE improvement: `{metrics['topk1_transfer_ce_improvement']}`",
        f"- Top-k-2 transfer CE improvement: `{metrics['topk2_transfer_ce_improvement']}`",
        "- Source top-k-1 singleton gain mean: "
        f"`{metrics['source_topk1_singleton_gain_mean']}`",
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--separability-dir", type=Path, default=DEFAULT_SEPARABILITY_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_retention_churn_probe(
        config_path=args.config,
        separability_dir=args.separability_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "evidence": summary["evidence"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
