from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_causal_retention_synthesis_audit import (
    CAUSAL_RETENTION_CLAIM_BLOCKED_DEPLOYABLE_GATE,
    CAUSAL_RETENTION_CLAIM_SUPPORTED,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_causal_retention_synthesis_audit,
)


class ActiveTopk1CausalRetentionSynthesisAuditTest(unittest.TestCase):
    def test_blocks_causal_claim_when_deployable_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root, deployable_gate_passes=False)

            summary = run_active_topk1_causal_retention_synthesis_audit(
                retention_followup_dir=paths["retention_followup"],
                functional_retention_dir=paths["functional_retention"],
                singleton_reconciliation_dir=paths["singleton_reconciliation"],
                interference_dir=paths["interference"],
                gate_calibration_dir=paths["gate_calibration"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                CAUSAL_RETENTION_CLAIM_BLOCKED_DEPLOYABLE_GATE,
            )
            self.assertTrue(summary["signals"]["local_retention_churn_bracket_supported"])
            self.assertTrue(summary["signals"]["context_gated_singleton_efficacy_supported"])
            self.assertTrue(summary["signals"]["deployable_context_gate_failed"])
            self.assertFalse(summary["signals"]["causal_retention_claim_supported"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "evidence_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_supports_claim_when_deployable_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root, deployable_gate_passes=True)

            summary = run_active_topk1_causal_retention_synthesis_audit(
                retention_followup_dir=paths["retention_followup"],
                functional_retention_dir=paths["functional_retention"],
                singleton_reconciliation_dir=paths["singleton_reconciliation"],
                interference_dir=paths["interference"],
                gate_calibration_dir=paths["gate_calibration"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], CAUSAL_RETENTION_CLAIM_SUPPORTED)
            self.assertTrue(summary["signals"]["deployable_context_gate_passed"])
            self.assertTrue(summary["signals"]["causal_retention_claim_supported"])

    def test_fails_closed_when_source_packet_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root, deployable_gate_passes=False)
            (paths["interference"] / "summary.json").unlink()

            summary = run_active_topk1_causal_retention_synthesis_audit(
                retention_followup_dir=paths["retention_followup"],
                functional_retention_dir=paths["functional_retention"],
                singleton_reconciliation_dir=paths["singleton_reconciliation"],
                interference_dir=paths["interference"],
                gate_calibration_dir=paths["gate_calibration"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                ("context_conditioned_interference", "summary_json"),
                {
                    (failure.get("source"), failure.get("field"))
                    for failure in summary["failures"]
                },
            )


def _write_sources(root: Path, *, deployable_gate_passes: bool) -> dict[str, Path]:
    paths = {
        "retention_followup": root / "retention_followup",
        "functional_retention": root / "functional_retention",
        "singleton_reconciliation": root / "singleton_reconciliation",
        "interference": root / "interference",
        "gate_calibration": root / "gate_calibration",
    }
    for path in paths.values():
        path.mkdir()

    _write_json(
        paths["retention_followup"] / "summary.json",
        {
            "status": "pass",
            "decision": "retention_functional_churn_bracket_supported",
            "signals": {"branch_supported": True},
            "aggregates": {
                "min_support_churn_advantage_topk1_vs_topk2": 0.8,
                "min_commutator_anchor_advantage_topk1_vs_topk2": 0.17,
                "min_transfer_advantage_topk1_vs_dense": 0.39,
                "mean_topk1_anchor_support_churn_after_transfer": 0.01,
                "mean_topk2_anchor_support_churn_after_transfer": 0.86,
            },
        },
    )
    _write_json(
        paths["functional_retention"] / "summary.json",
        {
            "status": "pass",
            "decision": "functional_retention_bracket_only",
            "evidence": {
                "aggregates": {
                    "mean_commutator_anchor_logit_mse_advantage_topk1_vs_dense": 0.05
                },
                "claim_signals": {
                    "claim_supported": False,
                    "offcontext_singleton_interference_present": True,
                },
            },
        },
    )
    _write_json(
        paths["singleton_reconciliation"] / "summary.json",
        {
            "status": "pass",
            "decision": "context_gated_singleton_efficacy_with_offcontext_interference",
            "evidence": {
                "metrics": {
                    "selected_singleton_gain_mean": 1.0,
                    "offcontext_fixed_dominant_singleton_gain_mean": -0.14,
                },
                "signals": {
                    "selected_incontext_positive": True,
                    "offcontext_fixed_dominant_negative": True,
                },
            },
        },
    )
    _write_json(
        paths["interference"] / "summary.json",
        {
            "status": "pass",
            "decision": "context_gate_reduces_offcontext_interference",
            "evidence": {
                "metrics": {
                    "context_gated_net_gain_holdout_mean": 0.77,
                    "context_gate_gain_minus_ungated_holdout_mean": 0.44,
                },
                "signals": {
                    "context_gate_holdout_net_gain_positive": True,
                    "context_gate_improves_over_ungated_holdout": True,
                },
            },
        },
    )
    _write_json(
        paths["gate_calibration"] / "summary.json",
        {
            "status": "pass",
            "decision": (
                "deployable_context_gate_suppression_calibration_passed"
                if deployable_gate_passes
                else "deployable_context_gate_suppression_calibration_failed"
            ),
            "evidence": {
                "metrics": {
                    "deployable_holdout_net_gain": 0.43,
                    "deployable_gain_minus_ungated": 0.08 if deployable_gate_passes else -0.04,
                    "deployable_gain_minus_coverage_matched_random": (
                        0.06 if deployable_gate_passes else 0.02
                    ),
                    "deployable_offcontext_harm_suppression_fraction": (
                        0.6 if deployable_gate_passes else 0.12
                    ),
                },
                "signals": {
                    "deployable_gate_passes_pre_registered_criteria": deployable_gate_passes,
                    "deployable_improves_over_ungated": deployable_gate_passes,
                    "deployable_suppresses_offcontext_harm": deployable_gate_passes,
                },
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
