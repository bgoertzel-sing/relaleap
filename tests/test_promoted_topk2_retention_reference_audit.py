from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_retention_reference_audit import (
    INSUFFICIENT_EVIDENCE,
    PROMOTED_TOPK2_ROUTER_DEFAULT_RETENTION_REFERENCE,
    run_promoted_topk2_retention_reference_audit,
)


class PromotedTopk2RetentionReferenceAuditTest(unittest.TestCase):
    def test_establishes_router_default_reference_not_low_churn_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            gate = root / "gate"
            _write_probe(seed1, topk2_churn=0.9, topk1_churn=0.004)
            _write_probe(seed2, topk2_churn=0.82, topk1_churn=0.008)
            _write_failed_gate(gate)

            summary = run_promoted_topk2_retention_reference_audit(
                probe_dirs=(seed1, seed2),
                gate_audit_dir=gate,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                PROMOTED_TOPK2_ROUTER_DEFAULT_RETENTION_REFERENCE,
            )
            self.assertTrue(summary["signals"]["topk1_context_gate_failed"])
            self.assertTrue(summary["signals"]["topk2_transfer_beats_dense_control"])
            self.assertTrue(summary["signals"]["topk2_support_churn_high"])
            self.assertTrue(
                summary["signals"]["topk2_support_churn_higher_than_topk1"]
            )
            self.assertFalse(
                summary["signals"]["topk2_low_churn_retention_claim_supported"]
            )
            self.assertGreater(
                summary["aggregates"]["mean_topk2_support_churn_minus_topk1"],
                0.8,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "probe_metrics.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_packet_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            gate = root / "gate"
            _write_probe(seed1)
            _write_failed_gate(gate)

            summary = run_promoted_topk2_retention_reference_audit(
                probe_dirs=(seed1, seed2),
                gate_audit_dir=gate,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("summary_json", fields)


def _write_probe(
    out_dir: Path,
    *,
    topk2_churn: float = 0.9,
    topk1_churn: float = 0.004,
) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "active_topk1_retention_churn_probe_established",
        "config_path": "configs/test.yaml",
        "evidence": {
            "metrics": {
                "topk2_anchor_support_churn_after_transfer": topk2_churn,
                "topk2_anchor_logit_mse_drift": 0.16,
                "topk2_anchor_residual_stream_l2_drift": 4.6,
                "topk2_anchor_ce_drift": -0.91,
                "topk2_transfer_ce_improvement": 0.91,
                "topk2_commutator_anchor_logit_mse": 0.2,
                "topk2_commutator_transfer_logit_mse": 0.21,
                "topk2_commutator_anchor_residual_stream_l2": 4.1,
                "topk2_commutator_transfer_residual_stream_l2": 4.2,
                "topk1_anchor_support_churn_after_transfer": topk1_churn,
                "topk1_anchor_logit_mse_drift": 0.14,
                "topk1_transfer_ce_improvement": 0.95,
                "dense_transfer_ce_improvement": 0.42,
            },
            "signals": {
                "required_variants_present": True,
                "finite_update_commutator_present": True,
            },
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_failed_gate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "deployable_context_gate_suppression_calibration_failed",
        "next_step": "keep top-k-1 singletons diagnostic-only",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
