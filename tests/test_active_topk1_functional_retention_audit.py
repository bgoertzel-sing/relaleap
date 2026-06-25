from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_functional_retention_audit import (
    BLOCKED_BY_NEGATIVE_SINGLETON_GAIN,
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_functional_retention_audit,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


class ActiveTopk1FunctionalRetentionAuditTest(unittest.TestCase):
    def test_audit_records_bracket_only_when_singleton_gain_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            _write_probe(seed1, topk1_churn=0.004, topk2_churn=0.91)
            _write_probe(seed2, topk1_churn=0.008, topk2_churn=0.81)

            summary = run_active_topk1_functional_retention_audit(
                probe_dirs=(seed1, seed2),
                singleton_reconciliation_dir=root / "missing_reconciliation",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], FUNCTIONAL_RETENTION_BRACKET_ONLY)
            self.assertEqual(summary["claim_status"], BLOCKED_BY_NEGATIVE_SINGLETON_GAIN)
            signals = summary["evidence"]["claim_signals"]
            self.assertTrue(signals["support_identity_churn_cleaner_than_topk2"])
            self.assertTrue(signals["functional_logit_churn_not_higher_than_topk2"])
            self.assertTrue(signals["transfer_improvement_beats_dense_control"])
            self.assertFalse(signals["singleton_gain_positive"])
            self.assertTrue(signals["finite_update_commutator_present"])
            self.assertTrue(signals["finite_update_commutator_not_worse_than_topk2"])
            self.assertFalse(signals["claim_supported"])
            aggregates = summary["evidence"]["aggregates"]
            self.assertGreater(
                aggregates["min_support_churn_advantage_topk1_vs_topk2"],
                0.8,
            )
            self.assertGreater(
                aggregates[
                    "min_commutator_anchor_logit_mse_advantage_topk1_vs_topk2"
                ],
                0.0,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "packet_metrics.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_audit_prefers_reconciled_singleton_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            reconciliation = root / "reconciliation"
            _write_probe(seed1, topk1_churn=0.004, topk2_churn=0.91)
            _write_probe(seed2, topk1_churn=0.008, topk2_churn=0.81)
            _write_reconciliation(reconciliation)

            summary = run_active_topk1_functional_retention_audit(
                probe_dirs=(seed1, seed2),
                singleton_reconciliation_dir=reconciliation,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], FUNCTIONAL_RETENTION_BRACKET_ONLY)
            self.assertEqual(
                summary["claim_status"],
                CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
            )
            signals = summary["evidence"]["claim_signals"]
            self.assertTrue(signals["singleton_reconciliation_present"])
            self.assertTrue(signals["singleton_gain_positive"])
            self.assertFalse(signals["packet_singleton_gain_positive"])
            self.assertTrue(signals["selected_incontext_singleton_gain_positive"])
            self.assertTrue(signals["offcontext_singleton_interference_present"])
            self.assertFalse(signals["claim_supported"])
            singleton = summary["evidence"]["singleton_reconciliation"]
            self.assertEqual(singleton["status"], "pass")
            self.assertEqual(
                singleton["decision"],
                CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
            )
            self.assertAlmostEqual(
                singleton["metrics"]["selected_singleton_gain_mean"], 1.0
            )

    def test_audit_fails_closed_when_packet_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            _write_probe(seed1)

            summary = run_active_topk1_functional_retention_audit(
                probe_dirs=(seed1, seed2),
                singleton_reconciliation_dir=root / "missing_reconciliation",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("summary_json", fields)


def _write_probe(
    out_dir: Path,
    *,
    topk1_churn: float = 0.004,
    topk2_churn: float = 0.9,
) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "active_topk1_retention_churn_probe_established",
        "evidence": {
            "metrics": {
                "source_topk1_singleton_gain_mean": -0.04,
                "source_context_level_topk1_singleton_gain_mean": -0.15,
                "topk1_anchor_support_churn_after_transfer": topk1_churn,
                "topk2_anchor_support_churn_after_transfer": topk2_churn,
                "topk1_anchor_logit_mse_drift": 0.14,
                "topk2_anchor_logit_mse_drift": 0.16,
                "topk1_anchor_residual_stream_l2_drift": 1.2,
                "topk2_anchor_residual_stream_l2_drift": 1.4,
                "topk1_anchor_ce_drift": 0.01,
                "topk2_anchor_ce_drift": 0.02,
                "dense_anchor_ce_drift": 0.03,
                "topk1_transfer_ce_improvement": 0.95,
                "topk2_transfer_ce_improvement": 0.91,
                "dense_transfer_ce_improvement": 0.42,
                "topk1_commutator_anchor_logit_mse": 0.04,
                "topk2_commutator_anchor_logit_mse": 0.08,
                "dense_commutator_anchor_logit_mse": 0.06,
                "topk1_commutator_transfer_logit_mse": 0.03,
                "topk2_commutator_transfer_logit_mse": 0.07,
                "dense_commutator_transfer_logit_mse": 0.05,
            },
            "signals": {
                "required_variants_present": True,
                "topk1_support_churn_lower_than_topk2": True,
                "topk1_logit_churn_not_higher_than_topk2": True,
                "topk1_transfer_improvement_at_least_topk2": True,
                "source_singleton_gain_still_negative": True,
            },
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_reconciliation(out_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        "evidence": {
            "metrics": {
                "selected_singleton_gain_mean": 1.0,
                "logged_oracle_singleton_gain_mean": 1.2,
                "offcontext_fixed_dominant_singleton_gain_mean": -0.14,
                "random_singleton_gain_mean": 0.0,
                "exhaustive_singleton_gain_mean": 1.44,
            },
            "signals": {
                "selected_incontext_positive": True,
                "logged_oracle_incontext_positive": True,
                "offcontext_fixed_dominant_negative": True,
                "random_singleton_control_present": True,
                "exhaustive_singleton_control_present": True,
            },
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
