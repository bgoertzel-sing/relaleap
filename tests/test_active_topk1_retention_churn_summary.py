from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_retention_churn_summary import (
    ACTIVE_TOPK1_RETENTION_CHURN_STABLE,
    INSUFFICIENT_EVIDENCE,
    summarize_active_topk1_retention_churn,
)


class ActiveTopk1RetentionChurnSummaryTest(unittest.TestCase):
    def test_summary_establishes_stable_two_seed_retention_churn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            _write_probe(seed1, topk1_churn=0.004, topk2_churn=0.91)
            _write_probe(seed2, topk1_churn=0.008, topk2_churn=0.81)

            summary = summarize_active_topk1_retention_churn(
                probe_dirs=(seed1, seed2),
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ACTIVE_TOPK1_RETENTION_CHURN_STABLE)
            self.assertEqual(summary["packet_count"], 2)
            self.assertEqual(summary["failures"], [])
            aggregates = summary["aggregates"]
            self.assertTrue(aggregates["all_required_signals_pass"])
            self.assertAlmostEqual(aggregates["mean_topk1_support_churn"], 0.006)
            self.assertGreater(aggregates["min_support_churn_advantage"], 0.8)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "probe_metrics.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_summary_fails_closed_on_missing_or_failing_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            _write_probe(seed1, decision="wrong_decision")

            summary = summarize_active_topk1_retention_churn(
                probe_dirs=(seed1, seed2),
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertGreaterEqual(len(summary["failures"]), 2)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("decision", fields)
            self.assertIn("summary_json", fields)


def _write_probe(
    out_dir: Path,
    *,
    status: str = "pass",
    decision: str = "active_topk1_retention_churn_probe_established",
    topk1_churn: float = 0.004,
    topk2_churn: float = 0.9,
) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": status,
        "decision": decision,
        "config_path": "configs/test.yaml",
        "evidence": {
            "metrics": {
                "topk1_anchor_support_churn_after_transfer": topk1_churn,
                "topk2_anchor_support_churn_after_transfer": topk2_churn,
                "topk1_anchor_logit_mse_drift": 0.14,
                "topk2_anchor_logit_mse_drift": 0.16,
                "topk1_transfer_ce_improvement": 0.95,
                "topk2_transfer_ce_improvement": 0.91,
                "dense_transfer_ce_improvement": 0.42,
                "source_topk1_singleton_gain_mean": -0.04,
                "source_context_level_topk1_singleton_gain_mean": -0.15,
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


if __name__ == "__main__":
    unittest.main()
