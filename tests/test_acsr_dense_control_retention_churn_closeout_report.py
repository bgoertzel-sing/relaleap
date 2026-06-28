from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_dense_control_retention_churn_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    NEXT_PATH,
    SUPPORT_DISCOVERY_FROZEN,
    run_acsr_dense_control_retention_churn_closeout_report,
)


class ACSRRetentionChurnCloseoutReportTest(unittest.TestCase):
    def test_selects_dense_control_synthesis_after_tiny_commutator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            support = root / "support.json"
            commutator = root / "commutator.json"
            review = root / "latest-review.md"
            _write_support_closeout(support)
            _write_commutator(commutator)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Complete local support-head nulls and keep RunPod deferred.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_dense_control_retention_churn_closeout_report(
                support_head_closeout_path=support,
                commutator_assay_path=commutator,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SUPPORT_DISCOVERY_FROZEN)
            self.assertEqual(summary["selected_next_step"], NEXT_PATH)
            self.assertEqual(summary["claim_statuses"]["runpod_validation"], "deferred_no_gpu_target")
            self.assertEqual(
                summary["claim_statuses"]["deployable_support_discovery"],
                "frozen_negative_tiny_headroom_sequence_holdout_and_tiny_commutator",
            )
            self.assertFalse(summary["failures"])
            criteria = {row["criterion"]: row for row in summary["closeout_criteria"]}
            self.assertTrue(criteria["sparse_commutator_absolute_signal_tiny"]["passed"])
            self.assertTrue(criteria["dense_control_remains_active_baseline"]["passed"])
            for artifact in (
                "summary.json",
                "source_rows.csv",
                "closeout_criteria.csv",
                "metrics.csv",
                "notes.md",
            ):
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_fails_closed_without_expected_commutator_closeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            support = root / "support.json"
            commutator = root / "commutator.json"
            _write_support_closeout(support)
            _write_commutator(
                commutator,
                claim_status="sparse_order_sensitivity_advantage_candidate_not_promoted",
            )

            summary = run_acsr_dense_control_retention_churn_closeout_report(
                support_head_closeout_path=support,
                commutator_assay_path=commutator,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(summary["selected_next_step"])
            self.assertTrue(
                any(
                    row["criterion"] == "commutator_assay_interpretable"
                    for row in summary["failures"]
                )
            )


def _write_support_closeout(path: Path) -> None:
    payload = {
        "status": "pass",
        "decision": "acsr_support_head_negative_closeout_redirect_selected",
        "selected_next_action": "finite_update_commutator_dense_control_assay",
        "claim_statuses": {
            "deployable_support_discovery": "frozen_negative_tiny_headroom_and_sequence_holdout_failure",
        },
        "metrics": {
            "learned_head_holdout_delta_vs_router": -0.00047850608825683594,
            "upstream_oracle_ce_headroom": -0.0023670196533203125,
            "sequence_head_delta_vs_router": 0.016778945922851562,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_commutator(
    path: Path,
    *,
    claim_status: str = "finite_update_commutator_too_small_for_sparse_mechanism_claim",
) -> None:
    payload = {
        "status": "fail",
        "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
        "claim_status": claim_status,
        "metrics": {
            "sparse_mean_logit_mse": 0.0006394577972709417,
            "dense_mean_logit_mse": 0.008194068482788723,
            "sparse_minus_dense_logit_mse": -0.007554610685517781,
            "sparse_support_churn_fraction": 0.8968253968253969,
            "sparse_mean_ce_abs_delta": 0.03140971584925576,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
