from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_support_head_closeout_redirect_report import (
    FINITE_UPDATE_DENSE_CONTROL_ACTION,
    INSUFFICIENT_EVIDENCE,
    SUPPORT_HEAD_NEGATIVE,
    run_acsr_support_head_closeout_redirect_report,
)


class ACSRSupportHeadCloseoutRedirectReportTest(unittest.TestCase):
    def test_selects_finite_update_dense_control_after_negative_support_head_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "gate.json"
            review = root / "latest-review.md"
            _write_gate(gate, decision="deployable_support_head_gate_blocks_claim_pending_nulls_or_headroom")
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Add support-head nulls and keep RunPod deferred.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_support_head_closeout_redirect_report(
                deployable_gate_path=gate,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SUPPORT_HEAD_NEGATIVE)
            self.assertEqual(summary["selected_next_action"], FINITE_UPDATE_DENSE_CONTROL_ACTION)
            self.assertEqual(summary["claim_statuses"]["runpod_validation"], "deferred_no_gpu_target")
            self.assertEqual(summary["claim_statuses"]["ben_notification"], "not_required")
            self.assertIn("finite-update commutator", summary["next_step"])
            self.assertFalse(summary["failures"])
            for artifact in ("summary.json", "source_rows.csv", "redirect_criteria.csv", "notes.md"):
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_fails_closed_when_gate_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "gate.json"
            _write_gate(gate, decision="deployable_support_head_gate_positive_ready_for_local_repeat")

            summary = run_acsr_support_head_closeout_redirect_report(
                deployable_gate_path=gate,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(summary["selected_next_action"])
            self.assertTrue(
                any(row["criterion"] == "deployable_gate_is_negative" for row in summary["failures"])
            )


def _write_gate(path: Path, *, decision: str) -> None:
    value = {
        "status": "pass",
        "decision": decision,
        "claim_status": "deployable_support_discovery_not_established_sparse_identity_retired",
        "aggregate_metrics": {
            "learned_head_holdout_intervention_minus_router_loss": -0.00047850608825683594,
            "learned_head_holdout_oracle_gap_recovery_fraction": 0.1893932244974993,
            "same_student_holdout_oracle_gap_recovery_fraction": 1.0,
            "sparse_oracle_minus_sparse_default_heldout_ce_delta": -0.0023670196533203125,
        },
        "support_head_metrics": [
            {
                "component": "learned_sequence_support_head",
                "holdout_intervention_minus_router_loss": 0.016778945922851562,
                "holdout_oracle_gap_recovery_fraction": -6.654311649016641,
            }
        ],
        "null_controls": [
            {
                "control": "shuffled_causal_feature_support_head_null",
                "present": True,
            },
            {
                "control": "token_position_support_null",
                "present": True,
            },
        ],
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
