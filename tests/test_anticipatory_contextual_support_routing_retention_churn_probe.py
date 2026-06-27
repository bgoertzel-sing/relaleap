from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_retention_churn_probe import (
    ACSR_CROSS_CONTEXT_RETENTION_CHURN_SUPPORTED,
    INSUFFICIENT_EVIDENCE,
    run_acsr_same_student_cross_context_retention_churn_probe,
)


class AnticipatoryContextualSupportRoutingRetentionChurnProbeTest(unittest.TestCase):
    def test_records_cross_context_same_student_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            packets = [root / f"packet{i}" for i in range(4)]
            for packet in packets:
                _write_packet(packet)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement and run the local CPU ACSR smoke pilot with explicit leakage, shuffled-feature, token/position-only, same-student, and retention/churn gates before any GPU/backend replication.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_same_student_cross_context_retention_churn_probe(
                audit_dirs=tuple(packets),
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                ACSR_CROSS_CONTEXT_RETENTION_CHURN_SUPPORTED,
            )
            self.assertTrue(
                summary["gate_status"]["passes_cross_context_retention_churn_gate"]
            )
            self.assertEqual(len(summary["comparison_rows"]), 8)
            self.assertTrue(summary["aggregate_rows"][0]["all_gates_pass"])
            self.assertEqual(
                summary["deferred_or_rejected_recommendations"][0]["status"],
                "accepted_already_satisfied",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_packets.csv").is_file())
            self.assertTrue(
                (root / "report" / "same_student_cross_context_metrics.csv").is_file()
            )
            self.assertTrue((root / "report" / "aggregate_metrics.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_token_position_churn_matches_acsr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            packet = root / "packet"
            _write_packet(packet, token_position_churn=0.20)

            summary = run_acsr_same_student_cross_context_retention_churn_probe(
                audit_dirs=(packet,),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failures = {
                failure.get("field") for failure in summary["failures"]
            }
            self.assertIn("cross_context_anchor_support_churn_not_worse", failures)


def _write_packet(path: Path, *, token_position_churn: float = 0.24) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "decision": "anticipatory_contextual_support_routing_smoke_completed",
        "claim_status": "local_acsr_smoke_evidence_recorded",
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "same_student_metrics.csv").write_text(
        "\n".join(
            [
                "comparison,acsr_forced_ce_loss,control_forced_ce_loss,acsr_minus_control_ce_loss",
                "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features,2.90,3.05,-0.15",
                "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features,2.90,3.57,-0.67",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "retention_churn_metrics.csv").write_text(
        "\n".join(
            [
                "phase,variant,transfer_steps,anchor_support_churn_after_transfer,anchor_logit_mse_after_transfer,transfer_ce_improvement,teacher_support_churn",
                "second_context_transfer,acsr_mlp_predicted_future,10,0.22,0.01,0.21,",
                f"second_context_transfer,token_position_only_predicted_features,10,{token_position_churn},0.03,0.20,",
                "second_context_transfer,shuffled_predicted_features,10,0.70,0.19,0.08,",
                "fixed_context_teacher_reference,acsr_mlp_predicted_future,,,,,0.03",
                "fixed_context_teacher_reference,token_position_only_predicted_features,,,,,0.30",
                "fixed_context_teacher_reference,shuffled_predicted_features,,,,,0.60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
