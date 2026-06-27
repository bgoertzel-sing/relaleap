from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_decision import (
    ACSR_POST_RUNPOD_CANDIDATE_RECORDED,
    INSUFFICIENT_EVIDENCE,
    NEXT_STRONGER_NON_CE_CONTROL,
    run_anticipatory_contextual_support_routing_decision,
)


class AnticipatoryContextualSupportRoutingDecisionTest(unittest.TestCase):
    def test_records_replicated_candidate_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local.json"
            runpod = root / "runpod.json"
            review = root / "latest-review.md"
            _write_synthesis(local, backend="local")
            _write_synthesis(runpod, backend="runpod")
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

            summary = run_anticipatory_contextual_support_routing_decision(
                local_synthesis_path=local,
                runpod_synthesis_path=runpod,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ACSR_POST_RUNPOD_CANDIDATE_RECORDED)
            self.assertEqual(summary["claim_status"], "acsr_replicated_candidate_not_promoted")
            self.assertEqual(summary["selected_next_step"], NEXT_STRONGER_NON_CE_CONTROL)
            self.assertEqual(
                summary["claim_statuses"]["promoted_default_router"],
                "blocked_pending_stronger_non_ce_controls",
            )
            self.assertTrue(summary["gate_status"]["passes_post_runpod_candidate_gate"])
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(
                summary["deferred_or_rejected_recommendations"][0]["status"],
                "accepted_already_satisfied",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "synthesis_metrics.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_runpod_synthesis_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local.json"
            _write_synthesis(local, backend="local")

            summary = run_anticipatory_contextual_support_routing_decision(
                local_synthesis_path=local,
                runpod_synthesis_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failures = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("runpod_two_seed_synthesis_local_check", "source_artifact"),
                failures,
            )


def _write_synthesis(path: Path, *, backend: str) -> None:
    packet = {
        "status": "pass",
        "decision": "acsr_two_seed_local_synthesis_recorded",
        "claim_status": "local_acsr_controls_consistently_discriminative",
        "gpu_backend": backend,
        "packet_count": 2,
        "aggregates": {
            "all_required_gates_pass": True,
            "all_same_student_beats_shuffled": True,
            "all_same_student_beats_token_position": True,
            "all_teacher_churn_below_shuffled": True,
            "all_teacher_churn_below_token_position": True,
            "mean_acsr_minus_causal_ce_loss": -0.16,
            "mean_acsr_minus_causal_regret": -0.16,
            "mean_acsr_minus_shuffled_ce_loss": -0.68,
            "mean_acsr_minus_teacher_ce_loss": 0.004,
            "mean_acsr_minus_token_position_ce_loss": -0.17,
            "mean_acsr_teacher_support_churn": 0.03,
            "mean_mlp_predictor_r2": 0.93,
            "mean_shuffled_teacher_support_churn": 0.63,
            "mean_token_position_predictor_r2": -0.17,
            "mean_token_position_teacher_support_churn": 0.32,
        },
    }
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
