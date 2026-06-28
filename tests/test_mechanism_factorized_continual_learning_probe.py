from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mechanism_factorized_continual_learning_probe import (
    REQUIRED_ARTIFACTS,
    RULE_SEQUENCE,
    run_mechanism_factorized_continual_learning_probe,
)


class MechanismFactorizedContinualLearningProbeTest(unittest.TestCase):
    def test_probe_writes_required_artifacts_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "probe"
            summary = run_mechanism_factorized_continual_learning_probe(
                out_dir=out_dir,
                seed=3,
                steps_per_phase=1,
                batch_size=4,
                hidden_dim=16,
                num_columns=4,
                atoms_per_column=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "mechanism_factorized_continual_learning_probe_recorded",
            )
            self.assertEqual(summary["rules"], list(RULE_SEQUENCE))
            self.assertTrue(summary["hidden_rule_boundaries"])
            self.assertFalse(summary["task_id_visible_to_model"])
            self.assertTrue(summary["shared_vocab_and_head"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertIn("dense_active_rank", summary["controls"])
            self.assertIn("contextual_topk1", summary["controls"])
            self.assertIn("random_frequency_matched_topk2", summary["controls"])
            self.assertIn("contextual_topk1_anchor_kl", summary["controls"])
            topk2 = next(row for row in summary["arm_metrics"] if row["arm"] == "contextual_topk2")
            self.assertIn("target_adaptation_improvement", topk2)
            self.assertIn("forgetting_per_target_improvement", topk2)
            self.assertIn(
                "topk2_minus_dense_forgetting_per_target_improvement",
                summary["primary_result"],
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

    def test_probe_records_fail_closed_gate_semantics_without_failing_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_mechanism_factorized_continual_learning_probe(
                out_dir=Path(tmpdir) / "probe",
                seed=5,
                steps_per_phase=1,
                batch_size=4,
                hidden_dim=16,
                num_columns=4,
                atoms_per_column=2,
            )

            hard_gates = [
                row for row in summary["gate_criteria"] if row["severity"] == "hard"
            ]
            self.assertTrue(hard_gates)
            self.assertTrue(all(row["passed"] for row in hard_gates))
            self.assertTrue(
                any(
                    row["criterion"] == "topk1_off_target_kl_no_worse_than_dense"
                    for row in summary["gate_criteria"]
                )
            )
            self.assertTrue(
                any(
                    row["criterion"] == "topk2_interference_per_gain_no_worse_than_dense"
                    for row in summary["gate_criteria"]
                )
            )
            self.assertIn(
                summary["claim_status"],
                {
                    "mechanism_factorized_sparse_retention_not_established",
                    "mechanism_factorized_sparse_retention_candidate_supported_not_promoted",
                },
            )
            if all(
                any(row["criterion"] == criterion and row["passed"] for row in summary["gate_criteria"])
                for criterion in (
                    "topk2_interference_per_gain_no_worse_than_dense",
                    "topk2_beats_random_support_tradeoff_null",
                )
            ):
                self.assertEqual(
                    summary["selected_next_step"],
                    "run_second_seed_mechanism_factorized_cl_repeat_before_any_gpu_validation",
                )
            self.assertIn("topk1_minus_dense_mean_off_target_kl", summary["primary_result"])


if __name__ == "__main__":
    unittest.main()
