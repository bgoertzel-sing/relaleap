from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.sparse_target_adaptation_rescue_probe import (
    REQUIRED_ARTIFACTS,
    run_sparse_target_adaptation_rescue_probe,
)


class SparseTargetAdaptationRescueProbeTest(unittest.TestCase):
    def test_probe_writes_required_artifacts_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "probe"
            summary = run_sparse_target_adaptation_rescue_probe(
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
                "sparse_target_adaptation_rescue_probe_recorded",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["hidden_rule_boundaries"])
            self.assertFalse(summary["task_id_visible_to_model"])
            self.assertTrue(summary["shared_vocab_and_head"])
            arms = {row["arm"] for row in summary["rescue_metrics"]}
            self.assertIn("dense_active_rank", arms)
            self.assertIn("contextual_topk2_baseline", arms)
            self.assertIn("contextual_topk2_value_lr2_anchor_kl", arms)
            self.assertIn("contextual_topk2_value_lr4_anchor_kl", arms)
            self.assertIn("random_frequency_matched_topk2", arms)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

    def test_probe_records_fail_closed_claim_gates_without_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_sparse_target_adaptation_rescue_probe(
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
                    row["criterion"]
                    == "best_rescue_target_adaptation_dense_matched"
                    for row in summary["gate_criteria"]
                )
            )
            self.assertIn(
                summary["claim_status"],
                {
                    "sparse_target_adaptation_rescue_not_established",
                    "sparse_target_adaptation_rescue_candidate_not_promoted",
                },
            )
            self.assertIn(
                "best_rescue_minus_dense_target_ce_delta",
                summary["primary_result"],
            )


if __name__ == "__main__":
    unittest.main()
