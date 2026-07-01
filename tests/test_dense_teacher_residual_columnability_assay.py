from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_columnability_assay import (
    ARMS,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_residual_columnability_assay,
)


class DenseTeacherResidualColumnabilityAssayTests(unittest.TestCase):
    def test_records_trained_dense_teacher_dictionary_and_null_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_residual_columnability_assay(
                out_dir=root / "assay",
                seed=7,
                train_steps=8,
                router_steps=8,
                column_count=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_teacher_residual_columnability_assay_recorded")
            self.assertTrue(summary["training_rows_present"])
            self.assertTrue(summary["teacher_trained"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["oracle_support_non_deployable"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["uses_future_hidden_or_delta"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["deployable_router_uses_oracle_support"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["uses_task_id"])
            arms = {row["arm"] for row in summary["arm_metrics"]}
            self.assertTrue(set(ARMS).issubset(arms))
            for required in (
                "oracle_support_sparse_dictionary",
                "learned_causal_router_sparse_dictionary",
                "shuffled_teacher_residual_null",
                "delayed_teacher_residual_null",
                "token_position_router_null",
                "rank_matched_residual_control",
                "same_router_flat_value_mlp_control",
            ):
                self.assertIn(required, arms)
            for row in summary["arm_metrics"]:
                self.assertIn("finite_update_commutator_proxy", row)
                self.assertIn("retention_proxy", row)
                self.assertIn("residual_l2_p95", row)
                if row["arm"] != "oracle_support_sparse_dictionary":
                    self.assertFalse(row["uses_oracle_support_at_eval"])
            hard_gates = [row for row in summary["gate_criteria"] if row["required"]]
            self.assertTrue(hard_gates)
            self.assertTrue(all(row["passed"] for row in hard_gates))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "assay" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
