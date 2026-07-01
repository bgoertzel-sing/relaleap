from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_value_capacity_norm_assay import (
    ARMS,
    DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_residual_value_capacity_norm_assay,
)


class DenseTeacherResidualValueCapacityNormAssayTests(unittest.TestCase):
    def test_trains_value_capacity_norm_control_assay_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate"
            pregate.mkdir()
            (pregate / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "dense_teacher_residual_value_capacity_norm_pregate_recorded",
                        "claim_status": "value_capacity_norm_pregate_ready_for_local_training_no_gpu",
                        "selected_next_step": "implement bounded local value-capacity/norm-control trained assay for dense-teacher residual dictionaries",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_residual_value_capacity_norm_assay(
                pregate_dir=pregate,
                out_dir=root / "assay",
                seed=5,
                teacher_steps=6,
                router_steps=6,
                value_steps=6,
                control_steps=6,
                column_count=4,
                values_per_column=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertTrue(summary["training_rows_present"])
            self.assertTrue(summary["teacher_trained"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["oracle_support_non_deployable"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["uses_future_hidden_or_delta"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["deployable_router_uses_oracle_support"])

            arms = {row["arm"] for row in summary["arm_metrics"]}
            self.assertTrue(set(ARMS).issubset(arms))
            required_arms = {
                "oracle_support_norm_matched_multi_value_dictionary",
                "oracle_support_low_rank_value_dictionary",
                "learned_router_norm_matched_multi_value_dictionary",
                "same_router_flat_value_norm_matched_control",
                "random_support_norm_matched_null",
                "frequency_support_norm_matched_null",
                "token_position_norm_matched_null",
                "shuffled_teacher_residual_norm_matched_null",
                "delayed_teacher_residual_norm_matched_null",
            }
            self.assertTrue(required_arms.issubset(arms))
            for row in summary["arm_metrics"]:
                self.assertIn("residual_l2_mean_ratio_vs_teacher", row)
                self.assertIn("finite_update_commutator_proxy", row)
                self.assertIn("retention_proxy", row)
                self.assertIn("active_params", row)
                self.assertIn("stored_params", row)
                if row["arm"].startswith("oracle_support"):
                    self.assertTrue(row["uses_oracle_support_at_eval"])
                    self.assertTrue(row["oracle_support_non_deployable"])
                else:
                    self.assertFalse(row["uses_future_hidden_or_delta"])
                    self.assertFalse(row["uses_task_id"])
                    self.assertFalse(row["uses_teacher_labels_in_deployable_router"])

            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["pregate_source_present"]["passed"])
            self.assertTrue(gates["required_arms_present"]["passed"])
            self.assertTrue(gates["oracle_support_non_deployable_labeled"]["passed"])
            self.assertTrue(gates["deployable_leakage_flags_false"]["passed"])
            self.assertEqual(len(summary["norm_control_rows"]), len(summary["arm_metrics"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "assay" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
