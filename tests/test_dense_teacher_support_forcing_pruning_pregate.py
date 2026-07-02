from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_support_forcing_pruning_pregate import (
    DECISION,
    REQUIRED_ARTIFACTS,
    SUPPORT_FORCING_ARMS,
    run_dense_teacher_support_forcing_pruning_pregate,
)


class DenseTeacherSupportForcingPruningPregateTests(unittest.TestCase):
    def test_runs_support_forcing_pruning_pregate_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector = root / "selector.json"
            selector.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "post_dense_teacher_sparse_dictionary_branch_selected",
                        "claim_status": "dense_teacher_support_forcing_pruning_pregate_selected_no_gpu",
                        "selected_next_action": "design_dense_teacher_support_forcing_pruning_pregate",
                        "requires_gpu_now": False,
                        "advance_to_gpu_validation": False,
                        "promotion_allowed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_support_forcing_pruning_pregate(
                selector_path=selector,
                out_dir=root / "pregate",
                seed=11,
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
            self.assertTrue(summary["teacher_trained"])
            self.assertTrue(summary["same_sparse_values_across_support_conditions"])
            self.assertTrue(summary["causal_efficacy_pruning_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])

            arms = {row["arm"] for row in summary["support_forcing_rows"]}
            self.assertTrue(set(SUPPORT_FORCING_ARMS).issubset(arms))
            sparse_value_ids = {
                row["value_model_id"]
                for row in summary["support_forcing_rows"]
                if row["arm"] != "same_router_flat_value_control" and "same_values" in row["arm"]
            }
            self.assertEqual(sparse_value_ids, {"oracle_trained_multi_value_dictionary_v1"})

            by_arm = {row["arm"]: row for row in summary["support_forcing_rows"]}
            self.assertTrue(by_arm["oracle_support_same_values"]["uses_oracle_support_at_eval"])
            self.assertTrue(by_arm["oracle_support_same_values"]["oracle_support_non_deployable"])
            self.assertFalse(by_arm["learned_support_same_values"]["uses_oracle_support_at_eval"])
            self.assertEqual(by_arm["learned_support_same_values"]["support_forcing_condition"], "learned_deployable")
            self.assertEqual(by_arm["load_permuted_support_same_values"]["support_forcing_condition"], "load_permuted_null")

            for row in summary["support_forcing_rows"]:
                self.assertIn("teacher_residual_reconstruction_r2", row)
                self.assertIn("teacher_ce_gap_closure_fraction", row)
                self.assertIn("finite_update_commutator_proxy", row)
                self.assertIn("retention_proxy", row)
                self.assertIn("support_overlap_with_oracle", row)
                self.assertIn("causal_efficacy_pruned", row)

            self.assertEqual(len(summary["pruning_rows"]), 4)
            self.assertTrue(any(row["retain_after_pruning"] for row in summary["pruning_rows"]))
            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["selector_source_present"]["passed"])
            self.assertTrue(gates["required_support_forcing_arms_present"]["passed"])
            self.assertTrue(gates["same_sparse_values_across_support_conditions"]["passed"])
            self.assertTrue(gates["oracle_support_non_deployable_labeled"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])

            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "pregate" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
