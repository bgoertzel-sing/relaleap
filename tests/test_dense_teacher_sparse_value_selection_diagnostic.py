from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_sparse_value_selection_diagnostic import (
    DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_sparse_value_selection_diagnostic,
)


class DenseTeacherSparseValueSelectionDiagnosticTests(unittest.TestCase):
    def test_records_value_selection_diagnostic_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            (source / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "dense_teacher_residual_value_capacity_norm_assay_recorded",
                        "claim_status": "value_capacity_norm_control_local_gates_block_gpu",
                        "selected_next_step": "diagnose learned sparse routing/value selection versus flat-value control dominance before GPU",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_sparse_value_selection_diagnostic(
                source_dir=source,
                out_dir=root / "out",
                seed=7,
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
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["oracle_value_code_non_deployable"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["uses_future_hidden_or_delta"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["deployable_router_uses_oracle_support"])

            arms = {row["arm"] for row in summary["diagnostic_rows"]}
            self.assertTrue(
                {
                    "oracle_support_learned_value_code_sparse",
                    "oracle_support_oracle_value_code_sparse",
                    "learned_support_learned_value_code_sparse",
                    "learned_support_oracle_value_code_sparse",
                    "global_oracle_support_value_code_sparse",
                    "same_router_flat_value_control",
                    "random_support_oracle_value_code_null",
                }.issubset(arms)
            )
            axes = {row["axis"] for row in summary["failure_axis_rows"]}
            self.assertTrue(
                {
                    "value_code_selection_regret",
                    "support_routing_regret_with_oracle_value_code",
                    "sparse_formulation_gap_vs_flat_value",
                    "learned_sparse_gap_vs_flat_value",
                    "oracle_support_value_over_random_support_value",
                    "in_column_gap_vs_global_dictionary_upper_bound",
                }.issubset(axes)
            )
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["source_assay_present"]["passed"])
            self.assertTrue(gates["diagnostic_rows_present"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])
            self.assertTrue(gates["deployable_leakage_flags_false"]["passed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
