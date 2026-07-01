from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.continuous_coefficient_sparse_value_pregate import (
    DECISION,
    REQUIRED_ARTIFACTS,
    _claim_status,
    _discordance_flags,
    _selected_next_step,
    run_continuous_coefficient_sparse_value_pregate,
)


class ContinuousCoefficientSparseValuePregateTests(unittest.TestCase):
    def test_trains_continuous_coefficient_pregate_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector = root / "selector.json"
            selector.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "post_dense_teacher_sparse_dictionary_branch_selected",
                        "claim_status": "continuous_coefficient_sparse_value_pregate_selected_no_gpu",
                        "selected_next_action": "design_continuous_coefficient_sparse_value_pregate",
                        "selected_next_step": "implement a local continuous-coefficient sparse-value pregate before any GPU validation",
                        "training_executed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_continuous_coefficient_sparse_value_pregate(
                selector_path=selector,
                out_dir=root / "pregate",
                seed=7,
                teacher_steps=5,
                router_steps=5,
                value_steps=5,
                control_steps=5,
                column_count=4,
                coeff_dim=2,
                values_per_column=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertTrue(summary["teacher_trained"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["source_rows"][0]["selected_next_action"], "design_continuous_coefficient_sparse_value_pregate")
            self.assertFalse(summary["uses_future_oracle_task_flags"]["uses_future_hidden_or_delta"])
            self.assertFalse(summary["uses_future_oracle_task_flags"]["deployable_router_uses_oracle_support"])

            arms = {row["arm"] for row in summary["arm_metrics"]}
            self.assertTrue(
                {
                    "continuous_coeff_oracle_support_ceiling",
                    "continuous_coeff_learned_support",
                    "hard_dictionary_learned_support_control",
                    "same_router_flat_value_control",
                    "random_support_continuous_null",
                    "frequency_support_continuous_null",
                    "shuffled_target_continuous_null",
                }.issubset(arms)
            )
            for row in summary["arm_metrics"]:
                self.assertIn("teacher_residual_reconstruction_mse", row)
                self.assertIn("active_params", row)
                self.assertIn("stored_params", row)
                self.assertFalse(row["uses_future_hidden_or_delta"])
                self.assertFalse(row["uses_task_id"])
                if row["arm"] == "continuous_coeff_oracle_support_ceiling":
                    self.assertTrue(row["oracle_support_non_deployable"])

            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["selector_source_present"]["passed"])
            self.assertTrue(gates["selector_chose_continuous_coefficients"]["passed"])
            self.assertTrue(gates["required_arms_present"]["passed"])
            self.assertTrue(gates["deployable_leakage_flags_false"]["passed"])
            self.assertTrue(summary["coefficient_rows"])
            self.assertIn("coeff_near_zero_fraction", summary["coefficient_rows"][0])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "pregate" / artifact).is_file(), artifact)

    def test_missing_selector_fails_closed_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_continuous_coefficient_sparse_value_pregate(
                selector_path=root / "missing.json",
                out_dir=root / "pregate",
                seed=9,
                teacher_steps=2,
                router_steps=2,
                value_steps=2,
                control_steps=2,
                column_count=3,
                coeff_dim=2,
                values_per_column=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "continuous_coefficient_sparse_value_pregate_failed_closed")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertFalse(gates["selector_source_present"]["passed"])
            self.assertFalse(gates["selector_chose_continuous_coefficients"]["passed"])

    def test_ce_positive_mse_negative_dense_coefficients_are_discordant(self) -> None:
        arm_rows = [
            {
                "arm": "continuous_coeff_learned_support",
                "ce": 1.37,
                "dense_teacher_ce": 1.39,
                "teacher_residual_reconstruction_mse": 1.23,
            },
            {
                "arm": "same_router_flat_value_control",
                "ce": 1.41,
                "dense_teacher_ce": 1.39,
                "teacher_residual_reconstruction_mse": 0.84,
            },
        ]
        coefficient_rows = [{"coeff_near_zero_fraction": 0.01}]
        failures = [
            {
                "criterion": "continuous_close_to_flat_value_mse",
                "passed": False,
                "required": False,
                "gate_type": "scientific",
                "evidence": "continuous_mse=1.23; flat_mse=0.84",
            },
            {
                "criterion": "coefficients_not_dense_like",
                "passed": False,
                "required": False,
                "gate_type": "scientific",
                "evidence": "coeff_near_zero_fraction=0.01",
            },
        ]

        discordance = _discordance_flags(arm_rows, coefficient_rows)

        self.assertTrue(discordance["ce_guardrail_positive"])
        self.assertTrue(discordance["teacher_mse_negative"])
        self.assertTrue(discordance["sparsity_gate_failed"])
        self.assertTrue(discordance["ce_mse_discordant"])
        self.assertEqual(_claim_status("pass", failures, discordance), "continuous_coeff_ce_mse_discordant_no_promotion")
        self.assertEqual(
            _selected_next_step("pass", failures, discordance),
            "run continuous CE/MSE discordance adjudicator with same-objective flat controls and no GPU",
        )


if __name__ == "__main__":
    unittest.main()
