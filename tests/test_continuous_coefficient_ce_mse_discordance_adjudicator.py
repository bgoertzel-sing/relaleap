from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.continuous_coefficient_ce_mse_discordance_adjudicator import (
    DECISION,
    REQUIRED_ARTIFACTS,
    _claim_status,
    _selected_next_step,
    run_continuous_coefficient_ce_mse_discordance_adjudicator,
)


class ContinuousCoefficientCeMseDiscordanceAdjudicatorTests(unittest.TestCase):
    def test_runs_same_objective_adjudicator_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate_summary.json"
            pregate.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "continuous_coefficient_sparse_value_pregate_recorded",
                        "claim_status": "continuous_coeff_ce_mse_discordant_no_promotion",
                        "selected_next_step": "run continuous CE/MSE discordance adjudicator with same-objective flat controls and no GPU",
                        "ce_mse_discordant": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_continuous_coefficient_ce_mse_discordance_adjudicator(
                pregate_path=pregate,
                out_dir=root / "out",
                seed=7,
                teacher_steps=4,
                router_steps=4,
                value_steps=4,
                control_steps=4,
                column_count=4,
                coeff_dim=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertTrue(summary["teacher_trained"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["source_rows"][0]["claim_status"], "continuous_coeff_ce_mse_discordant_no_promotion")
            self.assertEqual(len(summary["objective_rows"]), 18)
            self.assertEqual(len(summary["scale_rows"]), 18)
            self.assertEqual(len(summary["coefficient_rows"]), 3)

            arms = {row["arm"] for row in summary["objective_rows"]}
            self.assertIn("ce_only_continuous_coeff_norm_matched", arms)
            self.assertIn("ce_only_same_router_flat_norm_matched", arms)
            self.assertIn("mse_only_continuous_coeff_half_scale", arms)
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["pregate_source_present"]["passed"])
            self.assertTrue(gates["pregate_was_discordant"]["passed"])
            self.assertTrue(gates["required_objective_rows_present"]["passed"])
            self.assertTrue(gates["deployable_leakage_flags_false"]["passed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_or_wrong_pregate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_continuous_coefficient_ce_mse_discordance_adjudicator(
                pregate_path=root / "missing.json",
                out_dir=root / "out",
                seed=9,
                teacher_steps=2,
                router_steps=2,
                value_steps=2,
                control_steps=2,
                column_count=3,
                coeff_dim=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "continuous_coefficient_ce_mse_discordance_adjudicator_failed_closed")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertFalse(gates["pregate_source_present"]["passed"])
            self.assertFalse(gates["pregate_was_discordant"]["passed"])

    def test_claim_status_prioritizes_flat_ce_and_mse_parity_failures(self) -> None:
        flat_ce_failures = [
            {
                "criterion": "ce_objective_continuous_beats_flat_ce",
                "passed": False,
                "required": False,
                "gate_type": "scientific",
                "evidence": "flat caught CE",
            }
        ]
        mse_failures = [
            {
                "criterion": "mse_objective_continuous_matches_flat_mse",
                "passed": False,
                "required": False,
                "gate_type": "scientific",
                "evidence": "flat MSE lower",
            }
        ]

        self.assertEqual(
            _claim_status("pass", flat_ce_failures),
            "continuous_coeff_ce_gain_not_objective_parity_supported_no_gpu",
        )
        self.assertEqual(
            _selected_next_step("pass", flat_ce_failures),
            "close continuous coefficients as a flat-control-confounded CE artifact before GPU",
        )
        self.assertEqual(
            _claim_status("pass", mse_failures),
            "continuous_coeff_ce_mse_discordance_persists_no_promotion",
        )
        self.assertEqual(
            _selected_next_step("pass", mse_failures),
            "redesign continuous coefficients with sparse/scale constraints before any GPU validation",
        )


if __name__ == "__main__":
    unittest.main()
