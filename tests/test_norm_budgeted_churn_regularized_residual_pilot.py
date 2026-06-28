from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.norm_budgeted_churn_regularized_residual_pilot import (
    REQUIRED_ARTIFACTS,
    run_norm_budgeted_churn_regularized_residual_pilot,
)


class NormBudgetedChurnRegularizedResidualPilotTest(unittest.TestCase):
    def test_missing_design_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_norm_budgeted_churn_regularized_residual_pilot(
                design_dir=root / "missing",
                out_dir=root / "out",
                train_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "norm_budgeted_churn_regularized_residual_pilot_failed_closed",
            )
            self.assertFalse(summary["promotion_allowed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_runs_bounded_local_pilot_from_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design"
            _write_design(design)

            summary = run_norm_budgeted_churn_regularized_residual_pilot(
                design_dir=design,
                out_dir=root / "out",
                train_steps=1,
                seed=3,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "norm_budgeted_churn_regularized_residual_pilot_completed",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["arm_count"], 6)
            self.assertGreater(summary["per_token_row_count"], 0)
            self.assertGreater(summary["residual_scale_diagnostic_row_count"], 0)
            self.assertEqual(summary["norm_target_curriculum_arm_count"], 2)
            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in rows}
            self.assertIn("dense_rank24_norm_budgeted", arms)
            self.assertIn("sparse_contextual_topk2_norm_budgeted", arms)
            self.assertIn("bottleneck_gated_mlp_norm_budgeted", arms)
            self.assertTrue(all("scientific_gate" in row for row in rows))
            self.assertTrue(all("heldout_anchor_kl_vs_base" in row for row in rows))
            self.assertTrue(all("off_target_anchor_kl_vs_base" in row for row in rows))
            self.assertTrue(all("norm_target_curriculum" in row for row in rows))
            self.assertTrue(all("norm_target_hit_rate" in row for row in rows))
            self.assertTrue(all("curriculum_stage_count" in row for row in rows))
            self.assertTrue(all("realized_residual_l2_trajectory" in row for row in rows))
            self.assertTrue(all("scale_gate_trainable" in row for row in rows))
            self.assertTrue(all("learned_residual_scale" in row for row in rows))
            self.assertTrue(all("evaluation_operating_point" in row for row in rows))
            curriculum_rows = {row["arm"]: row for row in rows if row["norm_target_curriculum"] == "True"}
            self.assertEqual(
                {"dense_rank24_norm_budgeted", "sparse_contextual_topk2_norm_budgeted"},
                set(curriculum_rows),
            )
            for row in curriculum_rows.values():
                self.assertGreaterEqual(float(row["norm_target_hit_rate"]), 0.0)
                self.assertLessEqual(float(row["norm_target_hit_rate"]), 1.0)
                self.assertEqual(row["curriculum_stage_count"], "4")
                self.assertTrue(row["realized_residual_l2_trajectory"])
            sparse = curriculum_rows["sparse_contextual_topk2_norm_budgeted"]
            self.assertEqual(sparse["scale_gate_trainable"], "True")
            self.assertEqual(sparse["evaluation_operating_point"], "learned_scale_gate")
            self.assertGreater(float(sparse["learned_residual_scale"]), 0.0)
            with (root / "out" / "per_token_metrics.csv").open(newline="", encoding="utf-8") as handle:
                token_rows = list(csv.DictReader(handle))
            self.assertGreater(len(token_rows), 0)
            token_fields = set(token_rows[0])
            self.assertIn("anchor_kl_vs_base", token_fields)
            self.assertIn("intervention_stratum", token_fields)
            self.assertIn("is_off_target_anchor", token_fields)
            self.assertIn("off_target_anchor", {row["intervention_stratum"] for row in token_rows})
            self.assertIn("target_heldout", {row["intervention_stratum"] for row in token_rows})
            with (root / "out" / "residual_scale_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
                scale_rows = list(csv.DictReader(handle))
            self.assertGreater(len(scale_rows), 0)
            scale_fields = set(scale_rows[0])
            self.assertIn("target_budget_fraction", scale_fields)
            self.assertIn("ce_delta_vs_scaled_dense24", scale_fields)
            self.assertIn("scaled_direction_signal", scale_fields)
            self.assertEqual(
                {"0.25", "0.5", "0.75", "1.0"},
                {row["target_budget_fraction"] for row in scale_rows if row["arm"] == "dense_rank24_norm_budgeted"},
            )


def _write_design(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_next_step": "implement the local low-step pilot from pilot_arms.csv and objective_terms.csv",
                "residual_l2_budget": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "pilot_arms.csv").write_text("arm\nplaceholder\n", encoding="utf-8")
    (path / "objective_terms.csv").write_text("term\nplaceholder\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
