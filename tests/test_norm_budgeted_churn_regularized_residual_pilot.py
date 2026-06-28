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
            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in rows}
            self.assertIn("dense_rank24_norm_budgeted", arms)
            self.assertIn("sparse_contextual_topk2_norm_budgeted", arms)
            self.assertIn("bottleneck_gated_mlp_norm_budgeted", arms)
            self.assertTrue(all("scientific_gate" in row for row in rows))


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
