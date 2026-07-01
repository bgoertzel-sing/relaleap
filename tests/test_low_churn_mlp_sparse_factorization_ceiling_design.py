from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_ceiling_design import (
    IMPLEMENT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_sparse_factorization_ceiling_design,
)


class LowChurnMlpSparseFactorizationCeilingDesignTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_low_churn_mlp_sparse_factorization_ceiling_design(
                closeout_path=root / "missing_closeout.json",
                low_churn_pilot_path=root / "missing_low_churn.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_design_contract_with_required_arms_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            low_churn = root / "low_churn.json"
            review = root / "latest-review.md"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "context_contrastive_core_periphery_branch_closed",
                    "selected_next_action": "design_low_churn_mlp_sparse_factorization_ceiling",
                },
            )
            _write_json(
                low_churn,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_residual_control_pilot_completed",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Implement a local low-churn-MLP sparse-factorization ceiling.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_low_churn_mlp_sparse_factorization_ceiling_design(
                closeout_path=closeout,
                low_churn_pilot_path=low_churn,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], IMPLEMENT_ACTION)
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            arms = {row["arm"] for row in summary["support_arms"]}
            self.assertIn("oracle_support_sparse_ceiling", arms)
            self.assertIn("learned_router_sparse_factorization", arms)
            self.assertIn("token_position_router_sparse_factorization", arms)
            self.assertIn("route_scrambled_same_values", arms)
            self.assertIn("shuffled_teacher_residual_sparse_factorization", arms)
            controls = {row["control"] for row in summary["null_controls"]}
            self.assertIn("frequency_preserving_support_permutation", controls)
            metrics = {row["metric"] for row in summary["observable_rows"]}
            self.assertIn("finite_update_commutator", metrics)
            self.assertIn("intervention_fingerprint_specificity", metrics)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
