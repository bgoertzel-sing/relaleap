from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.orthogonalized_sparse_core_periphery_interference_pregate import (
    IMPLEMENT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_orthogonalized_sparse_core_periphery_interference_pregate,
)


class OrthogonalizedSparseCorePeripheryInterferencePregateTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_orthogonalized_sparse_core_periphery_interference_pregate(
                synthesis_path=root / "missing_synthesis.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_fail_closed_pregate_from_dense_mlp_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis.json"
            review = root / "latest-review.md"
            _write_json(
                synthesis,
                {
                    "status": "pass",
                    "decision": "dense_mlp_dominance_synthesized_sparse_interference_pregate_selected",
                    "claim_status": "pair_composer_closed_dense_mlp_dominance_sparse_interference_pregate_selected_no_gpu",
                    "selected_next_action": "design_orthogonalized_sparse_additive_core_periphery_interference_pregate",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Keep GPU blocked and implement the sparse interference pregate.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_orthogonalized_sparse_core_periphery_interference_pregate(
                synthesis_path=synthesis,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "orthogonalized_sparse_core_periphery_interference_pregate_recorded",
            )
            self.assertEqual(summary["selected_next_action"], IMPLEMENT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertTrue(summary["direction_shift_recorded"])
            arms = {row["arm"] for row in summary["mechanism_arms"]}
            self.assertIn("orthogonalized_sparse_additive_core_periphery", arms)
            self.assertIn("orthogonalized_sparse_no_norm_controller_ablation", arms)
            self.assertIn("orthogonalized_sparse_no_core_protection_ablation", arms)
            self.assertIn("orthogonalized_sparse_no_update_masks_ablation", arms)
            controls = {row["control"] for row in summary["matched_controls"]}
            self.assertIn("dense_ridge_residual", controls)
            self.assertIn("random_feature_mlp_residual", controls)
            self.assertIn("same_router_flat_value_mlp", controls)
            metrics = {row["metric"] for row in summary["observable_gates"]}
            self.assertIn("finite_update_commutator_symmetric_kl", metrics)
            self.assertIn("context_reuse_score", metrics)
            nulls = {row["control"] for row in summary["leakage_nulls"]}
            self.assertIn("feature_schema_hash", nulls)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
