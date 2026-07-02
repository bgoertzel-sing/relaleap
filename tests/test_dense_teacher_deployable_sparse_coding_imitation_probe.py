from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_deployable_sparse_coding_imitation_probe import (
    ARMS,
    DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_deployable_sparse_coding_imitation_probe,
)


class DenseTeacherDeployableSparseCodingImitationProbeTests(unittest.TestCase):
    def test_runs_deployable_imitation_probe_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oracle = root / "oracle_summary.json"
            oracle.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "dense_teacher_oracle_sparse_coding_feasibility_recorded",
                        "claim_status": "oracle_sparse_coding_feasible_router_imitation_blocks_gpu",
                        "selected_next_step": (
                            "improve deployable router/scalar imitation for the feasible oracle sparse-coding basis; "
                            "keep local CPU and require low oracle-gain regret before GPU"
                        ),
                        "requires_gpu_now": False,
                        "advance_to_gpu_validation": False,
                        "promotion_allowed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Implement a local dense-teacher residual-geometry/oracle-sparse-coding feasibility assay.",
                        "verdict: PIVOT",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run_dense_teacher_deployable_sparse_coding_imitation_probe(
                oracle_feasibility_path=oracle,
                strategy_review_path=review,
                out_dir=root / "probe",
                seed=7,
                teacher_steps=6,
                baseline_steps=6,
                enhanced_steps=8,
                combo_steps=8,
                control_steps=6,
                basis_size=4,
                top_k=2,
                hidden_dim=12,
                data_column_count=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertTrue(summary["teacher_trained"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["strategic_change_level"], "major")
            self.assertTrue(summary["ben_notification_recommended"])

            arms = {row["arm"] for row in summary["imitation_rows"]}
            self.assertTrue(set(ARMS).issubset(arms))
            by_arm = {row["arm"]: row for row in summary["imitation_rows"]}
            self.assertTrue(by_arm["oracle_topk_orthogonal_sparse_coding"]["oracle_support_non_deployable"])
            self.assertFalse(by_arm["enhanced_joint_mlp_router_scalar_imitation"]["uses_oracle_support_at_eval"])
            self.assertFalse(by_arm["combo_mlp_router_scalar_imitation"]["uses_oracle_support_at_eval"])
            for row in summary["imitation_rows"]:
                self.assertIn("teacher_residual_reconstruction_r2", row)
                self.assertIn("oracle_gain_retained_fraction", row)
                self.assertIn("oracle_mask_exact_cell_overlap", row)
                self.assertIn("oracle_selected_component_overlap", row)
                self.assertFalse(row["uses_future_hidden_or_delta"])
                self.assertFalse(row["uses_task_id"])

            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["oracle_feasibility_source_present"]["passed"])
            self.assertTrue(gates["oracle_feasibility_selected_router_imitation"]["passed"])
            self.assertTrue(gates["strategy_review_present"]["passed"])
            self.assertTrue(gates["required_arms_present"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])
            self.assertIn("combo_improves_linear_imitation", gates)
            self.assertIn("best_deployable_retains_oracle_gain", gates)
            self.assertIn("best_deployable_near_flat_control", gates)

            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "probe" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
