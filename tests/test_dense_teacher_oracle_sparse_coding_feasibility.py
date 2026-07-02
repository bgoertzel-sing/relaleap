from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_oracle_sparse_coding_feasibility import (
    ARMS,
    DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_oracle_sparse_coding_feasibility,
)


class DenseTeacherOracleSparseCodingFeasibilityTests(unittest.TestCase):
    def test_runs_oracle_sparse_coding_feasibility_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            support_forcing = root / "support_forcing.json"
            support_forcing.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "dense_teacher_support_forcing_pruning_pregate_recorded",
                        "claim_status": "support_forcing_pruning_local_gates_block_gpu",
                        "base_holdout_ce": 1.59,
                        "dense_teacher_holdout_ce": 1.39,
                        "requires_gpu_now": False,
                        "advance_to_gpu_validation": False,
                        "promotion_allowed": False,
                        "support_forcing_rows": [
                            {
                                "arm": "learned_support_same_values",
                                "ce": 1.71,
                                "ce_gap_vs_dense_teacher": 0.32,
                                "ce_improvement_vs_base": -0.12,
                                "teacher_ce_gap_closure_fraction": -0.6,
                                "teacher_residual_reconstruction_mse": 2.0,
                                "teacher_residual_reconstruction_r2": 0.03,
                                "functional_churn": 0.5,
                                "retention_proxy": 0.8,
                                "finite_update_commutator_proxy": 11.0,
                                "intervention_selectivity_proxy": 0.1,
                                "support_load_entropy": 0.9,
                                "support_overlap_with_oracle": 0.58,
                                "active_rank_proxy": 4,
                                "residual_l2_mean": 1.0,
                                "residual_l2_p95": 1.5,
                                "teacher_residual_l2_mean": 1.2,
                                "residual_l2_mean_ratio_vs_teacher": 0.8,
                                "active_params": 12,
                                "stored_params": 100,
                            }
                        ],
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

            summary = run_dense_teacher_oracle_sparse_coding_feasibility(
                support_forcing_path=support_forcing,
                strategy_review_path=review,
                out_dir=root / "assay",
                seed=7,
                teacher_steps=6,
                router_steps=6,
                control_steps=6,
                basis_size=4,
                top_k=2,
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

            arms = {row["arm"] for row in summary["arm_metrics"]}
            self.assertTrue(set(ARMS).issubset(arms))
            self.assertTrue(summary["spectrum_rows"])
            for row in summary["spectrum_rows"]:
                self.assertIn("effective_rank", row)
                self.assertIn("cumulative_energy_fraction", row)
                self.assertGreaterEqual(row["effective_rank"], 1.0)

            by_arm = {row["arm"]: row for row in summary["arm_metrics"]}
            self.assertTrue(by_arm["oracle_topk_orthogonal_sparse_coding"]["uses_oracle_support_at_eval"])
            self.assertTrue(by_arm["oracle_topk_orthogonal_sparse_coding"]["oracle_support_non_deployable"])
            self.assertFalse(by_arm["learned_router_topk_scalar_sparse_coding"]["uses_oracle_support_at_eval"])
            self.assertEqual(by_arm["current_sparse_support_forcing_reference"]["row_source"], "consumed_support_forcing_pruning_artifact")
            for row in summary["arm_metrics"]:
                self.assertIn("teacher_residual_reconstruction_r2", row)
                self.assertIn("teacher_ce_gap_closure_fraction", row)
                self.assertIn("finite_update_commutator_proxy", row)
                self.assertIn("support_overlap_with_oracle", row)

            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["support_forcing_source_present"]["passed"])
            self.assertTrue(gates["strategy_review_present"]["passed"])
            self.assertTrue(gates["required_arms_present"]["passed"])
            self.assertTrue(gates["spectrum_rows_present"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])
            self.assertIn("oracle_sparse_beats_flat_or_is_near", gates)
            self.assertIn("learned_support_retains_oracle_gain", gates)

            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "assay" / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
