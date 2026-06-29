from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_residual_control_pregate import (
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_residual_control_pregate,
)


class LowChurnMlpResidualControlPregateTest(unittest.TestCase):
    def test_records_design_pregate_from_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            branch = root / "branch.json"
            followup = root / "followup.json"
            fingerprint = root / "fingerprint"
            decision = root / "decision.json"
            review = root / "latest-review.md"
            fingerprint.mkdir()

            _write_json(
                branch,
                {
                    "status": "pass",
                    "decision": "post_dense_teacher_control_branch_selected",
                    "selected_next_action": "design_low_churn_mlp_residual_control_pregate",
                },
            )
            _write_json(
                followup,
                {
                    "status": "pass",
                    "decision": "mlp_primary_with_functional_churn_tradeoff",
                    "mechanism_comparison": [
                        {
                            "arm": "dense_rank24_best_norm",
                            "heldout_ce_loss": 3.7,
                            "heldout_logit_mse_vs_base": 0.01,
                            "heldout_prediction_changed_vs_base": 0.2,
                            "heldout_residual_update_l2": 1.0,
                        },
                        {
                            "arm": "parameter_matched_causal_mlp_control",
                            "heldout_ce_loss": 2.8,
                            "heldout_logit_mse_vs_base": 0.15,
                            "heldout_prediction_changed_vs_base": 0.8,
                            "heldout_residual_update_l2": 4.0,
                        },
                    ],
                },
            )
            _write_json(
                fingerprint / "summary.json",
                {"status": "pass", "decision": "mlp_churn_intervention_fingerprint_scaled_assay_completed"},
            )
            (fingerprint / "scaled_interventions.csv").write_text(
                "\n".join(
                    [
                        "arm,lambda,ce_loss,residual_update_l2,logit_mse_vs_base,prediction_changed_vs_base",
                        "dense_rank24_best_norm,1.0,3.7,1.0,0.01,0.2",
                        "parameter_matched_causal_mlp_control,1.0,2.8,4.0,0.15,0.8",
                        "parameter_matched_causal_mlp_control,0.25,3.8,1.0,0.009,0.18",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (fingerprint / "scaled_match_summary.csv").write_text(
                "\n".join(
                    [
                        "match_type,reference_arm,arm,lambda,ce_loss,residual_update_l2,logit_mse_vs_base,prediction_changed_vs_base,distance",
                        "residual_l2,dense_rank24_best_norm,parameter_matched_causal_mlp_control,0.25,3.8,1.0,0.009,0.18,0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                decision,
                {
                    "status": "pass",
                    "decision": "return_to_sparse_acsr_support_diagnostics",
                    "promotion_allowed": False,
                    "requires_gpu_now": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run local pregates before GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_low_churn_mlp_residual_control_pregate(
                branch_selector_path=branch,
                mlp_followup_path=followup,
                mlp_fingerprint_dir=fingerprint,
                mlp_decision_path=decision,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_residual_control_pregate_recorded")
            self.assertEqual(summary["selected_next_action"], "implement_low_churn_mlp_residual_control_pilot")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            budgets = {row["metric"]: row for row in summary["budget_rows"]}
            self.assertEqual(budgets["dense24_residual_l2_ceiling"]["value"], 1.0)
            self.assertEqual(budgets["dense24_flip_churn_ceiling"]["value"], 0.2)
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(criteria["raw_mlp_is_high_norm_high_churn_vs_dense24"]["passed"])
            arms = {row["arm"]: row for row in summary["pregate_arms"]}
            self.assertIn("low_churn_mlp_residual_control", arms)
            self.assertIn("raw_intervention_fingerprint", arms["low_churn_mlp_residual_control"]["required_outputs"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_when_branch_selector_did_not_select_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            branch = root / "branch.json"
            followup = root / "followup.json"
            fingerprint = root / "fingerprint"
            decision = root / "decision.json"
            review = root / "latest-review.md"
            fingerprint.mkdir()
            _write_json(branch, {"status": "pass", "selected_next_action": "different_action"})
            _write_json(followup, {"status": "pass", "mechanism_comparison": []})
            _write_json(fingerprint / "summary.json", {"status": "pass"})
            (fingerprint / "scaled_interventions.csv").write_text("arm,lambda\n", encoding="utf-8")
            (fingerprint / "scaled_match_summary.csv").write_text("match_type,arm\n", encoding="utf-8")
            _write_json(decision, {"status": "pass", "promotion_allowed": False, "requires_gpu_now": False})
            review.write_text("strategic_change_level: minor\nnotify_ben: false\nverdict: FIX\n", encoding="utf-8")

            summary = run_low_churn_mlp_residual_control_pregate(
                branch_selector_path=branch,
                mlp_followup_path=followup,
                mlp_fingerprint_dir=fingerprint,
                mlp_decision_path=decision,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], "repair_low_churn_mlp_pregate_sources")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("branch_selector_selected_low_churn_mlp_pregate", failed)
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
