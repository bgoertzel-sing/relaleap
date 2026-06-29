from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_dense_teacher_control_branch_selector import (
    LOW_CHURN_MLP_ACTION,
    REQUIRED_ARTIFACTS,
    RUNPOD_REPEAT_ACTION,
    run_post_dense_teacher_control_branch_selector,
)


class PostDenseTeacherControlBranchSelectorTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_post_dense_teacher_control_branch_selector(
                dense_teacher_control_path=root / "missing_dense_teacher.json",
                dense_primary_path=root / "missing_dense_primary.json",
                mlp_followup_path=root / "missing_mlp_followup.json",
                matched_decision_path=root / "missing_matched.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], "repair_missing_dense_teacher_control_sources")
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_blocked_dense_teacher_selects_low_churn_mlp_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_teacher = root / "dense_teacher.json"
            dense_primary = root / "dense_primary.json"
            mlp_followup = root / "mlp_followup.json"
            matched = root / "matched.json"
            review = root / "latest-review.md"
            _write_json(
                dense_teacher,
                {
                    "status": "pass",
                    "decision": "dense_teacher_control_mechanism_assay_blocked",
                    "scientific_gate": "blocked",
                    "claim_status": "dense_teacher_acsr_not_supported_against_dense24_mlp_controls",
                    "requires_gpu_now": False,
                    "failures": [{"criterion": "matched_dense_control_gate_passed"}],
                },
            )
            _write_json(
                dense_primary,
                {
                    "status": "pass",
                    "decision": "dense_primary_mechanism_assay_selected",
                    "claim_status": "dense_or_mlp_control_selected_as_primary_mechanism_assay",
                    "primary_arm": "parameter_matched_causal_mlp_control",
                    "primary_family": "parameter_matched_dense_control",
                },
            )
            _write_json(
                mlp_followup,
                {
                    "status": "pass",
                    "decision": "mlp_primary_with_functional_churn_tradeoff",
                    "claim_status": "mlp_control_leads_ce_retention_fingerprint_but_dense_has_lower_churn",
                    "mechanism_comparison": [
                        {
                            "arm": "dense_rank24_best_norm",
                            "heldout_prediction_changed_vs_base": 0.24,
                            "heldout_residual_update_l2": 1.0,
                        },
                        {
                            "arm": "parameter_matched_causal_mlp_control",
                            "heldout_prediction_changed_vs_base": 0.84,
                            "heldout_residual_update_l2": 4.4,
                        },
                    ],
                },
            )
            _write_json(
                matched,
                {
                    "status": "pass",
                    "decision": "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail",
                    "scientific_gate": "blocked",
                    "claim_status": "mlp_or_sparse_advantage_not_decisive_after_ce_l2_churn_matching",
                    "advancement_row_count": 0,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run a local decoder-exported pregate before GPU validation.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_dense_teacher_control_branch_selector(
                dense_teacher_control_path=dense_teacher,
                dense_primary_path=dense_primary,
                mlp_followup_path=mlp_followup,
                matched_decision_path=matched,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_dense_teacher_control_branch_selected")
            self.assertEqual(summary["selected_next_action"], LOW_CHURN_MLP_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            actions = {row["candidate_action"]: row for row in summary["candidate_actions"]}
            self.assertEqual(actions[RUNPOD_REPEAT_ACTION]["disposition"], "rejected")
            self.assertIn("low-churn", summary["selected_next_step"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
