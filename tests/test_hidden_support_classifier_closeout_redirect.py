from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.hidden_support_classifier_closeout_redirect import (
    CLOSE_AND_REDIRECT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_hidden_support_classifier_closeout_redirect,
)


class HiddenSupportClassifierCloseoutRedirectTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_hidden_support_classifier_closeout_redirect(
                hidden_audit_path=root / "missing_hidden_audit.json",
                hidden_closeout_rows_path=root / "missing_closeout_rows.csv",
                oracle_pregate_path=root / "missing_oracle_pregate.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_closed_hidden_branch_selects_oracle_overlap_redesign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden_audit = root / "hidden_audit.json"
            closeout_rows = root / "closeout_rows.csv"
            oracle_pregate = root / "oracle_pregate.json"
            review = root / "latest-review.md"
            _write_json(
                hidden_audit,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "close_hidden_classifier_branch": True,
                    "closeout_status": "closed_hidden_support_classifier_branch_before_gpu",
                    "sequence_heldout_gate_passes": False,
                    "rule_combo_heldout_gate_passes": False,
                    "budget_gate_passes": False,
                    "mean_hidden_classifier_ce_gain_vs_learned_router": -0.028,
                    "mean_oracle_regret_recovery_vs_learned_router": -0.93,
                },
            )
            closeout_rows.write_text(
                "\n".join(
                    [
                        "branch,status,next_step,deferred_exact_row_reason",
                        "direct_hidden_support_classifier,closed_hidden_support_classifier_branch_before_gpu,close_or_redesign_hidden_support_classifier_branch_before_gpu,sequence-heldout same-student intervention rows already lose to the learned router",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                oracle_pregate,
                {
                    "status": "pass",
                    "decision": "oracle_overlap_transformer_acsr_training_pregate_gpu_blocked",
                    "pregate_passes": False,
                    "selected_next_step": "replace_proxy_row_pregate_with_hidden_feature_same_student_intervention_training",
                    "primary_result": {
                        "uses_target_token_as_predictor_feature": False,
                        "uses_oracle_loss_as_predictor_feature": False,
                        "prefix_safe_feature_names": "position_index;normalized_position;learned_support_multihot",
                        "regret_recovery_fraction_vs_learned": -0.7,
                        "oracle_mean_jaccard_overlap": 0.2,
                    },
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet; repair hidden-classifier gate/report inconsistency.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_hidden_support_classifier_closeout_redirect(
                hidden_audit_path=hidden_audit,
                hidden_closeout_rows_path=closeout_rows,
                oracle_pregate_path=oracle_pregate,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "hidden_support_classifier_closed_redirect_selected")
            self.assertEqual(summary["selected_next_action"], CLOSE_AND_REDIRECT_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "direct_hidden_classifier_closed_oracle_overlap_redesign_selected",
            )
            self.assertTrue(summary["hidden_branch_closed"])
            self.assertTrue(summary["oracle_overlap_redesign_available"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("Oracle-overlap redesign available: `True`", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
