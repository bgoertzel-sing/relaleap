from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_pair_composer_closeout import (
    REDIRECT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_pair_composer_closeout,
)


class DenseTeacherPairComposerCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_pair_composer_closeout(
                probe_path=root / "missing_probe.json",
                truth_audit_path=root / "missing_truth.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_dense_mlp_control_dominance_selects_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / "probe.json"
            truth = root / "truth.json"
            review = root / "latest-review.md"
            _write_json(
                probe,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_control_extension_probe_gpu_blocked",
                    "claim_status": "pair_composer_interference_controls_recorded_gpu_blocked",
                    "selected_next_action": "close_or_redirect_pair_composer_after_interference_controls",
                    "advance_to_gpu_validation": False,
                    "oracle_holdout_true_decoder_ce_loss": 0.99,
                    "learned_router_holdout_true_decoder_ce_loss": 1.0,
                    "majority_pair_holdout_true_decoder_ce_loss": 9.5,
                    "control_rows": [
                        {
                            "arm": "same_parameter_independent_additive_control",
                            "split": "holdout",
                            "true_decoder_ce_loss": 1.8,
                        },
                        {
                            "arm": "matched_mlp_random_feature_residual_control",
                            "split": "holdout",
                            "true_decoder_ce_loss": 0.39,
                        },
                    ],
                    "gate_criteria": [
                        {
                            "criterion": "support_pair_class_balance_sufficient",
                            "passed": True,
                            "actual": {"majority_fraction": 0.07, "normalized_entropy": 0.94},
                        },
                        {"criterion": "matched_dense_mlp_control_rows_measured", "passed": True},
                        {"criterion": "exact_finite_update_commutator_measured", "passed": True},
                        {"criterion": "retention_churn_measured", "passed": True},
                        {"criterion": "pair_composer_beats_best_matched_control", "passed": False},
                        {"criterion": "remaining_controls_complete_for_gpu", "passed": False},
                    ],
                },
            )
            _write_json(
                truth,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_truth_audit_gpu_blocked",
                    "claim_status": "pair_composer_positive_signal_controls_recorded_but_not_cleared",
                    "advance_to_gpu_validation": False,
                    "pair_composer_vs_independent_ce_gain": 0.86,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Keep GPU blocked and add local controls.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_pair_composer_closeout(
                probe_path=probe,
                truth_audit_path=truth,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_teacher_pair_composer_branch_closed")
            self.assertEqual(summary["selected_next_action"], REDIRECT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["ben_should_be_notified"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
