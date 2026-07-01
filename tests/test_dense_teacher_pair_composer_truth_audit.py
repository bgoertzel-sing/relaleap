from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_pair_composer_truth_audit import (
    AUDIT_DECISION_GPU_BLOCKED,
    CONTROL_EXTENSION_ACTION,
    REQUIRED_ARTIFACTS,
    REPAIR_ACTION,
    run_dense_teacher_pair_composer_truth_audit,
)


class DenseTeacherPairComposerTruthAuditTest(unittest.TestCase):
    def test_positive_pair_signal_is_recorded_but_gpu_blocked_without_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            localization = root / "localization.json"
            closeout = root / "closeout.json"
            review = root / "latest-review.md"
            _write_json(localization, _localization_payload())
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_pregate_positive_truth_audit_selected",
                    "claim_status": "dense_teacher_pair_composer_positive_local_signal_truth_audit_needed",
                    "selected_next_action": "run_local_pair_composer_truth_audit_before_gpu",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run a local true-decoder pair-interaction audit with leakage/null/interference gates before any GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_pair_composer_truth_audit(
                failure_localization_path=localization,
                closeout_path=closeout,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], AUDIT_DECISION_GPU_BLOCKED)
            self.assertEqual(summary["selected_next_action"], CONTROL_EXTENSION_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertGreater(summary["pair_metrics"]["pair_vs_independent_holdout_ce_gain"], 0.0)
            self.assertGreater(summary["pair_metrics"]["pair_vs_feature_count_null_holdout_ce_gain"], 0.0)
            self.assertLessEqual(summary["pair_metrics"]["pair_train_holdout_ce_gap"], 1.0)
            criteria = {row["criterion"]: row for row in summary["criteria"]}
            self.assertTrue(criteria["pair_beats_independent_holdout_ce"]["passed"])
            self.assertTrue(criteria["pair_beats_feature_count_null_holdout_ce"]["passed"])
            self.assertFalse(criteria["mechanism_controls_complete_for_gpu"]["passed"])
            controls = {row["control"]: row for row in summary["control_matrix"]}
            self.assertEqual(controls["deployable_pair_router"]["status"], "blocked")
            self.assertEqual(controls["finite_update_commutator"]["status"], "blocked")
            self.assertEqual(summary["strategy_response"]["recommendation_disposition"], "accepted")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_pair_composer_truth_audit(
                failure_localization_path=root / "missing-localization.json",
                closeout_path=root / "missing-closeout.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _localization_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "dense_teacher_failure_localization_evaluator_recorded",
        "claim_status": "local_evaluator_complete_no_columnability_claim",
        "composer_train_holdout_split_recorded": True,
        "composer_uses_true_frozen_decoder_for_ce": True,
        "pair_composer_pregate_rows": [
            {
                "arm": "retrained_oracle_support_values",
                "split": "train",
                "split_seed": 1729,
                "token_count": 151,
                "feature_count": 24,
                "true_decoder_ce_loss": 1.3746991157531738,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
            },
            {
                "arm": "retrained_oracle_support_values",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 24,
                "true_decoder_ce_loss": 1.8582007884979248,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
            },
            {
                "arm": "oracle_support_gated_value_pair_composer",
                "split": "train",
                "split_seed": 1729,
                "token_count": 151,
                "feature_count": 77,
                "true_decoder_ce_loss": 0.5741084814071655,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
            },
            {
                "arm": "oracle_support_gated_value_pair_composer",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 77,
                "true_decoder_ce_loss": 0.9914494156837463,
                "teacher_logit_residual_r2": 0.8688835501670837,
                "teacher_hidden_residual_r2": 0.8618878126144409,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
                "beats_independent_holdout_ce": True,
                "beats_shuffled_pair_null_holdout_ce": True,
            },
            {
                "arm": "feature_count_matched_shuffled_pair_null",
                "split": "train",
                "split_seed": 1729,
                "token_count": 151,
                "feature_count": 77,
                "true_decoder_ce_loss": 0.6530514359474182,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
            },
            {
                "arm": "feature_count_matched_shuffled_pair_null",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 77,
                "true_decoder_ce_loss": 2.937685012817383,
                "uses_true_frozen_decoder_for_ce": True,
                "train_holdout_split_recorded": True,
            },
        ],
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
