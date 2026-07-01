from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_pair_composer_pregate_closeout import (
    CLOSEOUT_DECISION,
    NEXT_ACTION,
    POSITIVE_AUDIT_DECISION,
    REPAIR_ACTION,
    TRUTH_AUDIT_ACTION,
    run_dense_teacher_pair_composer_pregate_closeout,
)


class DenseTeacherPairComposerPregateCloseoutTest(unittest.TestCase):
    def test_closes_negative_pregate_and_redirects_to_core_periphery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            localization = root / "localization.json"
            review = root / "latest-review.md"
            _write_json(localization, _localization_payload())
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run a local decoder-exported pair-composer pregate.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_pair_composer_pregate_closeout(
                failure_localization_path=localization,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], CLOSEOUT_DECISION)
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["strategy_response"]["recommendation_disposition"], "accepted")
            self.assertFalse(summary["strategy_response"]["ben_should_be_notified"])
            self.assertFalse(summary["evidence"]["pair_beats_feature_count_null"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], NEXT_ACTION)
            for artifact in (
                "summary.json",
                "source_rows.csv",
                "gate_criteria.csv",
                "candidate_actions.csv",
                "pregate_metrics.csv",
                "notes.md",
            ):
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_positive_pair_composer_signal_selects_truth_audit_not_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            localization = root / "localization.json"
            review = root / "latest-review.md"
            payload = _localization_payload()
            payload["no_gpu_pregate_status"] = "pass"
            payload["composer_validation_blocker"] = ""
            for row in payload["pair_composer_pregate_rows"]:  # type: ignore[index]
                if row["arm"] == "oracle_support_gated_value_pair_composer":
                    row["true_decoder_ce_loss"] = 0.9914494156837463
                    row["teacher_logit_residual_r2"] = 0.8688835501670837
                    row["teacher_hidden_residual_r2"] = 0.8618878126144409
                    row["beats_shuffled_pair_null_holdout_ce"] = True
                    row["feature_count"] = 77
                if row["arm"] == "retrained_oracle_support_values":
                    row["true_decoder_ce_loss"] = 1.8582007884979248
                if row["arm"] == "feature_count_matched_shuffled_pair_null":
                    row["true_decoder_ce_loss"] = 2.937685012817383
                    row["feature_count"] = 77
            _write_json(localization, payload)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Fix the incoherent pair-composer closeout and run a local truth audit before any GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_pair_composer_pregate_closeout(
                failure_localization_path=localization,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POSITIVE_AUDIT_DECISION)
            self.assertEqual(summary["selected_next_action"], TRUTH_AUDIT_ACTION)
            self.assertNotEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["evidence"]["pair_beats_feature_count_null"])
            criteria = {row["criterion"]: row for row in summary["criteria"]}
            self.assertTrue(criteria["feature_count_matched_null_result_recorded"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], TRUTH_AUDIT_ACTION)

    def test_missing_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_pair_composer_pregate_closeout(
                failure_localization_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())


def _localization_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "dense_teacher_failure_localization_evaluator_recorded",
        "claim_status": "local_evaluator_complete_no_columnability_claim",
        "selected_next_step": "record negative local pair-composer pregate evidence",
        "composer_train_holdout_split_recorded": True,
        "composer_uses_true_frozen_decoder_for_ce": True,
        "composer_ce_metric_path": "true_frozen_decoder",
        "no_gpu_pregate_status": "fail",
        "composer_validation_blocker": (
            "split true-decoder pair-composer pregate failed at least one heldout performance control"
        ),
        "pair_composer_pregate_rows": [
            {
                "arm": "retrained_oracle_support_values",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 24,
                "true_decoder_ce_loss": 2.6465365886688232,
            },
            {
                "arm": "oracle_support_gated_value_pair_composer",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 67,
                "true_decoder_ce_loss": 2.234565019607544,
                "teacher_logit_residual_r2": 0.7122585773468018,
                "teacher_hidden_residual_r2": 0.71150803565979,
                "beats_independent_holdout_ce": True,
                "beats_shuffled_pair_null_holdout_ce": False,
            },
            {
                "arm": "feature_count_matched_shuffled_pair_null",
                "split": "holdout",
                "split_seed": 1729,
                "token_count": 101,
                "feature_count": 67,
                "true_decoder_ce_loss": 2.016756057739258,
            },
        ],
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
