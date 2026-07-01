from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_dense_teacher_sparse_dictionary_branch_selector import (
    CONTINUOUS_COEFFICIENT_ACTION,
    HARD_DICTIONARY_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_post_dense_teacher_sparse_dictionary_branch_selector,
)


class PostDenseTeacherSparseDictionaryBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_post_dense_teacher_sparse_dictionary_branch_selector(
                closeout_path=root / "missing_closeout.json",
                diagnostic_path=root / "missing_diagnostic.json",
                capacity_assay_path=root / "missing_capacity.json",
                failure_localization_path=root / "missing_failure.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["training_executed"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_kills_hard_dictionary_and_selects_continuous_coefficient_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closeout = root / "closeout.json"
            diagnostic = root / "diagnostic.json"
            capacity = root / "capacity.json"
            failure = root / "failure.json"
            review = root / "latest-review.md"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "dense_teacher_sparse_value_formulation_variant_closed_no_gpu",
                    "claim_status": "current_sparse_dictionary_value_formulation_retired_before_gpu",
                    "selected_next_step": "run a post-dense-teacher sparse-dictionary branch selector",
                    "training_executed": False,
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                    "evidence": {
                        "base_holdout_ce": 1.592115,
                        "dense_teacher_holdout_ce": 1.387565,
                        "dense_teacher_ce_improvement": 0.20455,
                        "flat_value_ce": 1.406017,
                        "learned_sparse_ce": 1.715321,
                        "learned_sparse_ce_gap_vs_flat": 0.309304,
                        "flat_value_mse": 0.864636,
                        "oracle_in_column_value_mse": 1.97803,
                        "global_dictionary_value_mse": 1.112734,
                        "oracle_in_column_mse_gap_vs_flat": 1.113394,
                        "global_dictionary_mse_gap_vs_flat": 0.248098,
                        "oracle_support_mse_advantage_vs_random": 0.552661,
                        "value_code_selection_regret": 0.494082,
                        "in_column_gap_vs_global": 0.865296,
                        "deployable_leakage_flags_false": True,
                        "oracle_value_code_non_deployable": True,
                    },
                },
            )
            _write_json(
                diagnostic,
                {
                    "status": "pass",
                    "decision": "dense_teacher_sparse_value_selection_diagnostic_recorded",
                    "claim_status": "sparse_value_formulation_and_code_selection_block_gpu",
                    "base_holdout_ce": 1.592115,
                    "dense_teacher_holdout_ce": 1.387565,
                    "training_executed": True,
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                    "failures": [{"criterion": "sparse_formulation_not_worse_than_flat", "passed": False}],
                },
            )
            _write_json(
                capacity,
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_value_capacity_norm_assay_recorded",
                    "claim_status": "value_capacity_norm_control_local_gates_block_gpu",
                    "training_executed": True,
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                failure,
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_columnability_failure_localization_recorded",
                    "claim_status": "dense_teacher_residual_columnability_failure_localized_gpu_blocked",
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
                        "recommended_next_action: Implement a local post-dense-teacher branch selector that kills the current hard sparse formulation and selects continuous coefficients.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_dense_teacher_sparse_dictionary_branch_selector(
                closeout_path=closeout,
                diagnostic_path=diagnostic,
                capacity_assay_path=capacity,
                failure_localization_path=failure,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_dense_teacher_sparse_dictionary_branch_selected")
            self.assertEqual(summary["selected_next_action"], CONTINUOUS_COEFFICIENT_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "continuous_coefficient_sparse_value_pregate_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["training_executed"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            selected = [row for row in summary["branch_rows"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            killed = {
                row["candidate_action"]: row["disposition"]
                for row in summary["branch_rows"]
                if row["candidate_action"] == HARD_DICTIONARY_ACTION
            }
            self.assertEqual(killed[HARD_DICTIONARY_ACTION], "killed")
            gates = {row["gate"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["flat_value_dominates_non_deployable_sparse_ceilings"]["passed"])
            self.assertTrue(gates["deployable_sparse_loses_ce_guardrail"]["passed"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Hard in-column sparse value-code dictionary branch is killed", notes)
            self.assertIn(CONTINUOUS_COEFFICIENT_ACTION, notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
