from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_support_forcing_pruning_closeout import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_support_forcing_pruning_closeout,
)


class DenseTeacherSupportForcingPruningCloseoutTests(unittest.TestCase):
    def test_closes_support_forcing_pruning_branch_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pregate = root / "pregate"
            review = root / "latest-review.md"
            _write_pregate(pregate)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Replace generic selector churn with local dense-teacher residual columnability.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_support_forcing_pruning_closeout(
                pregate_dir=pregate,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(
                summary["claim_status"],
                "support_forcing_pruning_sparse_specific_claim_not_established",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            self.assertEqual(summary["direction_shift"]["recommendation_disposition"], "accepted")
            self.assertFalse(summary["evidence"]["sparse_specific_gate_passed"])
            self.assertTrue(summary["evidence"]["oracle_support_gate_passed"])
            self.assertTrue(summary["evidence"]["pruning_gate_passed"])

            closeout = {row["criterion"]: row for row in summary["closeout_rows"]}
            self.assertTrue(closeout["gpu_validation_blocked"]["passed"])
            self.assertTrue(closeout["sparse_specific_flat_control_gate_failed"]["passed"])
            self.assertTrue(closeout["oracle_support_is_non_deployable_ceiling"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertIn("sparse value/support redesign", selected[0]["next_step"])
            rejected = {row["candidate_action"] for row in summary["candidate_actions"] if row["disposition"] == "rejected"}
            self.assertIn("launch_gpu_validation", rejected)
            self.assertIn("treat_pruned_oracle_as_promotion_evidence", rejected)

            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_pregate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_dense_teacher_support_forcing_pruning_closeout(
                pregate_dir=root / "missing",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_pregate(path: Path) -> None:
    rows = [
        {
            "arm": "oracle_support_same_values",
            "ce": 1.588191,
            "teacher_residual_reconstruction_mse": 2.472112,
            "teacher_residual_reconstruction_r2": 0.062943,
            "teacher_ce_gap_closure_fraction": 0.019181,
            "finite_update_commutator_proxy": 11.172819,
            "oracle_support_non_deployable": True,
        },
        {
            "arm": "learned_support_same_values",
            "ce": 1.715321,
            "teacher_residual_reconstruction_mse": 2.54473,
            "teacher_residual_reconstruction_r2": 0.035417,
            "support_overlap_with_oracle": 0.585938,
            "oracle_support_non_deployable": False,
        },
        {
            "arm": "load_permuted_support_same_values",
            "ce": 2.103923,
            "teacher_residual_reconstruction_mse": 3.163487,
            "teacher_residual_reconstruction_r2": -0.199124,
            "oracle_support_non_deployable": False,
        },
        {
            "arm": "random_support_same_values",
            "ce": 2.0,
            "teacher_residual_reconstruction_mse": 3.25,
            "teacher_residual_reconstruction_r2": -0.25,
            "oracle_support_non_deployable": False,
        },
        {
            "arm": "same_router_flat_value_control",
            "ce": 1.406017,
            "teacher_residual_reconstruction_mse": 0.864636,
            "teacher_residual_reconstruction_r2": 0.672258,
            "oracle_support_non_deployable": False,
        },
        {
            "arm": "pruned_oracle_support_same_values",
            "ce": 1.472054,
            "teacher_residual_reconstruction_mse": 1.5,
            "teacher_residual_reconstruction_r2": 0.4,
            "finite_update_commutator_proxy": 5.879564,
            "oracle_support_non_deployable": True,
        },
    ]
    gates = [
        {"criterion": "oracle_support_beats_support_nulls_same_values", "passed": True},
        {"criterion": "learned_support_low_forcing_regret", "passed": True},
        {"criterion": "pruning_retains_oracle_ce_gap_closure", "passed": True},
        {"criterion": "sparse_specific_beats_flat_value_control", "passed": False},
    ]
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_support_forcing_pruning_pregate_recorded",
            "claim_status": "support_forcing_pruning_local_gates_block_gpu",
            "base_holdout_ce": 1.592115,
            "dense_teacher_holdout_ce": 1.387565,
            "dense_teacher_ce_improvement": 0.20455,
            "support_forcing_rows": rows,
            "gate_criteria": gates,
            "retained_columns_after_pruning": [1, 3, 4],
            "same_sparse_values_across_support_conditions": True,
            "causal_efficacy_pruning_executed": True,
            "requires_gpu_now": False,
            "advance_to_gpu_validation": False,
            "promotion_allowed": False,
            "git_commit": "pregate-commit",
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
