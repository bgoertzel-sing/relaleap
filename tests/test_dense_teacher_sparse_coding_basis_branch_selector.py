from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_sparse_coding_basis_branch_selector import (
    DEMOTE_BASIS_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_sparse_coding_basis_branch_selector,
)


class DenseTeacherSparseCodingBasisBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_sparse_coding_basis_branch_selector(
                oracle_feasibility_path=root / "missing_oracle.json",
                imitation_probe_path=root / "missing_imitation.json",
                bottleneck_path=root / "missing_bottleneck.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_demotes_current_basis_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oracle = root / "oracle.json"
            imitation = root / "imitation.json"
            bottleneck = root / "bottleneck.json"
            review = root / "latest-review.md"
            _write_json(
                oracle,
                {
                    "status": "pass",
                    "decision": "dense_teacher_oracle_sparse_coding_feasibility_recorded",
                    "claim_status": "oracle_sparse_coding_feasible_router_imitation_blocks_gpu",
                    "training_executed": True,
                    "requires_gpu_now": False,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "arm_metrics": [
                        _arm("oracle_topk_orthogonal_sparse_coding", 0.83904, 1.342901, 1.0),
                        _arm("same_router_flat_value_control", 0.689996, 1.399378, 0.8),
                    ],
                },
            )
            _write_json(
                imitation,
                {
                    "status": "pass",
                    "decision": "dense_teacher_deployable_sparse_coding_imitation_probe_recorded",
                    "claim_status": "support_conditioned_combo_value_head_fails_flat_control_blocks_gpu",
                    "training_executed": True,
                    "requires_gpu_now": False,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "imitation_rows": [
                        _arm("combo_mlp_router_scalar_imitation", 0.605014, 1.394933, 0.727044, 0.652775),
                        _arm("support_conditioned_combo_sparse_value_head", 0.453912, 1.440525, 0.550806, 1.420481),
                        _arm("same_router_flat_value_control", 0.695806, 1.408506, 0.832939, 0.652775),
                        _arm("random_topk_sparse_coding_null", 0.204918, 1.674558, 0.260391),
                        _arm("oracle_support_learned_combo_coeff_sparse_coding", 0.717529, 1.338474, 0.858276),
                        _arm("learned_combo_support_oracle_coeff_sparse_coding", 0.710914, 1.413747, 0.85056),
                    ],
                },
            )
            _write_json(
                bottleneck,
                {
                    "status": "pass",
                    "decision": "dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator_recorded",
                    "claim_status": "naive_support_conditioned_sparse_value_closed_flat_control_blocks_gpu",
                    "selected_next_action": "close_naive_support_conditioned_sparse_value_head",
                    "training_executed": False,
                    "requires_gpu_now": False,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "bottleneck_rows": [
                        _comparison("best_deployable_sparse_vs_same_router_flat", -0.090792, -0.105895),
                        _comparison("naive_support_conditioned_value_head_vs_combo", -0.151102, -0.176238),
                        _comparison("best_deployable_sparse_vs_random_null", 0.400096, 0.466653),
                        _comparison("oracle_support_learned_coeff_vs_oracle_sparse", -0.121511, -0.141724),
                        _comparison("learned_support_oracle_coeff_vs_oracle_sparse", -0.128126, -0.14944),
                    ],
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: true",
                        "recommended_next_action: run one local support-vs-coefficient/value bottleneck audit before GPU",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_sparse_coding_basis_branch_selector(
                oracle_feasibility_path=oracle,
                imitation_probe_path=imitation,
                bottleneck_path=bottleneck,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_teacher_sparse_coding_basis_branch_selected")
            self.assertEqual(summary["selected_next_action"], DEMOTE_BASIS_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "orthogonal_sparse_coding_basis_demoted_to_diagnostic_no_gpu",
            )
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["ben_notification_recommended"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            gates = {row["gate"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["oracle_sparse_basis_feasible"]["passed"])
            self.assertTrue(gates["deployable_sparse_signal_beats_null"]["passed"])
            self.assertTrue(gates["deployable_sparse_fails_oracle_retention_gate"]["passed"])
            self.assertTrue(gates["flat_control_blocks_deployable_sparse_r2"]["passed"])
            self.assertTrue(gates["support_conditioned_head_closed"]["passed"])
            self.assertTrue(gates["crossed_support_coefficients_not_single_dominant_blocker"]["passed"])
            selected = [row for row in summary["branch_rows"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            rejected = {
                row["candidate_action"]: row["disposition"]
                for row in summary["branch_rows"]
                if row["candidate_action"] == "launch_gpu_validation_for_sparse_coding_basis"
            }
            self.assertEqual(rejected["launch_gpu_validation_for_sparse_coding_basis"], "rejected")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _arm(
    arm: str,
    r2: float,
    ce: float,
    retention: float,
    coeff_mse: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "arm": arm,
        "teacher_residual_reconstruction_r2": r2,
        "ce": ce,
        "oracle_gain_retained_fraction": retention,
    }
    if coeff_mse is not None:
        row["coefficient_mse_vs_oracle"] = coeff_mse
    return row


def _comparison(name: str, r2_delta: float, retention_delta: float) -> dict[str, object]:
    return {
        "comparison": name,
        "r2_delta_primary_minus_comparator": r2_delta,
        "retention_delta_primary_minus_comparator": retention_delta,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
