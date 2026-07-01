from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_value_capacity_norm_pregate import (
    DECISION,
    FAIL_DECISION,
    NEXT_STEP,
    REPAIR_STEP,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_residual_value_capacity_norm_pregate,
)


class DenseTeacherResidualValueCapacityNormPregateTests(unittest.TestCase):
    def test_records_redesign_contract_from_failure_localization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failure = root / "failure"
            seed_a = root / "seed_a"
            seed_b = root / "seed_b"
            review = root / "latest-review.md"
            _write_failure_localization(failure)
            _write_assay(seed_a)
            _write_assay(seed_b)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: local value-capacity and norm-control pregate",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_residual_value_capacity_norm_pregate(
                failure_localization_dir=failure,
                assay_dirs=(seed_a, seed_b),
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(summary["selected_next_step"], NEXT_STEP)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["strategy_review_handling"]["ben_should_be_notified"])
            families = {row["family"] for row in summary["redesign_arms"]}
            self.assertIn("oracle_sparse_ceiling", families)
            self.assertIn("deployable_candidate", families)
            self.assertIn("support_nulls", families)
            self.assertIn("target_nulls", families)
            self.assertGreaterEqual(len(summary["norm_control_contract"]), 4)
            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["required_sources_present"]["passed"])
            self.assertTrue(gates["previous_local_gates_block_gpu"]["passed"])
            self.assertTrue(gates["redesign_includes_oracle_deployable_and_null_families"]["passed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_residual_value_capacity_norm_pregate(
                failure_localization_dir=root / "missing_failure",
                assay_dirs=(root / "missing_a", root / "missing_b"),
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertEqual(summary["selected_next_step"], REPAIR_STEP)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_failure_localization(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_residual_columnability_failure_localization_recorded",
            "claim_status": "dense_teacher_residual_columnability_failure_localized_gpu_blocked",
            "localization": {
                "interpretation": "teacher adequacy is mixed, sparse residuals are under-scale, and nulls are too competitive"
            },
            "source_seeds": [
                {
                    "present": True,
                    "teacher_ce_improvement": 0.28,
                    "oracle_to_teacher_l2_ratio": 0.23,
                    "oracle_sparse_mse_advantage_vs_shuffled_null": 0.04,
                },
                {
                    "present": True,
                    "teacher_ce_improvement": -0.003,
                    "oracle_to_teacher_l2_ratio": 0.33,
                    "oracle_sparse_mse_advantage_vs_shuffled_null": -0.009,
                },
            ],
        },
    )


def _write_assay(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_residual_columnability_assay_recorded",
            "claim_status": "local_dense_teacher_columnability_gates_block_gpu",
            "arm_metrics": [
                {"arm": "dense_teacher_residual_control"},
                {"arm": "rank_matched_residual_control"},
                {"arm": "norm_clipped_mlp_control"},
                {"arm": "oracle_support_sparse_dictionary"},
                {"arm": "learned_causal_router_sparse_dictionary"},
            ],
        },
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
