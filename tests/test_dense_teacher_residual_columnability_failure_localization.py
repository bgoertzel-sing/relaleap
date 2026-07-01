from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_columnability_failure_localization import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_residual_columnability_failure_localization,
)


class DenseTeacherResidualColumnabilityFailureLocalizationTests(unittest.TestCase):
    def test_localizes_mixed_teacher_and_oracle_sparse_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_a = root / "seed29"
            source_b = root / "seed30"
            review = root / "latest-review.md"
            _write_assay_summary(source_a, base_ce=1.38, teacher_ce=1.10, oracle_mse=1.00, shuffled_mse=1.05)
            _write_assay_summary(source_b, base_ce=1.39, teacher_ce=1.40, oracle_mse=0.95, shuffled_mse=0.94)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: local dense-teacher residual columnability and no GPU",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_residual_columnability_failure_localization(
                source_dirs=(source_a, source_b),
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["complete_source_seed_count"], 2)
            self.assertTrue(summary["strategy_review_handling"]["ben_should_be_notified"])
            axes = {row["axis"]: row for row in summary["failure_axes"]}
            self.assertFalse(axes["teacher_training_adequacy"]["passed"])
            self.assertFalse(axes["sparse_residual_scale"]["passed"])
            self.assertFalse(axes["oracle_support_representability"]["passed"])
            self.assertFalse(axes["gpu_readiness"]["passed"])
            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(gates["source_artifacts_present"]["passed"])
            self.assertFalse(gates["teacher_training_replicates"]["passed"])
            self.assertFalse(gates["oracle_support_beats_shuffled_null_across_seeds"]["passed"])
            self.assertIn("value-capacity", summary["selected_next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_residual_columnability_failure_localization(
                source_dirs=(root / "missing_a", root / "missing_b"),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_assay_summary(
    path: Path,
    *,
    base_ce: float,
    teacher_ce: float,
    oracle_mse: float,
    shuffled_mse: float,
) -> None:
    path.mkdir(parents=True)
    teacher_l2 = 2.0
    oracle_l2 = 0.6
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_residual_columnability_assay_recorded",
            "claim_status": "local_dense_teacher_columnability_gates_block_gpu",
            "base_holdout_ce": base_ce,
            "dense_teacher_holdout_ce": teacher_ce,
            "failures": [
                {
                    "criterion": "dense_teacher_improves_base",
                    "gate_type": "scientific",
                    "passed": teacher_ce < base_ce,
                }
            ],
            "arm_metrics": [
                _arm(
                    "oracle_support_sparse_dictionary",
                    ce=1.34,
                    mse=oracle_mse,
                    residual_l2=oracle_l2,
                    teacher_l2=teacher_l2,
                    oracle=True,
                ),
                _arm(
                    "learned_causal_router_sparse_dictionary",
                    ce=1.32,
                    mse=oracle_mse - 0.02,
                    residual_l2=0.65,
                    teacher_l2=teacher_l2,
                ),
                _arm(
                    "token_position_router_null",
                    ce=1.33,
                    mse=oracle_mse + 0.01,
                    residual_l2=0.64,
                    teacher_l2=teacher_l2,
                ),
                _arm(
                    "shuffled_teacher_residual_null",
                    ce=1.35,
                    mse=shuffled_mse,
                    residual_l2=0.62,
                    teacher_l2=teacher_l2,
                ),
            ],
        },
    )


def _arm(
    arm: str,
    *,
    ce: float,
    mse: float,
    residual_l2: float,
    teacher_l2: float,
    oracle: bool = False,
) -> dict[str, object]:
    return {
        "arm": arm,
        "ce": ce,
        "ce_gap_vs_dense_teacher": 0.1,
        "teacher_residual_reconstruction_mse": mse,
        "oracle_support_regret": 0.0,
        "residual_l2_mean": residual_l2,
        "teacher_residual_l2_mean": teacher_l2,
        "functional_churn": 0.5,
        "finite_update_commutator_proxy": 0.4,
        "retention_proxy": 0.9,
        "intervention_selectivity_proxy": 0.5,
        "oracle_support_non_deployable": oracle,
        "uses_oracle_support_at_eval": oracle,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
