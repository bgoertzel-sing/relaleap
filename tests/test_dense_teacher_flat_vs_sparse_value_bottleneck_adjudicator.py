from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator import (
    DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator,
)


class DenseTeacherFlatVsSparseValueBottleneckAdjudicatorTests(unittest.TestCase):
    def test_records_flat_control_blocker_and_closes_naive_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / "probe_summary.json"
            review = root / "latest-review.md"
            _write_json(
                probe,
                {
                    "status": "pass",
                    "decision": "dense_teacher_deployable_sparse_coding_imitation_probe_recorded",
                    "claim_status": "support_conditioned_combo_value_head_fails_flat_control_blocks_gpu",
                    "selected_next_step": "close naive support-conditioned combo coefficient head and run a local flat-vs-sparse value bottleneck adjudicator before GPU",
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "imitation_rows": [
                        _arm("oracle_topk_orthogonal_sparse_coding", 0.84, 1.34, 1.0, 0.0, 1.0),
                        _arm("combo_mlp_router_scalar_imitation", 0.61, 1.39, 0.73, 0.65, 0.65),
                        _arm("support_conditioned_combo_sparse_value_head", 0.45, 1.44, 0.55, 1.42, 0.65),
                        _arm("same_router_flat_value_control", 0.70, 1.40, 0.83, 0.65, 0.65),
                        _arm("oracle_support_learned_combo_coeff_sparse_coding", 0.72, 1.34, 0.86, 0.65, 1.0),
                        _arm("learned_combo_support_oracle_coeff_sparse_coding", 0.71, 1.41, 0.85, 0.0, 0.65),
                        _arm("random_topk_sparse_coding_null", 0.20, 1.51, 0.24, 0.0, 0.25),
                        _arm("no_update_control", -0.02, 1.59, 0.0, 0.0, 0.0),
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
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run_dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator(
                imitation_probe_path=probe,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(
                summary["claim_status"],
                "naive_support_conditioned_sparse_value_closed_flat_control_blocks_gpu",
            )
            self.assertEqual(summary["selected_next_action"], "close_naive_support_conditioned_sparse_value_head")
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["ben_notification_recommended"])
            comparisons = {row["comparison"]: row for row in summary["bottleneck_rows"]}
            self.assertLess(
                comparisons["naive_support_conditioned_value_head_vs_combo"]["r2_delta_primary_minus_comparator"],
                0.0,
            )
            self.assertLess(
                comparisons["best_deployable_sparse_vs_same_router_flat"]["r2_delta_primary_minus_comparator"],
                0.0,
            )
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["imitation_probe_source_present"]["passed"])
            self.assertTrue(gates["imitation_probe_blocks_gpu"]["passed"])
            self.assertTrue(gates["required_key_arms_present"]["passed"])
            self.assertFalse(gates["support_conditioned_improves_combo_r2"]["passed"])
            self.assertFalse(gates["best_sparse_beats_flat_r2"]["passed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_probe_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator(
                imitation_probe_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "dense_teacher_flat_vs_sparse_value_bottleneck_adjudicator_failed_closed",
            )
            self.assertEqual(summary["selected_next_action"], "repair_adjudicator_sources")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _arm(
    arm: str,
    r2: float,
    ce: float,
    retention: float,
    coeff_mse: float,
    overlap: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "teacher_residual_reconstruction_r2": r2,
        "ce": ce,
        "oracle_gain_retained_fraction": retention,
        "coefficient_mse_vs_oracle": coeff_mse,
        "oracle_selected_component_overlap": overlap,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
