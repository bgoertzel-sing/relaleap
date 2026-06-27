from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_residual_control_baseline_report import (
    REQUIRED_ARTIFACTS,
    SELECTED_STEP,
    run_dense_residual_control_baseline_report,
)


class DenseResidualControlBaselineReportTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_residual_control_baseline_report(
                common_benchmark_path=root / "missing-common.json",
                branch_selector_path=root / "missing-selector.json",
                dense_teacher_path=root / "missing-teacher.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "dense_residual_control_baseline_failed_closed")
            self.assertIsNone(summary["selected_next_action"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_dense_rank_norm_interference_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common.json"
            selector = root / "selector.json"
            teacher = root / "teacher.json"
            review = root / "review.md"
            common.write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "decision": "acsr_common_causal_residual_benchmark_failed_gate",
                        "claim_status": "sparse_support_specific_effect_not_separated_from_common_dense_controls",
                        "arm_metrics": [
                            {
                                "arm": "sparse_contextual_topk2",
                                "heldout_delta_vs_base_ce": -0.31,
                            },
                            {
                                "arm": "rank_flop_matched_causal_dense",
                                "heldout_delta_vs_base_ce": -0.38,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            selector.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "acsr_post_negative_branch_selected",
                        "selected_next_action": "retire_acsr_promotion_in_favor_of_dense_residual_controls",
                        "next_step": "write the next experiment against dense residual controls rather than ACSR/default-router promotion",
                        "claim_statuses": {"dense_residual_controls": "active_comparison_baseline"},
                    }
                ),
                encoding="utf-8",
            )
            teacher.write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                        "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
                        "dense_teacher_ce_loss": 0.26,
                        "variant_rows": [
                            {"variant": "acsr_predicted_future_support", "ce_loss": 2.83},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Freeze ACSR promotion/GPU work.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_residual_control_baseline_report(
                common_benchmark_path=common,
                branch_selector_path=selector,
                dense_teacher_path=teacher,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], SELECTED_STEP)
            self.assertEqual(summary["claim_statuses"]["dense_residual_controls"], "active_baseline")
            self.assertEqual(summary["claim_statuses"]["ben_notification"], "required")
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            selected = [row for row in summary["candidate_steps"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
