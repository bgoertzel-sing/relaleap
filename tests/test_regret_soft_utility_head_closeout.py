from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.regret_soft_utility_head_closeout import (
    LOW_CHURN_MLP_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_regret_soft_utility_head_closeout,
)


class RegretSoftUtilityHeadCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_regret_soft_utility_head_closeout(
                probe_path=root / "missing_probe.json",
                post_flat_selector_path=root / "missing_post_flat.json",
                dense_teacher_closeout_path=root / "missing_dense_closeout.json",
                post_dense_selector_path=root / "missing_post_dense.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_failed_direct_regret_rows_select_low_churn_mlp_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.json"
            post_flat = root / "post_flat.json"
            dense_closeout = root / "dense_closeout.json"
            post_dense = root / "post_dense.json"
            review = root / "latest-review.md"
            _write_json(
                probe,
                {
                    "status": "pass",
                    "decision": "regret_soft_utility_head_probe_gpu_blocked",
                    "claim_status": "regret_soft_utility_head_not_established",
                    "selected_next_action": "close_regret_soft_utility_head_probe_before_gpu",
                    "direct_regret_soft_row_count": 12,
                    "passing_direct_row_count": 0,
                    "proxy_row_count": 6,
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                post_flat,
                {
                    "status": "pass",
                    "decision": "post_flat_value_branch_selected",
                    "claim_status": "dense_teacher_residual_distillation_local_comparison_selected_no_gpu",
                    "selected_next_action": "run_local_dense_teacher_residual_distillation_comparison",
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                dense_closeout,
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_distillation_branch_closed",
                    "claim_status": "dense_teacher_distillation_negative_closeout_no_gpu",
                    "selected_next_action": "close_dense_teacher_residual_distillation_before_gpu",
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                post_dense,
                {
                    "status": "pass",
                    "decision": "post_dense_teacher_control_branch_selected",
                    "claim_status": "mlp_high_power_baseline_needs_low_churn_pregate",
                    "selected_next_action": LOW_CHURN_MLP_ACTION,
                    "requires_gpu_now": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_regret_soft_utility_head_closeout(
                probe_path=probe,
                post_flat_selector_path=post_flat,
                dense_teacher_closeout_path=dense_closeout,
                post_dense_selector_path=post_dense,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "regret_soft_utility_head_branch_closed")
            self.assertEqual(summary["selected_next_action"], LOW_CHURN_MLP_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "regret_soft_closed_low_churn_mlp_pregate_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("Direct regret-soft/listwise utility-head rows failed local gates", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
