from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_dense_rank_norm_synthesis import (
    REQUIRED_ARTIFACTS,
    run_acsr_dense_rank_norm_synthesis,
)


class ACSRinsightDenseRankNormSynthesisTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_dense_rank_norm_synthesis(
                acsr_synthesis_dir=root / "missing_acsr",
                common_benchmark_dir=root / "missing_common",
                dense_matrix_dir=root / "missing_dense",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_dense_rank_norm_synthesis_failed_closed")
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_dense_rank_threshold_blocks_sparse_support_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            acsr = root / "acsr"
            common = root / "common"
            dense = root / "dense"
            for path in (acsr, common, dense):
                path.mkdir()
            (acsr / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "acsr_two_seed_local_synthesis_recorded",
                        "aggregates": {
                            "mean_acsr_minus_causal_ce_loss": -0.16,
                            "mean_acsr_teacher_support_churn": 0.03,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (common / "summary.json").write_text(
                json.dumps({"status": "fail", "decision": "acsr_common_causal_residual_benchmark_failed_gate"})
                + "\n",
                encoding="utf-8",
            )
            (common / "arm_metrics.csv").write_text(
                "arm,heldout_delta_vs_base_ce\n"
                "sparse_contextual_topk2,-0.30\n"
                "rank_flop_matched_causal_dense,-0.37\n",
                encoding="utf-8",
            )
            (dense / "summary.json").write_text(
                json.dumps({"status": "pass", "decision": "dense_rank_norm_matrix_completed"}) + "\n",
                encoding="utf-8",
            )
            (dense / "rank_summary.csv").write_text(
                "rank,best_heldout_delta_vs_base_ce,best_delta_minus_sparse_topk2,beats_sparse_topk2\n"
                "1,-0.04,0.26,False\n"
                "8,-0.19,0.11,False\n"
                "16,-0.31,-0.01,True\n"
                "24,-0.36,-0.06,True\n",
                encoding="utf-8",
            )
            review = root / "review.md"
            review.write_text(
                "strategic_change_level: major\n"
                "notify_ben: true\n"
                "recommended_next_action: create ACSR pilot skeleton\n",
                encoding="utf-8",
            )

            summary = run_acsr_dense_rank_norm_synthesis(
                acsr_synthesis_dir=acsr,
                common_benchmark_dir=common,
                dense_matrix_dir=dense,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls",
            )
            self.assertTrue(summary["strategy_review"]["ben_notification_required"])
            self.assertIn("rank-16/24 dense controls", summary["selected_next_step"])
            self.assertTrue(
                any(row["criterion"] == "dense_high_rank_does_not_beat_sparse" for row in summary["failures"])
            )


if __name__ == "__main__":
    unittest.main()
