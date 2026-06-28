from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_residual_control_baseline_retention_churn_synthesis import (
    REQUIRED_ARTIFACTS,
    run_dense_residual_control_baseline_retention_churn_synthesis,
)


class DenseResidualControlBaselineRetentionChurnSynthesisTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_residual_control_baseline_retention_churn_synthesis(
                closeout_path=root / "missing-closeout.json",
                dense_transfer_path=root / "missing-dense-transfer.json",
                rank_norm_dirs=(root / "missing-rank-norm", root / "missing-rank-norm-seed2"),
                topk1_stability_path=root / "missing-topk1.json",
                acsr_retention_churn_path=root / "missing-acsr.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "dense_retention_churn_synthesis_failed_closed",
            )
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_heldout_context_intervention_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            dense_transfer = root / "dense_transfer.json"
            rank1 = root / "rank1"
            rank2 = root / "rank2"
            topk1 = root / "topk1.json"
            acsr = root / "acsr.json"
            review = root / "review.md"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "acsr_dense_control_retention_churn_closeout_selected",
                    "claim_statuses": {
                        "deployable_support_discovery": "frozen_negative_tiny_headroom_sequence_holdout_and_tiny_commutator"
                    },
                },
            )
            _write_json(
                dense_transfer,
                {
                    "status": "fail",
                    "decision": "acsr_dense_residual_transfer_control_failed_gate",
                    "claim_status": "sparse_transfer_not_separated_from_dense_control",
                },
            )
            _write_rank_norm(rank1, dense=-0.38, sparse=-0.31)
            _write_rank_norm(rank2, dense=-0.42, sparse=-0.32)
            _write_json(
                topk1,
                {
                    "status": "pass",
                    "decision": "active_topk1_retention_churn_stable_across_local_seeds",
                    "packet_count": 2,
                    "aggregates": {
                        "mean_support_churn_advantage": 0.8,
                        "mean_logit_churn_advantage": 0.01,
                        "mean_transfer_improvement_advantage": 0.03,
                    },
                },
            )
            _write_json(
                acsr,
                {
                    "status": "pass",
                    "claim_status": "stronger_non_ce_acsr_control_supported_not_promoted",
                    "comparison_rows": [{"packet": "p1"}],
                    "claim_statuses": {
                        "causal_mechanism_claim": "blocked_pending_heldout_context_interventions"
                    },
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: keep RunPod deferred",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_residual_control_baseline_retention_churn_synthesis(
                closeout_path=closeout,
                dense_transfer_path=dense_transfer,
                rank_norm_dirs=(rank1, rank2),
                topk1_stability_path=topk1,
                acsr_retention_churn_path=acsr,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["selected_next_step"],
                "design_heldout_context_intervention_assay_dense_vs_rank_matched_topk1",
            )
            self.assertEqual(summary["claim_statuses"]["runpod_validation"], "deferred_no_gpu_target")
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))


def _write_rank_norm(path: Path, *, dense: float, sparse: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_residual_rank_norm_interference_supported",
            "claim_status": "causal_dense_control_remains_active_local_baseline",
            "rank_norm_rows": [
                {
                    "arm": "sparse_contextual_topk2",
                    "heldout_delta_vs_base_ce": sparse,
                    "heldout_ce_gain_per_l2": sparse,
                    "heldout_damage_fraction": 0.0,
                },
                {
                    "arm": "sparse_rank_matched_topk1",
                    "heldout_delta_vs_base_ce": sparse + 0.05,
                },
                {
                    "arm": "rank_flop_matched_causal_dense",
                    "heldout_delta_vs_base_ce": dense,
                    "heldout_ce_gain_per_l2": dense,
                    "heldout_damage_fraction": 0.01,
                },
                {
                    "arm": "rank_flop_matched_token_position_dense",
                    "heldout_delta_vs_base_ce": 0.02,
                },
            ],
            "interference_rows": [
                {
                    "row_type": "paired_arms",
                    "split": "heldout",
                    "arm": "rank_flop_matched_causal_dense",
                    "reference_arm": "sparse_contextual_topk2",
                    "mean_delta_advantage_vs_reference": dense - sparse,
                    "left_wins_fraction": 0.5,
                }
            ],
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
