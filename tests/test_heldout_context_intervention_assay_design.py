from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.heldout_context_intervention_assay_design import (
    REQUIRED_ARTIFACTS,
    run_heldout_context_intervention_assay_design,
)


class HeldoutContextInterventionAssayDesignTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_heldout_context_intervention_assay_design(
                synthesis_path=root / "missing-synthesis.json",
                topk1_stability_path=root / "missing-topk1.json",
                topk1_metrics_path=root / "missing-topk1.csv",
                rank_norm_dirs=(root / "missing-rank1", root / "missing-rank2"),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "heldout_context_intervention_assay_design_failed_closed",
            )
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_local_dense_topk1_assay_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis.json"
            topk1 = root / "topk1.json"
            metrics = root / "probe_metrics.csv"
            rank1 = root / "rank1"
            rank2 = root / "rank2"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_json(
                synthesis,
                {
                    "status": "pass",
                    "decision": "dense_retention_churn_synthesis_selects_heldout_context_intervention_design",
                    "claim_status": "dense_controls_active_topk1_retention_local_support_acsr_not_promoted",
                    "selected_next_step": "design_heldout_context_intervention_assay_dense_vs_rank_matched_topk1",
                    "claim_statuses": {
                        "dense_residual_controls": "active_baseline",
                        "rank_matched_topk1": "local_retention_churn_support_not_promoted",
                        "acsr_support_discovery": "frozen_negative",
                    },
                },
            )
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
                        "mean_topk1_support_churn": 0.01,
                        "mean_topk2_support_churn": 0.81,
                    },
                },
            )
            metrics.write_text(
                "\n".join(
                    [
                        "packet,probe_dir,topk1_anchor_support_churn_after_transfer,topk2_anchor_support_churn_after_transfer,topk1_transfer_ce_improvement,topk2_transfer_ce_improvement,dense_transfer_ce_improvement",
                        "seed1,probe1,0.01,0.8,0.9,0.85,0.5",
                        "seed2,probe2,0.02,0.7,0.95,0.91,0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_rank_norm(rank1, dense=-0.4, topk1_delta=-0.25, topk2=-0.31)
            _write_rank_norm(rank2, dense=-0.42, topk1_delta=-0.27, topk2=-0.33)
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

            summary = run_heldout_context_intervention_assay_design(
                synthesis_path=synthesis,
                topk1_stability_path=topk1,
                topk1_metrics_path=metrics,
                rank_norm_dirs=(rank1, rank2),
                strategy_review_path=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["selected_next_step"],
                "implement_local_heldout_context_intervention_probe_dense_vs_rank_matched_topk1",
            )
            self.assertEqual(summary["claim_status"], "design_only_dense_topk1_mechanism_not_retested")
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertTrue(
                any(row["component"] == "primary_dense_arm" for row in summary["assay_design"])
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)


def _write_rank_norm(path: Path, *, dense: float, topk1_delta: float, topk2: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_residual_rank_norm_interference_supported",
            "claim_status": "causal_dense_control_remains_active_local_baseline",
        },
    )
    (path / "rank_norm_rows.csv").write_text(
        "\n".join(
            [
                "arm,heldout_delta_vs_base_ce",
                f"sparse_contextual_topk2,{topk2}",
                f"sparse_rank_matched_topk1,{topk1_delta}",
                f"rank_flop_matched_causal_dense,{dense}",
                "rank_flop_matched_token_position_dense,0.01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "interference_rows.csv").write_text(
        "\n".join(
            [
                "row_type,arm,split,reference_arm,mean_delta_advantage_vs_reference,left_wins_fraction",
                f"paired_arms,rank_flop_matched_causal_dense,heldout,sparse_contextual_topk2,{dense - topk2},0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
