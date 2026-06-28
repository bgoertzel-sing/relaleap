from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.heldout_context_intervention_probe import (
    REQUIRED_ARTIFACTS,
    run_heldout_context_intervention_probe,
)


class HeldoutContextInterventionProbeTest(unittest.TestCase):
    def test_missing_required_nulls_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design.json"
            metrics = root / "probe_metrics.csv"
            rank = root / "rank"
            review = root / "latest-review.md"
            _write_design(design)
            _write_topk1_metrics(metrics)
            _write_rank_norm_packet(rank, include_required_nulls=False, dense_delta=-0.4, topk1_delta=-0.2)
            _write_review(review)

            summary = run_heldout_context_intervention_probe(
                design_path=design,
                topk1_metrics_path=metrics,
                rank_norm_dirs=(rank,),
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "heldout_context_intervention_probe_failed_closed")
            self.assertTrue(
                any(row["criterion"] == "required_arms_and_nulls_present" for row in summary["failures"])
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_complete_probe_contract_records_primary_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design.json"
            metrics = root / "probe_metrics.csv"
            rank1 = root / "rank1"
            rank2 = root / "rank2"
            review = root / "latest-review.md"
            _write_design(design)
            _write_topk1_metrics(metrics)
            _write_rank_norm_packet(rank1, include_required_nulls=True, dense_delta=-0.2, topk1_delta=-0.35)
            _write_rank_norm_packet(rank2, include_required_nulls=True, dense_delta=-0.21, topk1_delta=-0.36)
            _write_review(review)

            summary = run_heldout_context_intervention_probe(
                design_path=design,
                topk1_metrics_path=metrics,
                rank_norm_dirs=(rank1, rank2),
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "heldout_context_intervention_probe_passed")
            self.assertEqual(
                summary["claim_status"],
                "rank_matched_topk1_reopened_sparse_mechanism_claim",
            )
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertLess(summary["primary_result"]["mean_dense_minus_topk1_heldout_delta"], 0.2)


def _write_design(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "heldout_context_intervention_assay_design_recorded",
            "selected_next_step": "implement_local_heldout_context_intervention_probe_dense_vs_rank_matched_topk1",
        },
    )


def _write_topk1_metrics(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "packet,topk1_support_churn_lower_than_topk2,topk1_logit_churn_not_higher_than_topk2,topk1_transfer_improvement_at_least_topk2,topk1_anchor_support_churn_after_transfer,topk1_anchor_logit_mse_drift,topk1_transfer_ce_improvement",
                "seed1,True,True,True,0.01,0.1,0.9",
                "seed2,True,True,True,0.02,0.11,0.91",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rank_norm_packet(
    path: Path,
    *,
    include_required_nulls: bool,
    dense_delta: float,
    topk1_delta: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_residual_rank_norm_interference_supported",
        },
    )
    rows = [
        ("sparse_contextual_topk2", "sparse", -0.3, 1.0, 96, 96),
        ("sparse_rank_matched_topk1", "sparse", topk1_delta, 1.0, 96, 96),
        ("rank_flop_matched_causal_dense", "dense", dense_delta, 1.0, 96, 96),
        ("rank_flop_matched_token_position_dense", "dense", 0.02, 1.0, 96, 96),
    ]
    if include_required_nulls:
        rows.extend(
            [
                ("rank_flop_matched_shuffled_context_dense", "dense", 0.01, 1.0, 96, 96),
                ("rank_flop_matched_ablated_context_dense", "dense", 0.03, 1.0, 96, 96),
                ("sparse_frequency_matched_random_topk1", "sparse", -0.05, 1.0, 96, 96),
            ]
        )
    (path / "rank_norm_rows.csv").write_text(
        "arm,family,heldout_delta_vs_base_ce,heldout_residual_update_l2,active_params_proxy,flops_proxy\n"
        + "\n".join(f"{arm},{family},{delta},{l2},{params},{flops}" for arm, family, delta, l2, params, flops in rows)
        + "\n",
        encoding="utf-8",
    )
    (path / "interference_rows.csv").write_text(
        "row_type,arm,split,damage_fraction,improvement_fraction\n"
        + "\n".join(f"arm_split,{arm},heldout,0.0,1.0" for arm, *_ in rows)
        + "\n",
        encoding="utf-8",
    )


def _write_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: require null controls before evidence",
                "verdict: FIX",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
