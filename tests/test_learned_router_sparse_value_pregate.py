from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.learned_router_sparse_value_pregate import (
    NEXT_CLOSEOUT_ACTION,
    REPAIR_SOURCES_ACTION,
    REQUIRED_ARTIFACTS,
    run_learned_router_sparse_value_pregate,
)


class LearnedRouterSparseValuePregateTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_learned_router_sparse_value_pregate(
                branch_selector_path=root / "missing_selector.json",
                synthetic_dir=root / "missing_synthetic",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_SOURCES_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_learned_router_branch_fails_closed_when_flat_control_is_stronger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selector = root / "selector.json"
            synthetic = root / "synthetic"
            synthetic.mkdir()
            review = root / "latest-review.md"
            _write_json(
                selector,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_branch_selected",
                    "selected_next_action": "return_to_learned_router_non_pc_sparse_value_branch",
                    "claim_status": "direct_hidden_classifier_closed_learned_router_branch_active",
                },
            )
            (synthetic / "arm_metrics.csv").write_text(
                "\n".join(
                    [
                        "arm,holdout_ce,residual_l2,control_budget_role",
                        "promoted_contextual_topk2,2.70,0.07,sparse_or_null_or_base_reference",
                        "token_position_router_topk2,2.71,0.08,sparse_or_null_or_base_reference",
                        "random_support_topk2,2.76,0.08,sparse_or_null_or_base_reference",
                        "fixed_support_topk2,2.73,0.08,sparse_or_null_or_base_reference",
                        "intervention_trained_sparse_topk2,2.71,0.07,sparse_or_null_or_base_reference",
                        "flat_column_value_mlp_topk2,2.66,0.07,flat_same_router_value_capacity_control",
                        "dense_rank_norm_matched,2.75,0.08,active_proxy_matched_dense_mlp_control",
                        "low_churn_mlp_active_matched,2.74,0.12,active_proxy_matched_dense_mlp_control",
                        "dense_stored_parameter_matched,2.00,1.60,stored_parameter_matched_dense_mlp_upper_bound",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "residual_budget_accounting.csv").write_text(
                "\n".join(
                    [
                        "arm,flop_proxy_per_token",
                        "promoted_contextual_topk2,10",
                        "token_position_router_topk2,10",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_metric_rows(synthetic / "commutator_rows.csv", "finite_update_commutator_l2")
            _write_metric_rows(synthetic / "forgetting_rows.csv", "functional_churn")
            review.write_text(
                "strategic_change_level: minor\nnotify_ben: false\nverdict: FIX\n",
                encoding="utf-8",
            )

            summary = run_learned_router_sparse_value_pregate(
                branch_selector_path=selector,
                synthetic_dir=synthetic,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "learned_router_sparse_value_pregate_local_gpu_blocked")
            self.assertEqual(summary["selected_next_action"], NEXT_CLOSEOUT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            primary = summary["pregate_primary_result"]
            self.assertTrue(primary["branch_selector_ok"])
            self.assertFalse(primary["flat_control_ok"])
            self.assertTrue(primary["stored_upper_bound_blocks_promotion"])
            self.assertFalse(primary["pregate_passes"])
            self.assertIn("same_router_flat_value_control_stronger", primary["failure_reasons"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("RunPod and promotion remain blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_metric_rows(path: Path, key: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"arm,{key}",
                "promoted_contextual_topk2,0.01",
                "token_position_router_topk2,0.011",
                "dense_rank_norm_matched,0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
