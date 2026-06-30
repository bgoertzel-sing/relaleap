from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_capacity_diagnostic import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_same_router_flat_value_capacity_diagnostic,
)


class SameRouterFlatValueCapacityDiagnosticTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_capacity_diagnostic(
                design_path=root / "missing_design.json",
                synthetic_dir=root / "missing_synthetic",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_flat_value_diagnostic_blocks_gpu_on_interference_budget_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root / "synthetic"
            synthetic.mkdir()
            design = root / "design.json"
            review = root / "latest-review.md"
            _write_json(
                design,
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_capacity_diagnostic_design_recorded",
                    "selected_next_action": "implement_same_router_flat_value_capacity_diagnostic_locally",
                },
            )
            (synthetic / "arm_metrics.csv").write_text(
                "\n".join(
                    [
                        "arm,holdout_ce,residual_l2,active_parameters_proxy,stored_parameters",
                        "promoted_contextual_topk2,2.70,0.07,96,7174",
                        "flat_column_value_mlp_topk2,2.66,0.08,288,9007",
                        "token_position_router_topk2,2.71,0.08,96,7174",
                        "random_support_topk2,2.76,0.08,96,7174",
                        "fixed_support_topk2,2.73,0.08,96,7174",
                        "dense_rank_norm_matched,2.74,0.09,96,7174",
                        "low_churn_mlp_active_matched,2.72,0.12,96,7174",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "residual_budget_accounting.csv").write_text(
                "\n".join(
                    [
                        "arm,flop_proxy_per_token",
                        "promoted_contextual_topk2,192",
                        "flat_column_value_mlp_topk2,576",
                        "token_position_router_topk2,192",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "commutator_rows.csv").write_text(
                "\n".join(
                    [
                        "arm,finite_update_commutator_l2",
                        "promoted_contextual_topk2,0.02",
                        "token_position_router_topk2,0.02",
                        "flat_column_value_mlp_topk2,0.05",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "forgetting_rows.csv").write_text(
                "\n".join(
                    [
                        "arm,functional_churn",
                        "promoted_contextual_topk2,0.01",
                        "token_position_router_topk2,0.01",
                        "flat_column_value_mlp_topk2,0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "router_regret_ceiling_budget.csv").write_text(
                "\n".join(
                    [
                        "arm,learned_holdout_ce,oracle_support_ce_ceiling,mean_oracle_regret",
                        "promoted_contextual_topk2,2.70,2.66,0.04",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "router_value_regret_decomposition.csv").write_text(
                "\n".join(
                    [
                        "arm,latent_rule,mean_learned_ce_loss,mean_oracle_ce_loss,mean_oracle_regret",
                        "promoted_contextual_topk2,all,2.70,2.66,0.04",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_same_router_flat_value_capacity_diagnostic(
                design_path=design,
                synthetic_dir=synthetic,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "same_router_flat_value_capacity_diagnostic_gpu_blocked",
            )
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["primary_result"]["flat_beats_promoted_sparse"])
            self.assertTrue(summary["primary_result"]["flat_beats_support_policy_nulls"])
            self.assertFalse(summary["primary_result"]["interference_budgets_nonworse"])

            with (root / "out" / "budget_rows.csv").open(newline="", encoding="utf-8") as handle:
                budget_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["budget"] for row in budget_rows},
                {"residual_norm", "functional_churn", "finite_update_commutator"},
            )
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
