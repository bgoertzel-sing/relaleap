from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_capacity_diagnostic_design import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_ACTION,
    run_same_router_flat_value_capacity_diagnostic_design,
)


class SameRouterFlatValueCapacityDiagnosticDesignTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_capacity_diagnostic_design(
                closeout_path=root / "missing_closeout.json",
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

    def test_records_flat_value_diagnostic_contract_from_closeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closeout = root / "closeout.json"
            synthetic = root / "synthetic"
            review = root / "latest-review.md"
            synthetic.mkdir()
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "learned_router_sparse_value_branch_closed",
                    "claim_status": "sparse_value_closed_flat_value_diagnostic_selected",
                    "selected_next_action": "design_same_router_flat_value_capacity_diagnostic",
                    "evidence": {"flat_control_ce_gain": -0.024},
                },
            )
            (synthetic / "arm_metrics.csv").write_text(
                "\n".join(
                    [
                        "arm,holdout_ce,residual_l2",
                        "promoted_contextual_topk2,2.70,0.07",
                        "flat_column_value_mlp_topk2,2.66,0.08",
                        "token_position_router_topk2,2.71,0.08",
                        "dense_rank_norm_matched,2.74,0.09",
                        "low_churn_mlp_active_matched,2.73,0.10",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "value_capacity_core_periphery_diagnostic.csv").write_text(
                "\n".join(
                    [
                        "branch,ce_gap_vs_sparse",
                        "active_value_capacity_control,0.04",
                        "stored_value_capacity_upper_bound,0.80",
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
                        "flat_column_value_mlp_topk2,0.03",
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
                        "flat_column_value_mlp_topk2,-0.012",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "residual_budget_accounting.csv").write_text(
                "arm,flop_proxy_per_token\nflat_column_value_mlp_topk2,10\n",
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

            summary = run_same_router_flat_value_capacity_diagnostic_design(
                closeout_path=closeout,
                synthetic_dir=synthetic,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "same_router_flat_value_capacity_diagnostic_design_recorded",
            )
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertLess(summary["evidence"]["flat_minus_sparse_ce"], 0.0)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            roles = {row["diagnostic_role"] for row in summary["diagnostic_design"]}
            self.assertIn("same_router_flat_value_primary", roles)
            self.assertIn("interference_budgets", roles)
            self.assertIn("oracle_regret_strata", roles)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
