from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_commutator_mitigation_design import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_ACTION,
    run_same_router_flat_value_commutator_mitigation_design,
)


class SameRouterFlatValueCommutatorMitigationDesignTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_commutator_mitigation_design(
                closeout_path=root / "missing_closeout.json",
                budget_rows_path=root / "missing_budget.csv",
                gate_rows_path=root / "missing_gates.csv",
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

    def test_records_design_when_commutator_is_isolated_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closeout = root / "closeout.json"
            budget = root / "budget.csv"
            gates = root / "gates.csv"
            review = root / "latest-review.md"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_capacity_branch_closed_or_redirected",
                    "claim_status": "flat_value_signal_blocked_by_commutator_mitigation_selected",
                    "selected_next_action": "design_flat_value_finite_update_commutator_mitigation",
                },
            )
            budget.write_text(
                "\n".join(
                    [
                        "budget,nonworse_gate_passes,candidate_value,reference_budget_value",
                        "residual_norm,True,0.07,0.08",
                        "functional_churn,True,0.01,0.02",
                        "finite_update_commutator,False,0.06,0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gates.write_text(
                "\n".join(
                    [
                        "gate,passes",
                        "flat_beats_promoted_sparse,True",
                        "flat_beats_support_policy_nulls,True",
                        "flat_beats_dense_mlp_controls,True",
                        "oracle_regret_strata_available,True",
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

            summary = run_same_router_flat_value_commutator_mitigation_design(
                closeout_path=closeout,
                budget_rows_path=budget,
                gate_rows_path=gates,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "same_router_flat_value_commutator_mitigation_design_recorded",
            )
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "design_only_flat_value_commutator_mitigation_not_yet_evidence",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertAlmostEqual(summary["evidence"]["commutator_ratio_to_reference"], 2.0)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            variants = {row["variant"] for row in summary["mitigation_design"]}
            self.assertIn("flat_value_order_averaged_updates", variants)
            self.assertIn("flat_value_norm_clipped_updates", variants)
            self.assertIn("flat_value_commutator_penalty_probe", variants)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
