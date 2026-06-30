from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_capacity_closeout import (
    COMMUTATOR_MITIGATION_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_same_router_flat_value_capacity_closeout,
)


class SameRouterFlatValueCapacityCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_capacity_closeout(
                diagnostic_path=root / "missing_summary.json",
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

    def test_commutator_budget_failure_selects_local_mitigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostic = root / "summary.json"
            budget = root / "budget_rows.csv"
            gates = root / "gate_rows.csv"
            review = root / "latest-review.md"
            _write_json(
                diagnostic,
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_capacity_diagnostic_gpu_blocked",
                    "claim_status": "flat_value_capacity_blocked_by_controls_or_interference_budgets",
                    "diagnostic_passes": False,
                    "selected_next_action": "close_flat_value_capacity_as_interference_or_generic_capacity_before_gpu",
                    "primary_result": {
                        "flat_beats_promoted_sparse": True,
                        "flat_beats_support_policy_nulls": True,
                        "flat_beats_dense_mlp_controls": True,
                        "interference_budgets_nonworse": False,
                        "oracle_regret_strata_available": True,
                    },
                },
            )
            budget.write_text(
                "\n".join(
                    [
                        "budget,nonworse_gate_passes,candidate_value,reference_budget_value,failure_reason",
                        "residual_norm,True,0.07,0.08,",
                        "functional_churn,True,0.01,0.02,",
                        "finite_update_commutator,False,0.05,0.02,commutator too high",
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
                        "interference_budgets_nonworse,False",
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

            summary = run_same_router_flat_value_capacity_closeout(
                diagnostic_path=diagnostic,
                budget_rows_path=budget,
                gate_rows_path=gates,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "same_router_flat_value_capacity_branch_closed_or_redirected",
            )
            self.assertEqual(summary["selected_next_action"], COMMUTATOR_MITIGATION_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "flat_value_signal_blocked_by_commutator_mitigation_selected",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["evidence"]["flat_beats_support_policy_nulls"])
            self.assertFalse(summary["evidence"]["commutator_budget_passes"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("finite-update commutator mitigation", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
