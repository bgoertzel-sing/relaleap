from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.pc_amortized_error_branch_selector import (
    COMMUTATOR_MITIGATION_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_pc_amortized_error_branch_selector,
)


class PCAmortizedErrorBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_pc_amortized_error_branch_selector(
                synthetic_summary_path=root / "missing_synthetic.json",
                pc_closeout_rows_path=root / "missing_pc_closeout.csv",
                learned_router_closeout_path=root / "missing_learned.json",
                flat_value_closeout_path=root / "missing_flat.json",
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

    def test_closed_pc_path_selects_flat_value_commutator_mitigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root / "synthetic.json"
            pc_closeout = root / "pc_closeout.csv"
            learned = root / "learned.json"
            flat = root / "flat.json"
            review = root / "latest-review.md"
            _write_json(
                synthetic,
                {
                    "status": "pass",
                    "decision": "synthetic_mechanism_causal_modularity_active_matched_passed_stored_upper_bound_blocks_promotion",
                    "selected_next_step": "repeat hidden support classifier transformer acsr null gate locally",
                    "pc_amortized_error_pregate_closeout_primary_result": {
                        "closeout_status": "closed_current_label_free_amortized_pc_target_path",
                        "current_error_target_path_closed": True,
                        "source_pregate_passes": False,
                        "all_target_nulls_clear": False,
                        "flat_dense_controls_clear": False,
                        "interference_budgets_clear": False,
                        "branch_reopen_requires_new_causal_signal": True,
                        "selected_next_experiment": "return_to_non_pc_sparse_value_or_low_churn_dense_control_branch",
                    },
                },
            )
            pc_closeout.write_text(
                "\n".join(
                    [
                        "closeout_status,current_error_target_path_closed,source_pregate_passes,branch_reopen_requires_new_causal_signal,selected_next_experiment,source_failure_reasons",
                        "closed_current_label_free_amortized_pc_target_path,True,False,True,return_to_non_pc_sparse_value_or_low_churn_dense_control_branch,no_ce_or_stored_gap_signal;finite_update_commutator_budget_failed",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                learned,
                {
                    "status": "pass",
                    "decision": "learned_router_sparse_value_branch_closed",
                    "selected_next_action": "design_same_router_flat_value_capacity_diagnostic",
                    "claim_status": "sparse_value_closed_flat_value_diagnostic_selected",
                },
            )
            _write_json(
                flat,
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_capacity_branch_closed_or_redirected",
                    "selected_next_action": "design_flat_value_finite_update_commutator_mitigation",
                    "claim_status": "flat_value_signal_blocked_by_commutator_mitigation_selected",
                    "evidence": {"commutator_budget_passes": False},
                },
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

            summary = run_pc_amortized_error_branch_selector(
                synthetic_summary_path=synthetic,
                pc_closeout_rows_path=pc_closeout,
                learned_router_closeout_path=learned,
                flat_value_closeout_path=flat,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "pc_amortized_error_branch_selected")
            self.assertEqual(summary["selected_next_action"], COMMUTATOR_MITIGATION_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "pc_closed_flat_value_commutator_mitigation_active",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["evidence"]["pc_current_error_target_path_closed"])
            self.assertFalse(summary["evidence"]["pc_source_pregate_passes"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("flat-value finite-update commutator mitigation", notes)
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
