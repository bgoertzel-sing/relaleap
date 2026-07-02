from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_support_forcing_sparse_redesign_branch_selector import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    STRATEGY_REFRESH_ACTION,
    run_post_support_forcing_sparse_redesign_branch_selector,
)


class PostSupportForcingSparseRedesignBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_post_support_forcing_sparse_redesign_branch_selector(
                support_closeout_path=root / "missing_support.json",
                sparse_factor_closeout_path=root / "missing_sparse.json",
                value_dictionary_closeout_path=root / "missing_value.json",
                post_value_selector_path=root / "missing_post_value.json",
                mechanism_inventory_path=root / "missing_inventory.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["training_executed"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_closed_redesign_chain_selects_strategy_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            sparse = root / "sparse.json"
            value = root / "value.json"
            post_value = root / "post_value.json"
            inventory = root / "inventory.json"
            review = root / "latest-review.md"
            _write_json(
                support,
                {
                    "status": "pass",
                    "decision": "dense_teacher_support_forcing_pruning_branch_closed_no_gpu",
                    "claim_status": "support_forcing_pruning_sparse_specific_claim_not_established",
                    "selected_next_step": (
                        "select a new local sparse value/support redesign branch with stronger flat-value "
                        "controls before any backend validation"
                    ),
                    "requires_gpu_now": False,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "evidence": {
                        "learned_r2": 0.035417,
                        "flat_r2": 0.672258,
                        "oracle_r2": 0.062943,
                        "sparse_specific_gate_passed": False,
                    },
                },
            )
            _write_json(
                sparse,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_sparse_factorization_vector_centroid_ceiling_closed",
                    "claim_status": "current_sparse_factorization_ceiling_closed_value_dictionary_rescue_selected",
                    "selected_next_action": "design_value_dictionary_capacity_rescue_before_gpu",
                    "requires_gpu_now": False,
                },
            )
            _write_json(
                value,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_value_dictionary_capacity_rescue_closed",
                    "claim_status": "target_aware_value_dictionary_rescue_closed_no_gpu",
                    "selected_next_action": "select_next_post_value_dictionary_local_branch_before_gpu",
                    "requires_gpu_now": False,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                post_value,
                {
                    "status": "pass",
                    "decision": "post_value_dictionary_branch_selected",
                    "claim_status": "post_value_dictionary_all_local_gates_closed_strategy_refresh_selected",
                    "selected_next_action": "request_strategy_review_before_new_post_value_dictionary_branch",
                    "requires_gpu_now": False,
                },
            )
            _write_json(
                inventory,
                {
                    "status": "pass",
                    "decision": "mechanism_source_inventory_recorded",
                    "claim_status": "mechanism_inventory_all_local_gates_closed_strategy_needed",
                    "selected_next_action": "request_strategy_review_before_new_mechanism_branch",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Replace selector churn with dense-teacher local controls.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_support_forcing_sparse_redesign_branch_selector(
                support_closeout_path=support,
                sparse_factor_closeout_path=sparse,
                value_dictionary_closeout_path=value,
                post_value_selector_path=post_value,
                mechanism_inventory_path=inventory,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_support_forcing_sparse_redesign_branch_selected")
            self.assertEqual(summary["selected_next_action"], STRATEGY_REFRESH_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "all_current_sparse_value_support_redesigns_closed_strategy_refresh_selected",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            rejected = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
                if row["disposition"] == "rejected"
            }
            self.assertEqual(rejected["launch_gpu_validation_for_sparse_value_support_redesign"], "rejected")
            self.assertEqual(rejected["reopen_low_churn_sparse_factorization_ceiling"], "rejected")
            self.assertEqual(rejected["reopen_target_aware_value_dictionary_rescue"], "rejected")
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn(STRATEGY_REFRESH_ACTION, notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
