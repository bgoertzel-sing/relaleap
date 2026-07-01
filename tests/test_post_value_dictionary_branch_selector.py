from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_value_dictionary_branch_selector import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    STRATEGY_REFRESH_ACTION,
    run_post_value_dictionary_branch_selector,
)


class PostValueDictionaryBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_post_value_dictionary_branch_selector(
                value_closeout_path=root / "missing_value.json",
                sparse_closeout_path=root / "missing_sparse.json",
                context_closeout_path=root / "missing_context.json",
                mechanism_inventory_path=root / "missing_inventory.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_closed_value_dictionary_branch_selects_strategy_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = root / "value.json"
            sparse = root / "sparse.json"
            context = root / "context.json"
            inventory = root / "inventory.json"
            review = root / "latest-review.md"
            _write_json(
                value,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_value_dictionary_capacity_rescue_closed",
                    "claim_status": "target_aware_value_dictionary_rescue_closed_no_gpu",
                    "selected_next_action": "select_next_post_value_dictionary_local_branch_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
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
                context,
                {
                    "status": "pass",
                    "decision": "context_contrastive_core_periphery_branch_closed",
                    "claim_status": "context_contrastive_core_periphery_closed_sparse_factorization_ceiling_selected",
                    "selected_next_action": "design_low_churn_mlp_sparse_factorization_ceiling",
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
                        "recommended_next_action: Add the null audit before any close or GPU work.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_value_dictionary_branch_selector(
                value_closeout_path=value,
                sparse_closeout_path=sparse,
                context_closeout_path=context,
                mechanism_inventory_path=inventory,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_value_dictionary_branch_selected")
            self.assertEqual(summary["selected_next_action"], STRATEGY_REFRESH_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "post_value_dictionary_all_local_gates_closed_strategy_refresh_selected",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            rejected = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
                if row["disposition"] == "rejected"
            }
            self.assertEqual(rejected["run_runpod_value_dictionary_or_sparse_validation"], "rejected")
            self.assertEqual(rejected["extend_target_aware_value_dictionary_rescue"], "rejected")
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn(STRATEGY_REFRESH_ACTION, notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
