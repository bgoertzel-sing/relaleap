from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_order_averaging_deployable_mechanism_selector import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_ACTION,
    run_post_order_averaging_deployable_mechanism_selector,
)


class PostOrderAveragingDeployableMechanismSelectorTest(unittest.TestCase):
    def test_selects_deployable_commutator_regularized_update_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)

            summary = run_post_order_averaging_deployable_mechanism_selector(
                order_closeout_path=paths["order"],
                value_penalty_path=paths["value"],
                flat_closeout_path=paths["flat"],
                multisite_closeout_path=paths["multisite"],
                mechanism_inventory_path=paths["inventory"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "post_order_averaging_deployable_mechanism_selected",
            )
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "deployable_commutator_regularized_sparse_update_pregate_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [
                row for row in summary["candidate_actions"] if row["disposition"] == "selected"
            ]
            self.assertEqual(len(selected), 1)
            self.assertIn("dense/flat/random-support/no-update", selected[0]["reason"])
            matrix = {row["criterion"]: row for row in summary["decision_matrix"]}
            self.assertTrue(matrix["order_averaging_headroom_but_nondeployable"]["passed"])
            self.assertTrue(matrix["simple_value_penalty_already_failed"]["passed"])
            with (root / "out" / "candidate_actions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(SELECTED_ACTION, {row["candidate_action"] for row in rows})
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_order_closeout_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)
            paths["order"].unlink()

            summary = run_post_order_averaging_deployable_mechanism_selector(
                order_closeout_path=paths["order"],
                value_penalty_path=paths["value"],
                flat_closeout_path=paths["flat"],
                multisite_closeout_path=paths["multisite"],
                mechanism_inventory_path=paths["inventory"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            failure_fields = {
                (failure.get("source"), failure.get("field")) for failure in summary["failures"]
            }
            self.assertIn(("order_averaging_closeout", "source_artifact"), failure_fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "order": root / "order.json",
        "value": root / "value.json",
        "flat": root / "flat.json",
        "multisite": root / "multisite.json",
        "inventory": root / "inventory.json",
        "review": root / "latest-review.md",
    }
    _write_json(
        paths["order"],
        {
            "status": "pass",
            "decision": "promoted_topk2_order_averaging_closed_no_gpu",
            "claim_status": "order_averaging_closed_selector_required_for_next_deployable_mechanism",
            "selected_next_action": "close_order_averaging_before_gpu",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "evidence": {
                "order_average_ratio": 0.25,
                "flat_value_order_averaging_control_present": False,
            },
        },
    )
    _write_json(
        paths["value"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "metrics": {"best_penalty_reduction_fraction": 0.23},
        },
    )
    _write_json(
        paths["flat"],
        {
            "status": "pass",
            "decision": "flat_value_commutator_mitigation_branch_closed_or_redirected",
            "selected_next_action": "close_flat_value_capacity_as_generic_capacity_before_gpu",
        },
    )
    _write_json(
        paths["multisite"],
        {
            "status": "pass",
            "decision": "multisite_pc_core_periphery_branch_closed",
            "selected_next_action": "close_multisite_pc_core_periphery_branch_before_gpu",
        },
    )
    _write_json(
        paths["inventory"],
        {
            "status": "pass",
            "decision": "mechanism_source_inventory_recorded",
            "claim_status": "mechanism_inventory_all_local_gates_closed_strategy_needed",
            "selected_next_action": "request_strategy_review_before_new_mechanism_branch",
        },
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: none",
                "notify_ben: false",
                "recommended_next_action: Patch launcher fallback and run local order averaging.",
                "verdict: PAUSE-RECOVER",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
