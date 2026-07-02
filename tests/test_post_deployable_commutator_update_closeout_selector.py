from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_deployable_commutator_update_closeout_selector import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_post_deployable_commutator_update_closeout_selector,
)


class PostDeployableCommutatorUpdateCloseoutSelectorTests(unittest.TestCase):
    def test_closes_commutator_update_line_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)

            summary = run_post_deployable_commutator_update_closeout_selector(
                deployable_probe_path=paths["deployable"],
                order_closeout_path=paths["order"],
                flat_closeout_path=paths["flat"],
                mechanism_repeat_path=paths["mechanism_repeat"],
                mechanism_selector_path=paths["mechanism_selector"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "deployable_commutator_update_line_closed_no_gpu")
            self.assertEqual(summary["claim_status"], "commutator_update_mechanisms_not_established")
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            matrix = {row["criterion"]: row for row in summary["decision_matrix"]}
            self.assertTrue(matrix["candidate_loses_to_dense_controls"]["passed"])
            self.assertTrue(matrix["support_overlap_incomplete"]["passed"])
            selected = [
                row for row in summary["candidate_actions"] if row["disposition"] == "selected"
            ]
            self.assertEqual(len(selected), 1)
            self.assertIn("lost to dense controls", selected[0]["reason"])
            with (root / "out" / "candidate_actions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(CLOSE_ACTION, {row["candidate_action"] for row in rows})
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_deployable_probe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)
            paths["deployable"].unlink()

            summary = run_post_deployable_commutator_update_closeout_selector(
                deployable_probe_path=paths["deployable"],
                order_closeout_path=paths["order"],
                flat_closeout_path=paths["flat"],
                mechanism_repeat_path=paths["mechanism_repeat"],
                mechanism_selector_path=paths["mechanism_selector"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            failure_fields = {row["criterion"] for row in summary["failures"]}
            self.assertIn("required_sources_present", failure_fields)
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "deployable": root / "deployable.json",
        "order": root / "order.json",
        "flat": root / "flat.json",
        "mechanism_repeat": root / "mechanism_repeat.json",
        "mechanism_selector": root / "mechanism_selector.json",
        "review": root / "latest-review.md",
    }
    _write_json(
        paths["deployable"],
        {
            "status": "pass",
            "decision": "deployable_commutator_regularized_sparse_update_probe_recorded_gpu_blocked",
            "claim_status": "deployable_sparse_update_not_established",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
            "control_comparison": [
                {
                    "control": "sparse_unregularized_update",
                    "candidate_minus_control_commutator_anchor_logit_mse": -0.02,
                },
                {
                    "control": "dense_active_matched_update",
                    "candidate_minus_control_commutator_anchor_logit_mse": 0.10,
                },
                {
                    "control": "dense_stored_matched_update",
                    "candidate_minus_control_commutator_anchor_logit_mse": 0.16,
                },
                {
                    "control": "random_support_sparse_update",
                    "candidate_minus_control_commutator_anchor_logit_mse": -0.17,
                },
            ],
            "gate_criteria": [
                {"criterion": "candidate_improves_sparse_commutator", "passed": False},
                {"criterion": "candidate_beats_dense_and_random_controls", "passed": False},
                {"criterion": "support_overlap_bins_populated", "passed": False},
            ],
        },
    )
    _write_json(
        paths["order"],
        {
            "status": "pass",
            "decision": "promoted_topk2_order_averaging_closed_no_gpu",
            "claim_status": "order_averaging_closed_selector_required_for_next_deployable_mechanism",
        },
    )
    _write_json(
        paths["flat"],
        {
            "status": "pass",
            "decision": "flat_value_commutator_mitigation_branch_closed_or_redirected",
            "claim_status": "flat_value_capacity_closed_as_generic_capacity",
        },
    )
    _write_json(
        paths["mechanism_repeat"],
        {
            "status": "pass",
            "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
            "claim_status": "mechanism_factorized_sparse_retention_not_established",
        },
    )
    _write_json(
        paths["mechanism_selector"],
        {
            "status": "pass",
            "decision": "mechanism_factorized_cl_branch_selected",
            "claim_status": "mechanism_factorized_cl_closed_local_inventory_selected_no_gpu",
            "selected_next_action": "pivot_to_commutator_dense_teacher_source_inventory",
        },
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: none",
                "notify_ben: false",
                "recommended_next_action: run local deployable sparse update probe",
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
