from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.commutator_dense_teacher_source_inventory import (
    ORDER_AVERAGING_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_commutator_dense_teacher_source_inventory,
)


class CommutatorDenseTeacherSourceInventoryTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_commutator_dense_teacher_source_inventory(
                branch_selector_path=root / "missing_branch.json",
                dense_mlp_synthesis_path=root / "missing_dense.json",
                dense_teacher_distillation_closeout_path=root / "missing_distill.json",
                pair_composer_closeout_path=root / "missing_pair.json",
                flat_value_commutator_closeout_path=root / "missing_flat_comm.json",
                topk2_post_finite_update_closeout_path=root / "missing_post_finite.json",
                topk2_mitigation_selector_path=root / "missing_mitigation.json",
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

    def test_selects_order_averaging_when_dense_teacher_closed_and_commutator_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branch = root / "branch.json"
            dense = root / "dense.json"
            distill = root / "distill.json"
            pair = root / "pair.json"
            flat_comm = root / "flat_comm.json"
            post_finite = root / "post_finite.json"
            mitigation = root / "mitigation.json"
            review = root / "latest-review.md"
            _write_json(
                branch,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_branch_selected",
                    "claim_status": "mechanism_factorized_cl_closed_local_inventory_selected_no_gpu",
                    "selected_next_action": "pivot_to_commutator_dense_teacher_source_inventory",
                    "selected_next_step": "add a local commutator/dense-teacher source inventory before any new training or GPU validation",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                dense,
                {
                    "status": "pass",
                    "decision": "dense_mlp_dominance_synthesized_sparse_interference_pregate_selected",
                    "claim_status": "pair_composer_closed_dense_mlp_dominance_sparse_interference_pregate_selected_no_gpu",
                    "selected_next_action": "design_orthogonalized_sparse_additive_core_periphery_interference_pregate",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                distill,
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_distillation_branch_closed",
                    "claim_status": "dense_teacher_distillation_negative_closeout_no_gpu",
                    "selected_next_action": "close_dense_teacher_residual_distillation_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                pair,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_branch_closed",
                    "claim_status": "pair_composer_closed_dense_mlp_controls_dominate_no_gpu",
                    "selected_next_action": "redirect_from_pair_composer_to_dense_mlp_control_synthesis",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                flat_comm,
                {
                    "status": "pass",
                    "decision": "flat_value_commutator_mitigation_branch_closed_or_redirected",
                    "claim_status": "flat_value_capacity_closed_as_generic_capacity",
                    "selected_next_action": "close_flat_value_capacity_as_generic_capacity_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_json(
                post_finite,
                {
                    "status": "pass",
                    "decision": "post_finite_update_closeout_selected",
                    "selected_next_action": "extend_causal_fingerprint_control_matrix_with_finite_update_fields",
                },
            )
            _write_json(
                mitigation,
                {
                    "status": "pass",
                    "decision": "promoted_topk2_mitigation_branch_selected",
                    "selected_next_action": "explicit_order_averaging_mitigation_probe",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement a local executable scale-constrained sparse residual-compression pilot with norm-matched flat/dense controls and fail-closed mechanism gates before any GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_commutator_dense_teacher_source_inventory(
                branch_selector_path=branch,
                dense_mlp_synthesis_path=dense,
                dense_teacher_distillation_closeout_path=distill,
                pair_composer_closeout_path=pair,
                flat_value_commutator_closeout_path=flat_comm,
                topk2_post_finite_update_closeout_path=post_finite,
                topk2_mitigation_selector_path=mitigation,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "commutator_dense_teacher_source_inventory_recorded")
            self.assertEqual(summary["selected_next_action"], ORDER_AVERAGING_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "commutator_inventory_selects_order_averaging_probe_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            inventory = {row["criterion"]: row for row in summary["inventory_rows"]}
            self.assertTrue(inventory["dense_teacher_distillation_closed_negative"]["passed"])
            self.assertTrue(inventory["explicit_order_averaging_selected_but_not_yet_run_here"]["passed"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("explicit order-averaging mitigation probe", notes)
            self.assertIn("GPU validation remains blocked", notes)
            with (root / "out" / "candidate_actions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(ORDER_AVERAGING_ACTION, {row["candidate_action"] for row in rows})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
