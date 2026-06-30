from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_flat_value_branch_selector import (
    DENSE_TEACHER_DISTILLATION_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_post_flat_value_branch_selector,
)


class PostFlatValueBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_post_flat_value_branch_selector(
                learned_router_closeout_path=root / "missing_learned.json",
                flat_value_closeout_path=root / "missing_flat.json",
                core_closeout_path=root / "missing_core.json",
                dense_teacher_control_path=root / "missing_dense.json",
                mechanism_factor_repeat_path=root / "missing_factorized.json",
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

    def test_closed_value_branches_select_dense_teacher_distillation_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned = root / "learned.json"
            flat = root / "flat.json"
            core = root / "core.json"
            dense = root / "dense.json"
            factorized = root / "factorized.json"
            review = root / "latest-review.md"
            _write_json(
                learned,
                {
                    "status": "pass",
                    "decision": "learned_router_sparse_value_branch_closed",
                    "claim_status": "sparse_value_closed_flat_value_diagnostic_selected",
                    "selected_next_action": "design_same_router_flat_value_capacity_diagnostic",
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                flat,
                {
                    "status": "pass",
                    "decision": "flat_value_commutator_mitigation_branch_closed_or_redirected",
                    "claim_status": "flat_value_capacity_closed_as_generic_capacity",
                    "selected_next_action": "close_flat_value_capacity_as_generic_capacity_before_gpu",
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                core,
                {
                    "status": "pass",
                    "decision": "core_periphery_negative_evidence_closeout_branch_selected",
                    "claim_status": "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
                    "selected_next_action": "demote_current_core_periphery_mechanism_to_diagnostic_status",
                    "requires_gpu_now": False,
                },
            )
            _write_json(
                dense,
                {
                    "status": "pass",
                    "decision": "dense_teacher_control_mechanism_assay_blocked",
                    "claim_status": "dense_teacher_acsr_not_supported_against_dense24_mlp_controls",
                    "scientific_gate": "blocked",
                    "selected_next_step": "extend the dense-teacher pilot with matched residual-L2/churn/fingerprint observables",
                    "requires_gpu_now": False,
                },
            )
            _write_json(
                factorized,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                    "selected_next_step": "stop_mechanism_factorized_sparse_retention_branch_and_pivot",
                    "requires_gpu_now": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_flat_value_branch_selector(
                learned_router_closeout_path=learned,
                flat_value_closeout_path=flat,
                core_closeout_path=core,
                dense_teacher_control_path=dense,
                mechanism_factor_repeat_path=factorized,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_flat_value_branch_selected")
            self.assertEqual(summary["selected_next_action"], DENSE_TEACHER_DISTILLATION_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "dense_teacher_residual_distillation_local_comparison_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("dense_teacher_residual_distillation", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
