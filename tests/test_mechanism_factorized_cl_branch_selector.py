from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mechanism_factorized_cl_branch_selector import (
    PIVOT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_mechanism_factorized_cl_branch_selector,
)


class MechanismFactorizedClBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_mechanism_factorized_cl_branch_selector(
                scale_closeout_path=root / "missing_scale.json",
                mechanism_repeat_path=root / "missing_repeat.json",
                interference_mitigation_path=root / "missing_mitigation.json",
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

    def test_closed_compression_and_nonreplicated_repeat_select_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale = root / "scale.json"
            repeat = root / "repeat.json"
            mitigation = root / "mitigation.json"
            review = root / "latest-review.md"
            _write_json(
                scale,
                {
                    "status": "pass",
                    "decision": "scale_constrained_sparse_residual_compression_closed_no_gpu",
                    "claim_status": "scale_constrained_sparse_residual_compression_retired_before_gpu",
                    "selected_next_action": "redirect_to_mechanism_factorized_continual_learning_local_gate",
                    "advance_to_gpu_validation": False,
                    "strategy_review_handling": "accepted",
                    "evidence": {
                        "ce_gap_sparse_minus_flat": 0.17,
                        "mse_gap_sparse_minus_flat": 0.27,
                    },
                },
            )
            _write_json(
                repeat,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                    "topk2_tradeoff_repeat_status": "not_replicated",
                    "requires_gpu_now": False,
                    "selected_next_step": "stop_mechanism_factorized_sparse_retention_branch_and_pivot_to_commutator_or_dense_teacher_probe",
                    "primary_result": {
                        "topk2_tradeoff_supporting_seed_count": 1,
                        "full_sparse_claim_supporting_seed_count": 0,
                    },
                },
            )
            _write_json(
                mitigation,
                {
                    "status": "pass",
                    "decision": "residual_interference_mitigation_probe_recorded",
                    "claim_status": "support_width_mitigation_partial_candidate_not_promoted",
                    "requires_gpu_now": False,
                    "selected_next_step": "design one sparse target-adaptation rescue",
                    "primary_result": {"topk2_minus_dense_target_ce_delta": 0.4},
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

            summary = run_mechanism_factorized_cl_branch_selector(
                scale_closeout_path=scale,
                mechanism_repeat_path=repeat,
                interference_mitigation_path=mitigation,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "mechanism_factorized_cl_branch_selected")
            self.assertEqual(summary["selected_next_action"], PIVOT_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "mechanism_factorized_cl_closed_local_inventory_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("commutator/dense-teacher source inventory", notes)
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
