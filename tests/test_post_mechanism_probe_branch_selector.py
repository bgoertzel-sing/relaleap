from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_mechanism_probe_branch_selector import (
    REQUIRED_ARTIFACTS,
    run_post_mechanism_probe_branch_selector,
)


class PostMechanismProbeBranchSelectorTest(unittest.TestCase):
    def test_closes_local_branches_and_selects_broader_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mechanism = root / "mechanism.json"
            commutator = root / "commutator.json"
            dense_teacher = root / "dense_teacher.json"
            review = root / "latest-review.md"
            _write_json(
                mechanism,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                },
            )
            _write_json(
                commutator,
                {
                    "status": "fail",
                    "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
                    "claim_status": "finite_update_commutator_too_small_for_sparse_mechanism_claim",
                },
            )
            _write_json(
                dense_teacher,
                {
                    "status": "fail",
                    "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                    "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Use broader ACSR gate.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_mechanism_probe_branch_selector(
                mechanism_repeat_path=mechanism,
                commutator_assay_path=commutator,
                dense_teacher_path=dense_teacher,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_mechanism_probe_branches_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(
                summary["selected_next_step"],
                "run_acsr_broader_mechanism_gate_with_existing_local_packets",
            )
            self.assertEqual(
                summary["claim_status"],
                "sparse_retention_commutator_and_dense_teacher_mechanisms_not_established",
            )
            self.assertFalse(summary["direction_shift"]["notify_ben"])
            claim_criteria = [
                row for row in summary["branch_criteria"] if row["severity"] == "claim"
            ]
            self.assertTrue(all(not row["passed"] for row in claim_criteria))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_when_required_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mechanism = root / "mechanism.json"
            commutator = root / "commutator.json"
            _write_json(
                mechanism,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                },
            )
            _write_json(
                commutator,
                {
                    "status": "fail",
                    "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
                    "claim_status": "finite_update_commutator_too_small_for_sparse_mechanism_claim",
                },
            )

            summary = run_post_mechanism_probe_branch_selector(
                mechanism_repeat_path=mechanism,
                commutator_assay_path=commutator,
                dense_teacher_path=root / "missing_dense_teacher.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"], "post_mechanism_probe_branch_selector_failed_closed"
            )
            self.assertEqual(
                summary["selected_next_step"],
                "repair_missing_or_failed_local_mechanism_probe_artifacts",
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
