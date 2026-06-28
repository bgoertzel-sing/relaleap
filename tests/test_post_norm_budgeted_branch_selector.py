from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_norm_budgeted_branch_selector import (
    DENSE_TEACHER_ACTION,
    REPAIR_SOURCES_ACTION,
    REQUIRED_ARTIFACTS,
    STRONG_COMMUTATOR_ACTION,
    TASK_FREE_CL_ACTION,
    run_post_norm_budgeted_branch_selector,
)


class PostNormBudgetedBranchSelectorTest(unittest.TestCase):
    def test_selects_dense_pivot_when_sparse_branches_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            norm = root / "norm.json"
            cl = root / "cl.json"
            commutator = root / "commutator.json"
            review = root / "latest-review.md"
            _write_json(
                norm,
                {
                    "status": "pass",
                    "decision": "norm_budgeted_churn_strata_synthesis_completed",
                    "claim_status": "local_strata_signal_does_not_warrant_gpu_repeat",
                    "runpod_repeat_warranted": False,
                    "interpretation": "sparse branch stopped",
                },
            )
            _write_json(
                cl,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                    "topk2_tradeoff_repeat_status": "not_replicated",
                    "selected_next_step": "pivot_to_commutator_or_dense_teacher_probe",
                },
            )
            _write_json(
                commutator,
                {
                    "status": "fail",
                    "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
                    "claim_status": "finite_update_commutator_too_small_for_sparse_mechanism_claim",
                    "metrics": {"sparse_mean_logit_mse": 0.0006, "dense_mean_logit_mse": 0.008},
                },
            )
            review.write_text(
                "strategic_change_level: minor\nnotify_ben: false\nrecommended_next_action: prefer dense controls\nverdict: FIX\n",
                encoding="utf-8",
            )

            summary = run_post_norm_budgeted_branch_selector(
                norm_synthesis_path=norm,
                cl_repeat_path=cl,
                commutator_path=commutator,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], DENSE_TEACHER_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(
                summary["claim_status"],
                "sparse_branches_locally_blocked_dense_control_pivot_selected",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_task_free_cl_when_sparse_signal_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            norm = root / "norm.json"
            cl = root / "cl.json"
            commutator = root / "commutator.json"
            _write_json(norm, {"status": "pass", "runpod_repeat_warranted": True})
            _write_json(cl, {"status": "pass", "topk2_tradeoff_repeat_status": "not_replicated"})
            _write_json(commutator, {"status": "fail", "metrics": {"sparse_mean_logit_mse": 0.0006}})

            summary = run_post_norm_budgeted_branch_selector(
                norm_synthesis_path=norm,
                cl_repeat_path=cl,
                commutator_path=commutator,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], TASK_FREE_CL_ACTION)

    def test_selects_stronger_commutator_when_material_commutator_is_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            norm = root / "norm.json"
            cl = root / "cl.json"
            commutator = root / "commutator.json"
            _write_json(norm, {"status": "pass", "runpod_repeat_warranted": False})
            _write_json(cl, {"status": "pass", "topk2_tradeoff_repeat_status": "not_replicated"})
            _write_json(commutator, {"status": "fail", "metrics": {"sparse_mean_logit_mse": 0.006}})

            summary = run_post_norm_budgeted_branch_selector(
                norm_synthesis_path=norm,
                cl_repeat_path=cl,
                commutator_path=commutator,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], STRONG_COMMUTATOR_ACTION)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_post_norm_budgeted_branch_selector(
                norm_synthesis_path=root / "missing-norm.json",
                cl_repeat_path=root / "missing-cl.json",
                commutator_path=root / "missing-commutator.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_SOURCES_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
