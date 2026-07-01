from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mechanism_branch_inventory import (
    DENSE_TEACHER_COLUMNABILITY_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_mechanism_branch_inventory,
)


class MechanismBranchInventoryTest(unittest.TestCase):
    def test_selects_dense_teacher_columnability_after_closed_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_closed_branch_sources(root)
            review = root / "latest-review.md"
            _write_major_review(review)

            summary = run_mechanism_branch_inventory(
                context_closeout_path=paths["context"],
                sparse_factor_closeout_path=paths["sparse"],
                value_dictionary_closeout_path=paths["value"],
                post_value_selector_path=paths["post_value"],
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], DENSE_TEACHER_COLUMNABILITY_ACTION)
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertTrue(summary["direction_shift_recorded"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            actions = {row["candidate_action"]: row["disposition"] for row in summary["candidate_actions"]}
            self.assertEqual(actions[DENSE_TEACHER_COLUMNABILITY_ACTION], "selected")
            self.assertEqual(actions["run_gpu_validation_now"], "rejected")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = root / "latest-review.md"
            _write_major_review(review)

            summary = run_mechanism_branch_inventory(
                context_closeout_path=root / "missing_context.json",
                sparse_factor_closeout_path=root / "missing_sparse.json",
                value_dictionary_closeout_path=root / "missing_value.json",
                post_value_selector_path=root / "missing_post_value.json",
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_contradictory_transition_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_closed_branch_sources(root)
            review = root / "latest-review.md"
            _write_major_review(review)
            payload = json.loads(paths["sparse"].read_text(encoding="utf-8"))
            payload["selected_next_action"] = "run_runpod_sparse_factorization_validation"
            paths["sparse"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            summary = run_mechanism_branch_inventory(
                context_closeout_path=paths["context"],
                sparse_factor_closeout_path=paths["sparse"],
                value_dictionary_closeout_path=paths["value"],
                post_value_selector_path=paths["post_value"],
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("closed_branch_transitions_match", failed)
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)


def _write_closed_branch_sources(root: Path) -> dict[str, Path]:
    paths = {
        "context": root / "context.json",
        "sparse": root / "sparse.json",
        "value": root / "value.json",
        "post_value": root / "post_value.json",
    }
    _write_summary(
        paths["context"],
        "context_contrastive_core_periphery_branch_closed",
        "context_contrastive_core_periphery_closed_sparse_factorization_ceiling_selected",
        "design_low_churn_mlp_sparse_factorization_ceiling",
    )
    _write_summary(
        paths["sparse"],
        "low_churn_mlp_sparse_factorization_vector_centroid_ceiling_closed",
        "current_sparse_factorization_ceiling_closed_value_dictionary_rescue_selected",
        "design_value_dictionary_capacity_rescue_before_gpu",
    )
    _write_summary(
        paths["value"],
        "low_churn_mlp_value_dictionary_capacity_rescue_closed",
        "target_aware_value_dictionary_rescue_closed_no_gpu",
        "select_next_post_value_dictionary_local_branch_before_gpu",
    )
    _write_summary(
        paths["post_value"],
        "post_value_dictionary_branch_selected",
        "post_value_dictionary_all_local_gates_closed_strategy_refresh_selected",
        "request_strategy_review_before_new_post_value_dictionary_branch",
    )
    return paths


def _write_summary(path: Path, decision: str, claim_status: str, selected_next_action: str) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": decision,
                "claim_status": claim_status,
                "selected_next_action": selected_next_action,
                "requires_gpu_now": False,
                "promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_major_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: major",
                "notify_ben: true",
                "recommended_next_action: Stop reopening ACSR/support-imitation or MLP-selector branches; recover the selector state and start a local dense-teacher columnability/continual-interference pregate with matched nulls.",
                "verdict: PIVOT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
