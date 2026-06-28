from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_core_periphery_contextual_dense_branch_selector import (
    DENSE_MECHANISM_ACTION,
    run_post_core_periphery_contextual_dense_branch_selector,
)


class PostCorePeripheryContextualDenseBranchSelectorTest(unittest.TestCase):
    def test_selects_dense_track_after_sparse_paths_are_demoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            core = root / "core.json"
            acsr = root / "acsr.json"
            dense = root / "dense.json"
            causal = root / "causal.json"
            review = root / "latest-review.md"
            _write_json(
                core,
                {
                    "status": "pass",
                    "selected_next_action": "demote_current_core_periphery_mechanism_to_diagnostic_status",
                    "claim_status": "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
                    "requires_gpu_now": False,
                },
            )
            _write_json(
                acsr,
                {
                    "status": "pass",
                    "selected_next_action": "retire_acsr_promotion_in_favor_of_dense_residual_controls",
                    "claim_statuses": {"dense_residual_controls": "active_comparison_baseline"},
                },
            )
            _write_json(
                dense,
                {
                    "status": "pass",
                    "claim_status": "dense_or_mlp_control_selected_as_primary_mechanism_assay",
                    "primary_arm": "parameter_matched_causal_mlp_control",
                    "primary_family": "parameter_matched_dense_control",
                },
            )
            _write_json(
                causal,
                {
                    "status": "pass",
                    "claim_status": "distilled_causal_router_cross_seed_mechanism_supported_not_promoted",
                    "selected_next_step": "next local mechanism audit",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Test contrastive periphery locally before GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_core_periphery_contextual_dense_branch_selector(
                core_closeout_path=core,
                acsr_selector_path=acsr,
                dense_primary_path=dense,
                causal_router_synthesis_path=causal,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], DENSE_MECHANISM_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["strategy_response"]["deferred_or_rejected"], "deferred")
            self.assertFalse(summary["strategy_response"]["ben_should_be_notified"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], DENSE_MECHANISM_ACTION)
            for artifact in (
                "summary.json",
                "source_rows.csv",
                "candidate_actions.csv",
                "gate_criteria.csv",
                "notes.md",
            ):
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_required_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense = root / "dense.json"
            _write_json(dense, {"status": "pass", "primary_arm": "dense_rank24_best_norm"})

            summary = run_post_core_periphery_contextual_dense_branch_selector(
                core_closeout_path=root / "missing-core.json",
                acsr_selector_path=root / "missing-acsr.json",
                dense_primary_path=dense,
                causal_router_synthesis_path=root / "missing-causal.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], "repair_post_core_periphery_branch_source_artifacts")
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
