from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_low_churn_mlp_branch_selector import (
    CONTEXT_CONTRASTIVE_CORE_PERIPHERY_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_post_low_churn_mlp_branch_selector,
)


class PostLowChurnMlpBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_post_low_churn_mlp_branch_selector(
                low_churn_pilot_path=root / "missing_low_churn.json",
                acsr_closeout_path=root / "missing_acsr.json",
                core_closeout_path=root / "missing_core.json",
                post_core_selector_path=root / "missing_post_core.json",
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

    def test_blocked_low_churn_and_demoted_sparse_select_core_periphery_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low_churn = root / "low_churn.json"
            acsr = root / "acsr.json"
            core = root / "core.json"
            post_core = root / "post_core.json"
            review = root / "latest-review.md"
            _write_json(
                low_churn,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_residual_control_pilot_completed",
                    "claim_status": "low_churn_mlp_no_budgeted_advancement_claim",
                    "scientific_gate": "blocked",
                    "advancement_row_count": 0,
                    "selected_next_action": "return_to_sparse_core_periphery_mechanism_work",
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                acsr,
                {
                    "status": "pass",
                    "decision": "acsr_negative_evidence_closeout_branch_selected",
                    "claim_status": "acsr_promotion_path_demoted_to_diagnostic_no_default_change",
                    "selected_next_action": "demote_acsr_to_diagnostic_status",
                    "requires_gpu_now": False,
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
                post_core,
                {
                    "status": "pass",
                    "decision": "post_core_periphery_contextual_dense_branch_selected",
                    "claim_status": "dense_mlp_mechanism_track_selected_no_gpu_or_default_change",
                    "selected_next_action": "continue_dense_mlp_mechanism_track_with_causal_router_diagnostics",
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

            summary = run_post_low_churn_mlp_branch_selector(
                low_churn_pilot_path=low_churn,
                acsr_closeout_path=acsr,
                core_closeout_path=core,
                post_core_selector_path=post_core,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "post_low_churn_mlp_branch_selected")
            self.assertEqual(
                summary["selected_next_action"],
                CONTEXT_CONTRASTIVE_CORE_PERIPHERY_ACTION,
            )
            self.assertEqual(
                summary["claim_status"],
                "context_contrastive_core_periphery_design_selected_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("context-contrastive core/periphery", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
