from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.context_contrastive_core_periphery_probe_design import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    REQUIRED_MECHANISM_FIELDS,
    SELECTED_NEXT_ACTION,
    run_context_contrastive_core_periphery_probe_design,
)


class ContextContrastiveCorePeripheryProbeDesignTest(unittest.TestCase):
    def test_records_ready_local_design_without_gpu_advancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector = root / "selector.json"
            core = root / "core.json"
            low_churn = root / "low_churn.json"
            review = root / "latest-review.md"
            _write_json(
                selector,
                {
                    "status": "pass",
                    "decision": "post_low_churn_mlp_branch_selected",
                    "claim_status": "context_contrastive_core_periphery_design_selected_no_gpu",
                    "selected_next_action": "design_context_contrastive_core_periphery_probe_before_gpu",
                },
            )
            _write_json(
                core,
                {
                    "status": "pass",
                    "decision": "core_periphery_negative_evidence_closeout_branch_selected",
                    "claim_status": "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
                    "selected_next_action": "demote_current_core_periphery_mechanism_to_diagnostic_status",
                },
            )
            _write_json(
                low_churn,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_residual_control_pilot_completed",
                    "claim_status": "low_churn_mlp_no_budgeted_advancement_claim",
                    "advancement_row_count": 0,
                    "advance_to_gpu_validation": False,
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

            summary = run_context_contrastive_core_periphery_probe_design(
                branch_selector_path=selector,
                core_closeout_path=core,
                low_churn_pilot_path=low_churn,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "context_contrastive_core_periphery_probe_design_recorded")
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertEqual(summary["claim_status"], "context_contrastive_probe_design_ready_no_gpu")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["failures"], [])
            self.assertTrue(
                set(REQUIRED_MECHANISM_FIELDS).issubset(
                    {row["field"] for row in summary["mechanism_contract"]}
                )
            )
            self.assertTrue(
                set(REQUIRED_CONTROLS).issubset(
                    {row["control"] for row in summary["control_matrix"]}
                )
            )
            self.assertEqual(
                set(REQUIRED_GATES),
                {row["criterion"] for row in summary["gate_criteria"]},
            )
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_context_contrastive_core_periphery_probe_design(
                branch_selector_path=root / "missing-selector.json",
                core_closeout_path=root / "missing-core.json",
                low_churn_pilot_path=root / "missing-low-churn.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
