from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.multisite_continual_pc_core_periphery_assay_design import (
    REQUIRED_ARMS,
    REQUIRED_ARTIFACTS,
    REQUIRED_OBSERVABLES,
    REQUIRED_SITES,
    run_multisite_continual_pc_core_periphery_assay_design,
)


class MultiSiteContinualPCCorePeripheryAssayDesignTests(unittest.TestCase):
    def test_records_ready_local_design_with_controls_sites_and_observables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            closeout.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "orthogonalized_sparse_core_periphery_branch_closed_or_redirected",
                        "claim_status": "redirect_to_multisite_continual_pc_core_periphery_assay_no_gpu",
                        "selected_next_action": "design_multisite_continual_pc_core_periphery_assay_before_gpu",
                        "advance_to_gpu_validation": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run local trained rows, then multi-site fallback if blocked.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_multisite_continual_pc_core_periphery_assay_design(
                closeout_path=closeout,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["scientific_gate"],
                "ready_for_local_multisite_pc_core_periphery_assay_implementation",
            )
            self.assertEqual(
                summary["claim_status"],
                "design_contract_only_no_training_gpu_or_promotion_evidence",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["failures"], [])
            self.assertTrue(
                set(REQUIRED_ARMS).issubset({row["arm"] for row in summary["assay_arms"]})
            )
            self.assertTrue(
                set(REQUIRED_OBSERVABLES).issubset(
                    {row["observable"] for row in summary["observable_contract"]}
                )
            )
            self.assertTrue(
                set(REQUIRED_SITES).issubset({row["site"] for row in summary["site_schedule"]})
            )
            self.assertTrue(
                any(row["phase_role"] == "revisit_retention_probe" for row in summary["site_schedule"])
            )
            self.assertTrue(all(row["task_id_visible_to_model"] is False for row in summary["site_schedule"]))
            self.assertFalse(summary["strategy_review_handling"]["ben_should_be_notified"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_closeout_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_multisite_continual_pc_core_periphery_assay_design(
                closeout_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertTrue(summary["failures"])
            self.assertTrue(
                any(
                    row["criterion"] == "closeout_selected_multisite_design"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            self.assertTrue((root / "report" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
