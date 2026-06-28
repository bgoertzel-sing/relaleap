from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_pc_column_design import (
    MANDATORY_CONTROLS,
    MANDATORY_MECHANISM_FIELDS,
    MANDATORY_OBSERVABLES,
    REQUIRED_ARTIFACTS,
    run_core_periphery_pc_column_design,
)


class CorePeripheryPCColumnDesignTest(unittest.TestCase):
    def test_records_fail_closed_contract_ready_for_tiny_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Create and run local fail-closed core/periphery design",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_core_periphery_pc_column_design(
                out_dir=root / "report",
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "ready_for_tiny_pilot")
            self.assertEqual(summary["claim_status"], "design_contract_only_not_training_evidence")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["direction_shift"]["notify_ben"])
            self.assertEqual(summary["failures"], [])
            mechanism_fields = {row["field"] for row in summary["mechanism_fields"]}
            controls = {row["control"] for row in summary["controls"]}
            observables = {row["observable"] for row in summary["observables"]}
            self.assertTrue(set(MANDATORY_MECHANISM_FIELDS).issubset(mechanism_fields))
            self.assertTrue(set(MANDATORY_CONTROLS).issubset(controls))
            self.assertTrue(set(MANDATORY_OBSERVABLES).issubset(observables))
            self.assertIn("dirty_diff_hash", summary)
            self.assertIn("generated_from_head", summary)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_major_or_notify_review_is_recorded_without_blocking_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Major shift",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_core_periphery_pc_column_design(
                out_dir=root / "report",
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertTrue(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue(summary["direction_shift"]["notify_ben"])
            self.assertEqual(summary["direction_shift"]["level"], "major")


if __name__ == "__main__":
    unittest.main()
