from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_transfer_objective_probe import (
    REQUIRED_ARTIFACTS,
    run_acsr_transfer_objective_probe,
)


class ACSRTransferObjectiveProbeTest(unittest.TestCase):
    def test_missing_design_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_transfer_objective_probe(
                config_path=root / "missing-config.yaml",
                design_summary=root / "missing-summary.json",
                out_dir=root / "out",
                max_steps=1,
                router_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_transfer_objective_probe_failed_closed")
            self.assertEqual(summary["claim_status"], "transfer_objective_probe_not_run")
            self.assertTrue(
                any(row["criterion"] == "objective_design_present" for row in summary["failures"])
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_failed_design_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "summary.json"
            design.write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "objective_terms": [
                            {"term": "cross_value_partner_support_ce", "weight": 1.0}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_transfer_objective_probe(
                config_path=root / "missing-config.yaml",
                design_summary=design,
                out_dir=root / "out",
                max_steps=1,
                router_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(row["criterion"] == "objective_design_passed" for row in summary["failures"])
            )
            self.assertEqual(summary["arm_count"], 0)


if __name__ == "__main__":
    unittest.main()
