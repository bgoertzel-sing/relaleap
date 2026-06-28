from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_pc_column_pilot import (
    REQUIRED_ARTIFACTS,
    REQUIRED_VARIANTS,
    run_core_periphery_pc_column_pilot,
)


class CorePeripheryPCColumnPilotTest(unittest.TestCase):
    def test_pilot_writes_required_controls_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "scientific_gate": "ready_for_tiny_pilot",
                        "claim_status": "design_contract_only_not_training_evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_core_periphery_pc_column_pilot(
                contract_path=contract,
                out_dir=root / "pilot",
                seed=3,
                steps_per_task=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertIn(
                summary["scientific_gate"],
                {"blocked", "ready_for_repeat_only"},
            )
            variants = {row["variant"] for row in summary["variant_metrics"]}
            self.assertTrue(set(REQUIRED_VARIANTS).issubset(variants))
            self.assertGreaterEqual(
                len(summary["intervention_fingerprints"]),
                len(REQUIRED_VARIANTS) * 2,
            )
            self.assertTrue(
                any(row["criterion"] == "required_controls_present" for row in summary["gate_criteria"])
            )
            self.assertIn("core_periphery_update_norm_ratio", summary["primary_result"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "pilot" / artifact).is_file(), artifact)

    def test_pilot_fails_closed_without_ready_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps({"status": "pass", "scientific_gate": "blocked"}) + "\n",
                encoding="utf-8",
            )

            summary = run_core_periphery_pc_column_pilot(
                contract_path=contract,
                out_dir=root / "pilot",
                seed=3,
                steps_per_task=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["claim_status"], "design_contract_missing_or_not_ready")
            self.assertTrue(
                any(
                    row["criterion"] == "contract_ready_for_tiny_pilot"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )


if __name__ == "__main__":
    unittest.main()
