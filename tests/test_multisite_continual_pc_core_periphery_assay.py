from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.multisite_continual_pc_core_periphery_assay import (
    CANDIDATE,
    REQUIRED_ARTIFACTS,
    run_multisite_continual_pc_core_periphery_assay,
)
from relaleap.experiments.multisite_continual_pc_core_periphery_assay_design import (
    REQUIRED_ARMS,
    REQUIRED_SITES,
)


class MultiSiteContinualPCCorePeripheryAssayTests(unittest.TestCase):
    def test_records_local_cpu_training_rows_for_required_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design-summary.json"
            _write_design(design)

            summary = run_multisite_continual_pc_core_periphery_assay(
                design_summary_path=design,
                out_dir=root / "assay",
                seed=3,
                steps_per_site=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "multisite_continual_pc_core_periphery_assay_recorded",
            )
            self.assertTrue(summary["training_rows_present"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["sites"], list(REQUIRED_SITES))
            self.assertFalse(summary["task_id_visible_to_model"])
            arms = {row["arm"] for row in summary["arm_metrics"]}
            self.assertTrue(set(REQUIRED_ARMS).issubset(arms))
            candidate = next(row for row in summary["arm_metrics"] if row["arm"] == CANDIDATE)
            self.assertEqual(
                candidate["row_source"],
                "bounded_local_cpu_trained_multisite_synthetic_rule_stream",
            )
            self.assertIn("heldout_ce", candidate)
            self.assertIn("finite_update_commutator", candidate)
            self.assertIn("periphery_first_pruning_delta", candidate)
            hard_gates = [row for row in summary["gate_criteria"] if row["severity"] == "hard"]
            self.assertTrue(hard_gates)
            self.assertTrue(all(row["passed"] for row in hard_gates))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "assay" / artifact).is_file(), artifact)

    def test_missing_design_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_multisite_continual_pc_core_periphery_assay(
                design_summary_path=root / "missing.json",
                out_dir=root / "assay",
                seed=5,
                steps_per_site=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertTrue(
                any(
                    row["criterion"] == "design_contract_passed" and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue((root / "assay" / "summary.json").is_file())


def _write_design(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "multisite_continual_pc_core_periphery_assay_design_recorded",
                "scientific_gate": "ready_for_local_multisite_pc_core_periphery_assay_implementation",
                "advance_to_gpu_validation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
