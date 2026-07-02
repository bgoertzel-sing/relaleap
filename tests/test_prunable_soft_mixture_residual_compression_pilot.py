from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.prunable_soft_mixture_residual_compression_pilot import (
    DECISION,
    OBJECTIVES,
    PRUNE_RULES,
    REQUIRED_ARTIFACTS,
    run_prunable_soft_mixture_residual_compression_pilot,
)


class PrunableSoftMixtureResidualCompressionPilotTests(unittest.TestCase):
    def test_runs_local_pilot_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate.json"
            _write_json(pregate, _pregate_payload())

            summary = run_prunable_soft_mixture_residual_compression_pilot(
                pregate_path=pregate,
                out_dir=root / "out",
                seed=11,
                teacher_steps=3,
                value_steps=3,
                control_steps=3,
                component_count=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(set(summary["objectives"]), set(OBJECTIVES))
            self.assertEqual(set(summary["prune_rules"]), set(PRUNE_RULES))
            objective_keys = {
                (row["objective"], row["family"], row["variant"])
                for row in summary["objective_rows"]
            }
            self.assertIn(("ce_only", "same_objective_flat", "norm_matched"), objective_keys)
            self.assertIn(("ce_only", "prunable_soft_mixture_entropy_l1", "norm_matched"), objective_keys)
            self.assertIn(("mse_only", "shuffled_target_soft_mixture_null", "raw"), objective_keys)
            self.assertGreaterEqual(len(summary["mixture_rows"]), len(OBJECTIVES) * 3)
            self.assertGreaterEqual(len(summary["pruning_rows"]), len(OBJECTIVES) * 2 * len(PRUNE_RULES))
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["pregate_selected_pilot"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])
            self.assertEqual(gates["soft_mixture_beats_flat_ce_same_objective"]["gate_type"], "scientific")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_when_pregate_did_not_select_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = _pregate_payload()
            pregate["selected_next_action"] = "different_action"
            _write_json(root / "pregate.json", pregate)

            summary = run_prunable_soft_mixture_residual_compression_pilot(
                pregate_path=root / "pregate.json",
                out_dir=root / "out",
                seed=12,
                teacher_steps=2,
                value_steps=2,
                control_steps=2,
                component_count=4,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["claim_status"], "prunable_soft_mixture_pilot_runtime_failed")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("pregate_selected_pilot", failed)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _pregate_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "prunable_soft_mixture_residual_compression_pregate_recorded",
        "claim_status": "design_only_prunable_soft_mixture_no_gpu_claim",
        "selected_next_action": "implement_prunable_soft_mixture_residual_compression_pilot",
        "selected_next_step": "implement a local prunable soft-mixture residual-compression pilot with same-objective flat/dense controls",
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
