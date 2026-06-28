from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_pc_column_synthesis import (
    REQUIRED_ARTIFACTS,
    run_core_periphery_pc_column_synthesis,
)


class CorePeripheryPCColumnSynthesisTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_core_periphery_pc_column_synthesis(
                pilot_dirs=(root / "missing_default", root / "missing_seed11"),
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(summary["decision"], "core_periphery_pc_column_synthesis_failed_closed")
            self.assertTrue(summary["failures"])
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_two_passing_repeats_select_non_synthetic_design_not_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            default = root / "default"
            seed11 = root / "seed11"
            _write_source(default, seed=7, dense_delta=-0.0049, mlp_delta=-0.0025, prune_delta=0.0015)
            _write_source(seed11, seed=11, dense_delta=-0.0050, mlp_delta=-0.0026, prune_delta=0.0017)

            summary = run_core_periphery_pc_column_synthesis(
                pilot_dirs=(default, seed11),
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "core_periphery_pc_column_local_repeats_supported")
            self.assertEqual(summary["scientific_gate"], "ready_for_non_synthetic_pilot_design")
            self.assertEqual(summary["claim_status"], "synthetic_local_repeat_only_not_gpu_or_promotion_evidence")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["failures"], [])
            self.assertIn("non-synthetic command-driven", summary["selected_next_step"])
            self.assertIn("mean_core_minus_dense_anchor_mse_drift", summary["aggregate_metrics"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_claim_gate_blocks_when_repeat_fails_dense_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            default = root / "default"
            seed11 = root / "seed11"
            _write_source(default, seed=7, dense_delta=-0.0049, mlp_delta=-0.0025, prune_delta=0.0015)
            _write_source(seed11, seed=11, dense_delta=0.0002, mlp_delta=-0.0026, prune_delta=0.0017)

            summary = run_core_periphery_pc_column_synthesis(
                pilot_dirs=(default, seed11),
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(summary["decision"], "core_periphery_pc_column_synthesis_recorded_but_blocked")
            self.assertTrue(
                any(row["criterion"] == "repeat_dense_retention_consistent" for row in summary["failures"])
            )


def _write_source(
    out_dir: Path,
    *,
    seed: int,
    dense_delta: float,
    mlp_delta: float,
    prune_delta: float,
) -> None:
    out_dir.mkdir()
    summary = {
        "status": "pass",
        "decision": "core_periphery_pc_column_pilot_local_candidate",
        "scientific_gate": "ready_for_repeat_only",
        "claim_status": "tiny_local_candidate_not_promoted",
        "seed": seed,
        "steps_per_task": 24,
        "gate_criteria": [
            {"criterion": "contract_present", "passed": True, "severity": "hard"},
            {"criterion": "core_periphery_update_separation", "passed": True, "severity": "claim"},
            {"criterion": "matched_dense_retention", "passed": True, "severity": "claim"},
            {"criterion": "matched_mlp_retention", "passed": True, "severity": "claim"},
            {"criterion": "periphery_first_pruning_signal", "passed": True, "severity": "claim"},
        ],
        "primary_result": {
            "core_periphery_update_norm_ratio": 1.75,
            "core_minus_dense_anchor_mse_drift": dense_delta,
            "core_minus_mlp_anchor_mse_drift": mlp_delta,
            "periphery_first_minus_core_first_prune_delta": prune_delta,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
