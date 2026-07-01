from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.orthogonalized_sparse_core_periphery_interference_pilot import (
    CANDIDATE_ARM,
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_orthogonalized_sparse_core_periphery_interference_pilot,
)


class OrthogonalizedSparseCorePeripheryInterferencePilotTests(unittest.TestCase):
    def test_missing_pregate_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_orthogonalized_sparse_core_periphery_interference_pilot(
                pregate_dir=root / "missing",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_default_records_bounded_trained_rows_without_gpu_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate"
            _write_pregate(pregate)

            summary = run_orthogonalized_sparse_core_periphery_interference_pilot(
                pregate_dir=pregate,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "orthogonalized_sparse_core_periphery_interference_pilot_recorded")
            self.assertEqual(summary["claim_status"], "bounded_local_cpu_training_rows_recorded_no_gpu_claim")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertFalse(summary["schema_only"])
            self.assertFalse(summary["synthetic_rows_only"])
            self.assertTrue(summary["training_rows_present"])
            self.assertEqual(summary["training_status"], "trained_local_cpu_rows")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["arm_count"], 13)
            self.assertGreaterEqual(summary["matched_control_row_count"], 4)
            self.assertGreaterEqual(summary["leakage_null_row_count"], 5)

            gates = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertEqual(gates["real_training_rows_present"]["passed"], True)
            self.assertEqual(gates["real_training_rows_present"]["actual"], "trained_local_cpu_rows")

            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            candidate = next(row for row in arm_rows if row["arm"] == CANDIDATE_ARM)
            self.assertEqual(candidate["row_source"], "bounded_local_cpu_trained_synthetic_mechanism_stream")
            self.assertNotEqual(candidate["ce"], "")

            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("local CPU training rows", notes)

    def test_schema_only_records_deterministic_schema_pilot_from_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate"
            _write_pregate(pregate)

            summary = run_orthogonalized_sparse_core_periphery_interference_pilot(
                pregate_dir=pregate,
                out_dir=root / "out",
                schema_only=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "orthogonalized_sparse_core_periphery_interference_pilot_recorded",
            )
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertTrue(summary["schema_only"])
            self.assertTrue(summary["synthetic_rows_only"])
            self.assertFalse(summary["training_rows_present"])
            self.assertEqual(summary["training_status"], "schema_only")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["arm_count"], 13)
            self.assertGreaterEqual(summary["matched_control_row_count"], 4)
            self.assertGreaterEqual(summary["leakage_null_row_count"], 5)

            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in arm_rows}
            self.assertIn(CANDIDATE_ARM, arms)
            self.assertIn("random_feature_mlp_residual", arms)
            self.assertIn("orthogonalized_sparse_no_update_masks_ablation", arms)
            self.assertIn("token_position_only_router", arms)
            candidate = next(row for row in arm_rows if row["arm"] == CANDIDATE_ARM)
            self.assertEqual(candidate["row_source"], "deterministic_tiny_synthetic_schema_pilot")
            self.assertEqual(candidate["uses_future_hidden_or_delta"], "False")
            self.assertEqual(candidate["uses_teacher_residual_or_logits_at_eval"], "False")
            self.assertEqual(candidate["uses_oracle_support_at_eval"], "False")
            self.assertTrue(candidate["feature_schema_hash"])

            with (root / "out" / "observable_gates.csv").open(newline="", encoding="utf-8") as handle:
                gates = list(csv.DictReader(handle))
            gate_by_name = {row["criterion"]: row for row in gates}
            self.assertEqual(gate_by_name["ce_guardrail"]["passed"], "False")
            self.assertEqual(gate_by_name["functional_churn_flip_rate"]["passed"], "True")
            self.assertEqual(gate_by_name["deployable_feature_schema"]["passed"], "True")

            with (root / "out" / "matched_control_matrix.csv").open(newline="", encoding="utf-8") as handle:
                controls = list(csv.DictReader(handle))
            self.assertIn("random_feature_mlp_residual", {row["control"] for row in controls})
            mlp = next(row for row in controls if row["control"] == "random_feature_mlp_residual")
            self.assertGreater(float(mlp["candidate_ce_delta"]), 0.0)
            self.assertLess(float(mlp["candidate_churn_delta"]), 0.0)

            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_pregate(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_next_action": "implement_local_orthogonalized_sparse_core_periphery_interference_pilot",
                "requires_gpu_now": False,
                "advance_to_gpu_validation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("mechanism_arms.csv", "matched_controls.csv", "observable_gates.csv", "leakage_nulls.csv"):
        (path / name).write_text("name\nplaceholder\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
