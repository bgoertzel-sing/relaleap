from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_vector_capture import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_sparse_factorization_vector_capture,
)


class LowChurnMlpSparseFactorizationVectorCaptureTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_sparse_factorization_vector_capture(
                training_harness_dir=root / "missing_harness",
                pregate_dir=root / "missing_pregate",
                out_dir=root / "out",
                train_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["runtime_failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_captures_raw_vectors_and_blocks_gpu_advancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harness = root / "harness"
            pregate = root / "pregate"
            _write_harness(harness)
            _write_pregate(pregate)

            summary = run_low_churn_mlp_sparse_factorization_vector_capture(
                training_harness_dir=harness,
                pregate_dir=pregate,
                out_dir=root / "out",
                train_steps=1,
                seed=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_sparse_factorization_vector_capture_recorded")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["raw_teacher_vector_row_count"], 124)
            self.assertGreater(summary["heldout_raw_teacher_vector_row_count"], 0)
            self.assertEqual(summary["logit_intervention_row_count"], 124)
            self.assertTrue(summary["advancement_failures"])

            with (root / "out" / "raw_teacher_residual_vectors.csv").open(newline="", encoding="utf-8") as handle:
                vector_rows = list(csv.DictReader(handle))
            self.assertEqual(len(vector_rows), 124)
            first = vector_rows[0]
            residual_vector = json.loads(first["teacher_residual_update_vector"])
            base_logits = json.loads(first["base_logits"])
            teacher_logits = json.loads(first["teacher_logits"])
            logit_delta = json.loads(first["teacher_logit_delta"])
            self.assertEqual(len(residual_vector), 32)
            self.assertEqual(len(base_logits), int(first["vocab_size"]))
            self.assertEqual(len(teacher_logits), int(first["vocab_size"]))
            self.assertEqual(len(logit_delta), int(first["vocab_size"]))
            self.assertEqual({row["raw_teacher_vector_available"] for row in vector_rows}, {"True"})

            with (root / "out" / "logit_intervention_rows.csv").open(newline="", encoding="utf-8") as handle:
                intervention_rows = list(csv.DictReader(handle))
            self.assertEqual(len(intervention_rows), 124)
            self.assertIn("teacher_gain_vs_base_ce", intervention_rows[0])
            self.assertIn("necessity_zero_update_ce_delta", intervention_rows[0])


def _write_harness(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_ceiling_training_harness_recorded",
                "selected_next_action": "capture_raw_low_churn_teacher_residual_vectors_for_sparse_factorization",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_pregate(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_next_action": "implement_low_churn_mlp_residual_control_pilot",
                "budget_rows": [
                    {"metric": "dense24_residual_l2_ceiling", "value": 1.0},
                    {"metric": "dense24_anchor_logit_mse_ceiling", "value": 0.02},
                    {"metric": "dense24_flip_churn_ceiling", "value": 0.25},
                    {"metric": "dense24_ce_reference", "value": 3.7},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
