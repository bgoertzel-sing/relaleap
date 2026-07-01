from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_vector_training import (
    ARMS,
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_sparse_factorization_vector_training,
)


class LowChurnMlpSparseFactorizationVectorTrainingTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_sparse_factorization_vector_training(
                vector_capture_dir=root / "missing_capture",
                design_dir=root / "missing_design",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["runtime_failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_vector_training_rows_and_keeps_gpu_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            capture = root / "capture"
            design = root / "design"
            _write_capture(capture)
            _write_design(design)

            summary = run_low_churn_mlp_sparse_factorization_vector_training(
                vector_capture_dir=capture,
                design_dir=design,
                out_dir=root / "out",
                column_count=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_sparse_factorization_vector_training_recorded")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertTrue(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["arm_count"], len(ARMS))
            self.assertEqual(summary["heldout_training_row_count"], len(ARMS) * 2)

            with (root / "out" / "training_rows.csv").open(newline="", encoding="utf-8") as handle:
                training_rows = list(csv.DictReader(handle))
            self.assertEqual(len(training_rows), len(ARMS) * 4)
            self.assertEqual({row["arm"] for row in training_rows}, set(ARMS))
            self.assertEqual({row["training_mode"] for row in training_rows}, {"vector_centroid_dictionary"})

            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            self.assertEqual({row["arm"] for row in arm_rows}, set(ARMS))
            self.assertIn("teacher_residual_reconstruction_r2", arm_rows[0])
            self.assertIn("functional_churn_flip_rate_proxy", arm_rows[0])
            self.assertIn("finite_update_commutator_proxy", arm_rows[0])
            self.assertIn("intervention_fingerprint_specificity_proxy", arm_rows[0])


def _write_capture(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_vector_capture_recorded",
                "selected_next_action": "implement_vector_sparse_factorization_ceiling_training",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        _vector_row("row0", 0, "train_anchor", [1.0, 0.0], 4.0, 3.8),
        _vector_row("row1", 1, "train_anchor", [0.0, 1.0], 4.0, 3.9),
        _vector_row("row2", 2, "heldout", [1.0, 0.2], 4.0, 3.7),
        _vector_row("row3", 3, "heldout", [0.2, 1.0], 4.0, 3.8),
    ]
    _write_csv(path / "raw_teacher_residual_vectors.csv", rows)
    _write_csv(
        path / "logit_intervention_rows.csv",
        [
            {
                "teacher_row_id": row["teacher_row_id"],
                "token_index": row["token_index"],
                "split": row["split"],
                "teacher_gain_vs_base_ce": 0.1,
                "necessity_zero_update_ce_delta": 0.1,
                "teacher_anchor_kl_vs_base": 0.001,
            }
            for row in rows
        ],
    )


def _write_design(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "low_churn_mlp_sparse_factorization_ceiling_design_recorded"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "support_arms.csv",
        [{"arm": arm, "support_type": "test", "trainable": arm.startswith("oracle") or arm.startswith("learned")} for arm in ARMS],
    )


def _vector_row(row_id: str, token_index: int, split: str, vector: list[float], base_ce: float, teacher_ce: float) -> dict[str, object]:
    return {
        "teacher_arm": "low_churn_mlp_residual_control",
        "teacher_row_id": row_id,
        "token_index": token_index,
        "split": split,
        "target_token_id": token_index,
        "hidden_dim": len(vector),
        "vocab_size": 3,
        "base_ce_loss": base_ce,
        "teacher_ce_loss": teacher_ce,
        "teacher_delta_vs_base_ce": teacher_ce - base_ce,
        "teacher_residual_update_l2": sum(item * item for item in vector) ** 0.5,
        "teacher_anchor_kl_vs_base": 0.001,
        "teacher_prediction_changed_vs_base": False,
        "raw_teacher_vector_available": True,
        "raw_intervention_available": True,
        "teacher_residual_update_vector": json.dumps(vector),
        "base_logits": json.dumps([0.0, 0.0, 0.0]),
        "teacher_logits": json.dumps([0.1, 0.0, 0.0]),
        "teacher_logit_delta": json.dumps([0.1, 0.0, 0.0]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
