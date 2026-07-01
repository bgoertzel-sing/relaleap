from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_decision_audit import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    ROUTER_REDESIGN_ACTION,
    run_low_churn_mlp_sparse_factorization_decision_audit,
)


class LowChurnMlpSparseFactorizationDecisionAuditTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_sparse_factorization_decision_audit(
                vector_training_dir=root / "missing_training",
                vector_capture_dir=root / "missing_capture",
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

    def test_exact_oracle_is_nondeployable_and_learned_router_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training = root / "training"
            capture = root / "capture"
            _write_training(training)
            _write_capture(capture)

            summary = run_low_churn_mlp_sparse_factorization_decision_audit(
                vector_training_dir=training,
                vector_capture_dir=capture,
                out_dir=root / "out",
                dictionary_size=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], ROUTER_REDESIGN_ACTION)
            self.assertTrue(summary["exact_oracle_nondeployable"])
            self.assertTrue(summary["learned_router_blocks_gpu"])
            self.assertGreaterEqual(summary["global_dictionary_oracle_r2"], 0.5)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])

            with (root / "out" / "blame_rows.csv").open(newline="", encoding="utf-8") as handle:
                blame_rows = list(csv.DictReader(handle))
            support = [row for row in blame_rows if row["blame_category"] == "support_router_failure"][0]
            self.assertEqual(support["disposition"], "blocking")

            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("exact oracle is labeled nondeployable", notes.lower())


def _write_training(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_vector_training_recorded",
                "claim_status": "vector_sparse_factorization_ceiling_local_gates_block_gpu",
                "oracle_learned_r2_gap": 1.2,
                "advance_to_gpu_validation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "arm_metrics.csv",
        [
            {
                "arm": "oracle_support_sparse_ceiling",
                "teacher_residual_reconstruction_r2": 1.0,
                "teacher_residual_reconstruction_mse": 0.0,
            },
            {
                "arm": "learned_router_sparse_factorization",
                "teacher_residual_reconstruction_r2": -0.2,
                "teacher_residual_reconstruction_mse": 0.3,
            },
            {
                "arm": "route_scrambled_same_values",
                "teacher_residual_reconstruction_r2": -0.1,
                "teacher_residual_reconstruction_mse": 0.2,
            },
        ],
    )


def _write_capture(path: Path) -> None:
    path.mkdir(parents=True)
    rows = [
        _vector_row("r0", "train_anchor", 0, [1.0, 0.0]),
        _vector_row("r1", "train_anchor", 1, [0.0, 1.0]),
        _vector_row("r2", "train_anchor", 2, [1.1, 0.0]),
        _vector_row("r3", "train_anchor", 3, [0.0, 1.1]),
        _vector_row("h0", "heldout", 4, [1.05, 0.0]),
        _vector_row("h1", "heldout", 5, [0.0, 1.05]),
    ]
    _write_csv(path / "raw_teacher_residual_vectors.csv", rows)


def _vector_row(row_id: str, split: str, token_index: int, vector: list[float]) -> dict[str, object]:
    return {
        "teacher_row_id": row_id,
        "split": split,
        "token_index": token_index,
        "teacher_residual_update_vector": json.dumps(vector),
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
