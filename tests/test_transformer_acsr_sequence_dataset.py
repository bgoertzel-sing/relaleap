from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_sequence_dataset import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_sequence_dataset,
)


class TransformerACSRSequenceDatasetTests(unittest.TestCase):
    def test_materializes_support_target_rows_and_labels_missing_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for seed in range(1, 4):
                path = root / f"seed{seed}_per_token_supports.csv"
                _write_per_token_source(path)
                sources.append((f"seed{seed}", path, True))

            summary = run_transformer_acsr_sequence_dataset(
                source_paths=tuple(sources),
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_support_target_row_dataset_materialized",
            )
            self.assertEqual(
                summary["claim_status"],
                "support_distribution_dataset_ready_hidden_chunks_missing_no_gpu",
            )
            self.assertTrue(summary["support_target_dataset_available"])
            self.assertTrue(summary["sequence_split_available"])
            self.assertTrue(summary["trainable_support_only_now"])
            self.assertFalse(summary["hidden_or_future_chunk_targets_available"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["row_count"], 24)
            self.assertEqual(summary["train_row_count"], 18)
            self.assertEqual(summary["heldout_row_count"], 6)
            self.assertGreaterEqual(summary["prefix_safe_feature_count"], 4)
            self.assertGreaterEqual(summary["nondeployable_teacher_target_count"], 1)
            self.assertGreaterEqual(summary["missing_tensor_field_count"], 5)
            self.assertEqual(
                summary["selected_next_step"],
                "train_local_support_only_prefix_safe_transformer_acsr_pregate_with_nulls",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "dataset_rows.csv").open(newline="", encoding="utf-8") as handle:
                dataset_rows = list(csv.DictReader(handle))
            self.assertEqual(dataset_rows[0]["split"], "train")
            self.assertEqual(dataset_rows[0]["previous_teacher_support_left"], "-1")
            self.assertEqual(dataset_rows[1]["previous_teacher_support_left"], "1")
            self.assertIn("target_token_eval_only", dataset_rows[0])

            with (root / "out" / "field_provenance.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                field_rows = list(csv.DictReader(handle))
            target_fields = {
                row["field"]: row for row in field_rows if row["field"] == "target_token_eval_only"
            }
            self.assertEqual(target_fields["target_token_eval_only"]["forbidden_predictor_input"], "True")

            missing = (root / "out" / "missing_tensor_fields.csv").read_text(encoding="utf-8")
            self.assertIn("future_hidden", missing)
            self.assertIn("teacher_support_logits_or_soft_distribution", missing)

    def test_fails_closed_when_required_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = run_transformer_acsr_sequence_dataset(
                source_paths=(("missing", root / "missing.csv", True),),
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_support_target_row_dataset_failed_closed",
            )
            self.assertFalse(summary["support_target_dataset_available"])
            self.assertFalse(summary["trainable_support_only_now"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(len(summary["failures"]), 1)


def _write_per_token_source(path: Path) -> None:
    rows = []
    for fold in range(4):
        for flat_position in range(2):
            rows.append(
                {
                    "flat_position": str(flat_position),
                    "fold": str(fold),
                    "linear_support": "0,1",
                    "linear_support_forced_into_student_loss": "4.0",
                    "marginal_shuffled_student_support_loss": "4.1",
                    "oracle_best_support_for_student_loss": "2.0",
                    "oracle_support": "2,3",
                    "student_router_support_loss": "2.4",
                    "student_support": "3,4",
                    "target_token": "9",
                    "teacher_student_exact_pair_match": "False",
                    "teacher_support": "1,5",
                    "teacher_support_forced_into_student_loss": "2.5",
                    "token_position_null_support": "4,5",
                    "token_position_null_support_forced_into_student_loss": "2.6",
                    "uniform_random_support_loss": "4.2",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
