from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_sequence_dataset import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_sequence_dataset,
)


class TransformerACSRHiddenFutureSequenceDatasetTests(unittest.TestCase):
    def test_materializes_dataset_and_fails_trainability_closed_when_heldout_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden_future_rows.csv"
            interventions = root / "intervention_rows_exact.csv"
            _write_hidden_future_rows(hidden, splits=("heldout",))
            _write_intervention_rows(interventions, splits=("heldout",))

            summary = run_transformer_acsr_hidden_future_sequence_dataset(
                hidden_future_rows_path=hidden,
                intervention_rows_path=interventions,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_sequence_dataset_materialized_trainability_blocked",
            )
            self.assertEqual(
                summary["claim_status"],
                "hidden_future_sequence_dataset_heldout_only_no_training_no_gpu",
            )
            self.assertEqual(summary["hidden_future_row_count"], 2)
            self.assertEqual(summary["exact_intervention_row_count"], 12)
            self.assertEqual(summary["expected_pair_count_per_position"], 6)
            self.assertFalse(summary["split_coverage_available"])
            self.assertFalse(summary["trainability_gate_passes"])
            self.assertEqual(summary["trainability_status"], "fail_closed")
            self.assertTrue(summary["exact_loss_lookup_complete"])
            self.assertTrue(summary["leakage_contract_passes"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["failure_count"], 0)
            self.assertIn("extend hidden/future capture", summary["selected_next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "field_provenance.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fields = {row["field"]: row for row in csv.DictReader(handle)}
            self.assertEqual(fields["current_hidden_json"]["allowed_as_predictor_input"], "True")
            self.assertEqual(
                fields["future_hidden_json_target_only"]["forbidden_predictor_input"], "True"
            )
            self.assertEqual(
                fields["teacher_support_logits_json_target_only"]["allowed_as_training_target"],
                "True",
            )

    def test_trainability_gate_passes_when_train_and_heldout_sequences_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden_future_rows.csv"
            interventions = root / "intervention_rows_exact.csv"
            _write_hidden_future_rows(hidden, splits=("train", "heldout"))
            _write_intervention_rows(interventions, splits=("train", "heldout"))

            summary = run_transformer_acsr_hidden_future_sequence_dataset(
                hidden_future_rows_path=hidden,
                intervention_rows_path=interventions,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_sequence_dataset_trainable_locally",
            )
            self.assertTrue(summary["split_coverage_available"])
            self.assertTrue(summary["trainability_gate_passes"])
            self.assertEqual(summary["train_sequence_count"], 1)
            self.assertEqual(summary["heldout_sequence_count"], 1)

    def test_missing_exact_pair_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden_future_rows.csv"
            interventions = root / "intervention_rows_exact.csv"
            _write_hidden_future_rows(hidden, splits=("train", "heldout"))
            _write_intervention_rows(interventions, splits=("train", "heldout"), omit_last=True)

            summary = run_transformer_acsr_hidden_future_sequence_dataset(
                hidden_future_rows_path=hidden,
                intervention_rows_path=interventions,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_sequence_dataset_failed_closed",
            )
            self.assertFalse(summary["trainability_gate_passes"])
            self.assertGreater(summary["failure_count"], 0)


def _write_hidden_future_rows(path: Path, *, splits: tuple[str, ...]) -> None:
    rows = []
    for split_index, split in enumerate(splits):
        for position in range(2):
            sequence_id = f"{split}_sequence"
            rows.append(
                {
                    "batch_index": str(split_index),
                    "current_hidden_json": json.dumps([1.0 + position, 2.0, 3.0]),
                    "flat_position": str(position),
                    "fold": str(split_index),
                    "forbidden_predictor_fields": (
                        "future_hidden_json;future_delta_json;teacher_support_logits_json;"
                        "teacher_topk_support;target_token_eval_only;oracle_support_eval_only"
                    ),
                    "future_delta_json": json.dumps([0.1, 0.2, 0.3]),
                    "future_hidden_json": json.dumps([1.1, 2.2, 3.3]),
                    "future_targets_nondeployable": "True",
                    "oracle_support_eval_only": "0,1",
                    "position_index": str(position),
                    "prefix_safe_fields": "current_hidden_json;previous_hidden_json;position_index",
                    "previous_hidden_json": json.dumps([0.0, 0.0, 0.0]),
                    "router_teacher_provenance": "unit_test_teacher",
                    "sequence_id": sequence_id,
                    "split": split,
                    "student_router_support": "1,2",
                    "target_token_eval_only": "7",
                    "teacher_support_logits_json": json.dumps([0.0, 1.0, 2.0, 3.0]),
                    "teacher_target_fields": (
                        "future_hidden_json;future_delta_json;teacher_support_logits_json;"
                        "teacher_topk_support"
                    ),
                    "teacher_topk_support": "2,3",
                    "token_position_null_support": "0,3",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_intervention_rows(
    path: Path,
    *,
    splits: tuple[str, ...],
    omit_last: bool = False,
) -> None:
    rows = []
    pairs = ["0,1", "0,2", "0,3", "1,2", "1,3", "2,3"]
    for split_index, split in enumerate(splits):
        for position in range(2):
            sequence_id = f"{split}_sequence"
            for pair_index, pair in enumerate(pairs):
                if omit_last and split == "heldout" and position == 1 and pair_index == len(pairs) - 1:
                    continue
                rows.append(
                    {
                        "flat_position": str(position),
                        "fold": str(split_index),
                        "forced_minus_oracle_loss": "0.1",
                        "forced_minus_student_router_loss": "0.2",
                        "forced_support_loss": str(3.0 + pair_index),
                        "forced_support_pair": pair,
                        "forced_support_pair_index": str(pair_index),
                        "is_oracle_support_pair": str(pair == "0,1"),
                        "is_student_router_support_pair": str(pair == "1,2"),
                        "is_teacher_support_pair": str(pair == "2,3"),
                        "oracle_support": "0,1",
                        "oracle_support_loss": "2.0",
                        "position_index": str(position),
                        "row_family": "same_student_forced_support_exact_pair",
                        "sequence_id": sequence_id,
                        "split": split,
                        "student_router_support": "1,2",
                        "student_router_support_loss": "2.5",
                        "teacher_support": "2,3",
                        "teacher_support_forced_loss": "2.7",
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
