from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_support_value_headroom import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_support_value_headroom,
)


class TransformerACSRHiddenFutureSupportValueHeadroomTests(unittest.TestCase):
    def test_computes_oracle_router_headroom_and_value_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            lookup = root / "loss_lookup.csv"
            predictions = root / "heldout_predictions.csv"
            _write_dataset(dataset)
            _write_loss_lookup(lookup)
            _write_predictions(predictions)

            summary = run_transformer_acsr_hidden_future_support_value_headroom(
                dataset_path=dataset,
                loss_lookup_path=lookup,
                predictions_path=predictions,
                out_dir=root / "out",
                headroom_threshold=0.1,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "support_value_headroom_nontrivial_train_value_router_locally",
            )
            self.assertTrue(summary["value_target_training_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["context_count"], 4)
            self.assertEqual(summary["expected_pair_count_per_context"], 6)
            self.assertGreater(summary["train_mean_oracle_router_gap"], 0.1)
            self.assertGreater(summary["heldout_mean_oracle_router_gap"], 0.1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            rows = _read_csv(root / "out" / "context_value_headroom.csv")
            heldout_rows = [row for row in rows if row["split"] == "heldout"]
            self.assertEqual(heldout_rows[0]["predicted_support_pair"], "0,1")
            self.assertEqual(heldout_rows[0]["predicted_rank"], "1")
            self.assertIn("value_entropy", heldout_rows[0])
            split_rows = _read_csv(root / "out" / "split_summary.csv")
            self.assertEqual({row["split"] for row in split_rows}, {"heldout", "train"})

    def test_fails_closed_on_duplicate_context_pair_loss_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            lookup = root / "loss_lookup.csv"
            _write_dataset(dataset, contexts=(("train", "train_sequence0", 0),))
            _write_loss_lookup(lookup, contexts=(("train", "train_sequence0", 0),), duplicate_first=True)

            summary = run_transformer_acsr_hidden_future_support_value_headroom(
                dataset_path=dataset,
                loss_lookup_path=lookup,
                predictions_path=root / "missing_predictions.csv",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "support_value_headroom_failed_closed")
            self.assertFalse(summary["value_target_training_allowed"])
            self.assertTrue(
                any(failure["reason"] == "duplicate exact loss row for context/support pair" for failure in summary["failures"])
            )


def _write_dataset(
    path: Path,
    *,
    contexts: tuple[tuple[str, str, int], ...] = (
        ("train", "train_sequence0", 0),
        ("train", "train_sequence0", 1),
        ("heldout", "heldout_sequence0", 0),
        ("heldout", "heldout_sequence0", 1),
    ),
) -> None:
    rows = []
    for split, sequence_id, position in contexts:
        rows.append(
            {
                "sequence_id": sequence_id,
                "split": split,
                "fold": "0",
                "batch_index": "0",
                "flat_position": str(position),
                "position_index": str(position),
                "current_hidden_json": json.dumps([float(position), 1.0, 0.0]),
                "previous_hidden_json": json.dumps([0.0, 0.0, 0.0]),
                "future_hidden_json_target_only": json.dumps([9.0, 9.0, 9.0]),
                "future_delta_json_target_only": json.dumps([1.0, 1.0, 1.0]),
                "teacher_support_logits_json_target_only": json.dumps([1.0, 0.0, 0.0, 1.0]),
                "teacher_topk_support_target_only": "0,2",
                "student_router_support_eval_only": "2,3",
                "oracle_support_eval_only": "0,1",
                "target_token_eval_only": "7",
                "hidden_dim": "3",
                "teacher_support_logit_dim": "4",
            }
        )
    _write_csv(path, rows)


def _write_loss_lookup(
    path: Path,
    *,
    contexts: tuple[tuple[str, str, int], ...] = (
        ("train", "train_sequence0", 0),
        ("train", "train_sequence0", 1),
        ("heldout", "heldout_sequence0", 0),
        ("heldout", "heldout_sequence0", 1),
    ),
    duplicate_first: bool = False,
) -> None:
    pair_losses = {
        "0,1": 1.0,
        "0,2": 1.2,
        "0,3": 1.7,
        "1,2": 1.8,
        "1,3": 1.9,
        "2,3": 2.0,
    }
    rows = []
    for split, sequence_id, position in contexts:
        for pair_index, (pair, loss) in enumerate(pair_losses.items()):
            row = {
                "sequence_id": sequence_id,
                "split": split,
                "fold": "0",
                "flat_position": str(position),
                "position_index": str(position),
                "forced_support_pair_index": str(pair_index),
                "forced_support_pair": pair,
                "forced_support_loss": str(loss),
                "forced_minus_oracle_loss": str(loss - 1.0),
                "forced_minus_student_router_loss": str(loss - 2.0),
                "is_teacher_support_pair": str(pair == "0,2"),
                "is_student_router_support_pair": str(pair == "2,3"),
                "is_oracle_support_pair": str(pair == "0,1"),
                "teacher_support": "0,2",
                "teacher_support_forced_loss": "1.2",
                "student_router_support": "2,3",
                "student_router_support_loss": "2.0",
                "oracle_support": "0,1",
                "oracle_support_loss": "1.0",
                "row_family": "same_student_forced_support_exact_pair",
            }
            rows.append(row)
            if duplicate_first and pair_index == 0:
                rows.append(dict(row))
    _write_csv(path, rows)


def _write_predictions(path: Path) -> None:
    rows = []
    for position in (0, 1):
        rows.append(
            {
                "sequence_id": "heldout_sequence0",
                "split": "heldout",
                "fold": "0",
                "flat_position": str(position),
                "position_index": str(position),
                "predicted_support_pair": "0,1",
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
