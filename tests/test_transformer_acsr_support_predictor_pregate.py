from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_support_predictor_pregate import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_support_predictor_pregate,
)


class TransformerACSRSupportPredictorPregateTests(unittest.TestCase):
    def test_trains_support_predictor_and_keeps_gpu_blocked_without_budget_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            summary_path = root / "dataset_summary.json"
            _write_dataset(dataset)
            _write_json(summary_path, {"trainable_support_only_now": True})

            summary = run_transformer_acsr_support_predictor_pregate(
                dataset_path=dataset,
                dataset_summary_path=summary_path,
                out_dir=root / "out",
                seed=3,
                epochs=3,
                learning_rate=0.005,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_support_predictor_pregate_gpu_blocked",
            )
            self.assertEqual(
                summary["claim_status"],
                "support_only_prefix_transformer_does_not_clear_full_mechanism_gate",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["train_sequence_count"], 3)
            self.assertEqual(summary["heldout_sequence_count"], 1)
            self.assertGreaterEqual(summary["num_columns"], 4)
            self.assertIn(
                "exact_arbitrary_pair_same_student_intervention",
                summary["missing_downstream_controls"],
            )
            self.assertIn("retention_churn_budget", summary["missing_downstream_controls"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "model_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 5)
            self.assertEqual(metrics[0]["model"], "prefix_support_causal_transformer")

            controls = (root / "out" / "control_contract.csv").read_text(encoding="utf-8")
            self.assertIn("shuffled_target_transformer", controls)
            self.assertIn("finite_update_commutator_budget", controls)

    def test_fails_closed_when_dataset_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = run_transformer_acsr_support_predictor_pregate(
                dataset_path=root / "missing.csv",
                dataset_summary_path=root / "missing_summary.json",
                out_dir=root / "out",
                epochs=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_support_predictor_pregate_failed_closed",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(len(summary["failures"]), 1)


def _write_dataset(path: Path) -> None:
    rows = []
    for fold in range(4):
        split = "train" if fold in {0, 1, 2} else "heldout"
        previous = (-1, -1)
        for position in range(6):
            if position % 2 == 0:
                teacher = (0, 1)
            else:
                teacher = (2, 3)
            rows.append(
                {
                    "source": "synthetic",
                    "seed_index": "1",
                    "fold": str(fold),
                    "split": split,
                    "flat_position": str(position),
                    "position_fraction": str(position / 5),
                    "position_parity": str(position % 2),
                    "previous_teacher_support_left": str(previous[0]),
                    "previous_teacher_support_right": str(previous[1]),
                    "teacher_support_left": str(teacher[0]),
                    "teacher_support_right": str(teacher[1]),
                    "student_support_left": str(teacher[0]),
                    "student_support_right": str(teacher[1]),
                    "oracle_support_left": str(teacher[0]),
                    "oracle_support_right": str(teacher[1]),
                    "token_position_null_support_left": "0",
                    "token_position_null_support_right": "1",
                    "teacher_student_exact_pair_match": "True",
                    "target_token_eval_only": "0",
                    "teacher_support_forced_into_student_loss": "1.0",
                    "student_router_support_loss": "1.0",
                    "oracle_best_support_for_student_loss": "1.0",
                    "token_position_null_support_forced_into_student_loss": "1.1",
                }
            )
            previous = teacher
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
