from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_predictor_pregate import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_predictor_pregate,
)


class TransformerACSRHiddenFuturePredictorPregateTests(unittest.TestCase):
    def test_trains_prefix_safe_hidden_predictor_and_scores_exact_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            loss_lookup = root / "loss_lookup.csv"
            summary_path = root / "dataset_summary.json"
            _write_dataset(dataset)
            _write_loss_lookup(loss_lookup)
            _write_json(summary_path, {"trainability_gate_passes": True})

            summary = run_transformer_acsr_hidden_future_predictor_pregate(
                dataset_path=dataset,
                loss_lookup_path=loss_lookup,
                dataset_summary_path=summary_path,
                out_dir=root / "out",
                seed=5,
                epochs=3,
                learning_rate=0.003,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_predictor_pregate_gpu_blocked",
            )
            self.assertEqual(
                summary["claim_status"],
                "prefix_safe_hidden_transformer_does_not_clear_full_mechanism_gate",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["train_sequence_count"], 3)
            self.assertEqual(summary["heldout_sequence_count"], 1)
            self.assertEqual(summary["hidden_dim"], 3)
            self.assertEqual(summary["num_columns"], 4)
            self.assertIn(
                "future_hidden_json_target_only",
                summary["forbidden_predictor_fields_enforced"],
            )
            self.assertIn("retention_churn_budget", summary["missing_downstream_controls"])
            self.assertIn("finite_update_commutator_budget", summary["missing_downstream_controls"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "model_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 5)
            self.assertEqual(metrics[0]["model"], "prefix_hidden_causal_transformer")
            self.assertIn("mean_forced_minus_student_router_loss", metrics[0])

            controls = (root / "out" / "control_contract.csv").read_text(encoding="utf-8")
            self.assertIn("exact_arbitrary_pair_same_student_intervention", controls)
            self.assertIn("future_perturbation_invariance", controls)

    def test_fails_closed_when_loss_lookup_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            summary_path = root / "dataset_summary.json"
            _write_dataset(dataset)
            _write_json(summary_path, {"trainability_gate_passes": True})

            summary = run_transformer_acsr_hidden_future_predictor_pregate(
                dataset_path=dataset,
                loss_lookup_path=root / "missing_loss_lookup.csv",
                dataset_summary_path=summary_path,
                out_dir=root / "out",
                epochs=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_predictor_pregate_failed_closed",
            )
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(len(summary["failures"]), 1)


def _write_dataset(path: Path) -> None:
    rows = []
    for sequence_index in range(4):
        split = "train" if sequence_index < 3 else "heldout"
        sequence_id = f"{split}_sequence{sequence_index}"
        previous_hidden = [0.0, 0.0, 0.0]
        for position in range(6):
            if position % 2 == 0:
                teacher = "0,1"
            else:
                teacher = "2,3"
            current_hidden = [
                float(position % 2),
                float(sequence_index),
                float(position),
            ]
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "fold": str(sequence_index),
                    "batch_index": "0",
                    "flat_position": str(position),
                    "position_index": str(position),
                    "current_hidden_json": json.dumps(current_hidden),
                    "previous_hidden_json": json.dumps(previous_hidden),
                    "future_hidden_json_target_only": json.dumps([9.0, 9.0, 9.0]),
                    "future_delta_json_target_only": json.dumps([1.0, 1.0, 1.0]),
                    "teacher_support_logits_json_target_only": json.dumps([1.0, 0.0, 0.0, 1.0]),
                    "teacher_topk_support_target_only": teacher,
                    "student_router_support_eval_only": teacher,
                    "oracle_support_eval_only": teacher,
                    "token_position_null_support_eval_only": "0,1",
                    "target_token_eval_only": "7",
                    "prefix_safe_fields": "current_hidden_json;previous_hidden_json;position_index",
                    "forbidden_predictor_fields": (
                        "future_hidden_json;future_delta_json;teacher_support_logits_json;"
                        "teacher_topk_support;target_token_eval_only;oracle_support_eval_only"
                    ),
                    "teacher_target_fields": (
                        "future_hidden_json;future_delta_json;teacher_support_logits_json;"
                        "teacher_topk_support"
                    ),
                    "future_targets_nondeployable": "True",
                    "hidden_dim": "3",
                    "teacher_support_logit_dim": "4",
                }
            )
            previous_hidden = current_hidden
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_loss_lookup(path: Path) -> None:
    rows = []
    pairs = ["0,1", "0,2", "0,3", "1,2", "1,3", "2,3"]
    for sequence_index in range(4):
        split = "train" if sequence_index < 3 else "heldout"
        sequence_id = f"{split}_sequence{sequence_index}"
        for position in range(6):
            teacher = "0,1" if position % 2 == 0 else "2,3"
            for pair_index, pair in enumerate(pairs):
                is_teacher = pair == teacher
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "split": split,
                        "fold": str(sequence_index),
                        "flat_position": str(position),
                        "position_index": str(position),
                        "forced_support_pair_index": str(pair_index),
                        "forced_support_pair": pair,
                        "forced_support_loss": "1.0" if is_teacher else "2.0",
                        "forced_minus_oracle_loss": "0.0" if is_teacher else "1.0",
                        "forced_minus_student_router_loss": "0.0" if is_teacher else "1.0",
                        "is_teacher_support_pair": str(is_teacher),
                        "is_student_router_support_pair": str(is_teacher),
                        "is_oracle_support_pair": str(is_teacher),
                        "teacher_support": teacher,
                        "teacher_support_forced_loss": "1.0",
                        "student_router_support": teacher,
                        "student_router_support_loss": "1.0",
                        "oracle_support": teacher,
                        "oracle_support_loss": "1.0",
                        "row_family": "same_student_forced_support_exact_pair",
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
