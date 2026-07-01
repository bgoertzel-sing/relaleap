from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_control_audit import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_control_audit,
)


class TransformerACSRHiddenFutureControlAuditTests(unittest.TestCase):
    def test_writes_local_control_artifacts_and_blocks_gpu_without_exact_commutator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset_rows.csv"
            loss_lookup = root / "loss_lookup.csv"
            pregate = root / "pregate_summary.json"
            predictions = root / "heldout_predictions.csv"
            _write_dataset(dataset)
            _write_loss_lookup(loss_lookup)
            _write_json(
                pregate,
                {
                    "status": "pass",
                    "prefix_hidden_jaccard": 1.0,
                    "same_student_loss_gate_passes": False,
                },
            )
            _write_predictions(predictions)

            summary = run_transformer_acsr_hidden_future_control_audit(
                dataset_path=dataset,
                loss_lookup_path=loss_lookup,
                pregate_path=pregate,
                predictions_path=predictions,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_control_audit_gpu_blocked",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["controls"]["current_hidden_stratified_null_available"])
            self.assertTrue(summary["controls"]["retention_churn_artifact_available"])
            self.assertTrue(summary["controls"]["future_perturbation_invariance_passes"])
            self.assertFalse(summary["controls"]["commutator_exact_budget_available"])
            self.assertEqual(summary["heldout_prediction_count"], 4)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            future_rows = _read_csv(root / "out" / "future_perturbation_invariance.csv")
            self.assertEqual(
                future_rows[0]["prediction_invariant_to_future_target_perturbation"],
                "true",
            )
            commutator_rows = _read_csv(root / "out" / "commutator_proxy.csv")
            self.assertEqual(
                commutator_rows[0]["exact_finite_update_commutator_available"],
                "false",
            )

    def test_fails_closed_when_required_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_transformer_acsr_hidden_future_control_audit(
                dataset_path=root / "missing_dataset.csv",
                loss_lookup_path=root / "missing_loss_lookup.csv",
                pregate_path=root / "missing_summary.json",
                predictions_path=root / "missing_predictions.csv",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_control_audit_failed_closed",
            )
            self.assertEqual(len(summary["failures"]), 4)
            self.assertFalse(summary["advance_to_gpu_validation"])


def _write_dataset(path: Path) -> None:
    rows = []
    for split, sequence_id in (("train", "train_sequence0"), ("heldout", "heldout_sequence0")):
        previous_hidden = [0.0, 0.0, 0.0]
        for position in range(4):
            teacher = "0,1" if position % 2 == 0 else "2,3"
            current_hidden = [float(position % 2), float(position), 1.0]
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "fold": "0",
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
                    "target_token_eval_only": "7",
                    "hidden_dim": "3",
                    "teacher_support_logit_dim": "4",
                }
            )
            previous_hidden = current_hidden
    _write_csv(path, rows)


def _write_loss_lookup(path: Path) -> None:
    rows = []
    pairs = ["0,1", "0,2", "0,3", "1,2", "1,3", "2,3"]
    for split, sequence_id in (("train", "train_sequence0"), ("heldout", "heldout_sequence0")):
        for position in range(4):
            teacher = "0,1" if position % 2 == 0 else "2,3"
            for pair in pairs:
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "split": split,
                        "position_index": str(position),
                        "forced_support_pair": pair,
                        "forced_support_loss": "1.0" if pair == teacher else "2.0",
                        "forced_minus_student_router_loss": "0.0" if pair == teacher else "1.0",
                    }
                )
    _write_csv(path, rows)


def _write_predictions(path: Path) -> None:
    rows = []
    for position in range(4):
        pair = "0,1" if position % 2 == 0 else "2,3"
        rows.append(
            {
                "sequence_id": "heldout_sequence0",
                "split": "heldout",
                "fold": "0",
                "flat_position": str(position),
                "position_index": str(position),
                "predicted_support_pair": pair,
            }
        )
    _write_csv(path, rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
