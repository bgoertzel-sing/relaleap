from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_capture_design import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_capture_design,
)


class TransformerACSRHiddenFutureCaptureDesignTests(unittest.TestCase):
    def test_records_capture_contract_after_support_closeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            sequence = root / "sequence.json"
            missing = root / "missing.csv"
            review = root / "latest-review.md"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_support_only_branch_closed_hidden_future_capture_required",
                    "support_branch_closed": True,
                    "hidden_future_capture_required": True,
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                sequence,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_support_target_row_dataset_materialized",
                    "support_target_dataset_available": True,
                    "sequence_split_available": True,
                },
            )
            _write_missing_tensors(missing)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Start a bounded local Transformer-ACSR pregate.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_hidden_future_capture_design(
                closeout_path=closeout,
                sequence_dataset_path=sequence,
                missing_tensors_path=missing,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_capture_design_recorded",
            )
            self.assertEqual(
                summary["claim_status"],
                "hidden_future_capture_contract_ready_no_gpu",
            )
            self.assertTrue(summary["support_branch_closed"])
            self.assertTrue(summary["support_dataset_available"])
            self.assertTrue(summary["capture_design_ready"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["ben_should_be_notified"])
            self.assertEqual(summary["failures"], [])
            self.assertIn("future_hidden", summary["missing_tensor_fields"])
            self.assertIn("teacher_support_logits_or_soft_distribution", summary["missing_tensor_fields"])
            self.assertIn(
                "extend command-driven teacher artifact capture",
                summary["selected_next_step"],
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "field_capture_plan.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fields = {row["field"]: row for row in csv.DictReader(handle)}
            self.assertEqual(fields["current_hidden"]["allowed_as_predictor_input"], "True")
            self.assertEqual(fields["future_hidden"]["nondeployable_teacher_target"], "True")
            self.assertEqual(fields["target_token"]["forbidden_predictor_input"], "True")

            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("RunPod and Colab validation remain blocked", notes)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_transformer_acsr_hidden_future_capture_design(
                closeout_path=root / "missing_closeout.json",
                sequence_dataset_path=root / "missing_sequence.json",
                missing_tensors_path=root / "missing_tensors.csv",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_capture_design_failed_closed",
            )
            self.assertFalse(summary["support_branch_closed"])
            self.assertFalse(summary["capture_design_ready"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreaterEqual(len(summary["failures"]), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_missing_tensors(path: Path) -> None:
    rows = [
        {
            "field": "current_hidden",
            "required_for": "strict hidden-feature Transformer-ACSR predictor",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "previous_hidden",
            "required_for": "strict hidden-feature Transformer-ACSR predictor",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "future_hidden",
            "required_for": "teacher future-chunk prediction target",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "future_delta",
            "required_for": "teacher future-delta prediction target",
            "reason": "not present in per_token_supports.csv",
        },
        {
            "field": "teacher_support_logits_or_soft_distribution",
            "required_for": "support KL training target",
            "reason": "only hard top-k2 teacher support pairs are present",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
