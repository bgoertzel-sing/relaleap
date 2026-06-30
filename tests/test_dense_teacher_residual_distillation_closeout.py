from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_distillation_closeout import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_residual_distillation_closeout,
)


class DenseTeacherResidualDistillationCloseoutTest(unittest.TestCase):
    def test_closes_coherent_negative_comparison_before_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            comparison = root / "comparison.json"
            gates = root / "gate_criteria.csv"
            review = root / "latest-review.md"
            _write_json(
                comparison,
                {
                    "status": "fail",
                    "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                    "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
                    "dense_teacher_ce_loss": 0.31,
                    "base_ce_loss": 4.16,
                    "gate_status": {"passes_dense_teacher_distillation_gate": False},
                    "failures": [
                        {"criterion": "acsr_ce_not_worse_than_teacher_by_large_margin"},
                        {"criterion": "calibrated_teacher_scale_gate"},
                    ],
                },
            )
            _write_csv(
                gates,
                [
                    {
                        "criterion": "source_gates_present_and_passing",
                        "passed": True,
                    },
                    {
                        "criterion": "acsr_ce_not_worse_than_teacher_by_large_margin",
                        "passed": False,
                    },
                    {
                        "criterion": "calibrated_teacher_scale_gate",
                        "passed": False,
                    },
                ],
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_residual_distillation_closeout(
                comparison_path=comparison,
                gate_rows_path=gates,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_teacher_residual_distillation_branch_closed")
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "dense_teacher_distillation_negative_closeout_no_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["evidence"]["comparison_gate_passes"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)

    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_residual_distillation_closeout(
                comparison_path=root / "missing.json",
                gate_rows_path=root / "missing.csv",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
