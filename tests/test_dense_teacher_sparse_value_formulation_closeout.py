from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_sparse_value_formulation_closeout import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_sparse_value_formulation_closeout,
)


class DenseTeacherSparseValueFormulationCloseoutTests(unittest.TestCase):
    def test_closes_current_sparse_dictionary_variant_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostic = root / "diagnostic"
            source_assay = root / "assay"
            review = root / "latest-review.md"
            _write_diagnostic(diagnostic)
            _write_json(
                source_assay / "summary.json",
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_value_capacity_norm_assay_recorded",
                    "claim_status": "value_capacity_norm_control_local_gates_block_gpu",
                    "git_commit": "source-assay-commit",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Close or redesign sparse value formulation locally.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_sparse_value_formulation_closeout(
                diagnostic_dir=diagnostic,
                source_assay_dir=source_assay,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(
                summary["claim_status"],
                "current_sparse_dictionary_value_formulation_retired_before_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["training_executed"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            self.assertEqual(summary["direction_shift"]["recommendation_disposition"], "accepted")
            self.assertGreater(summary["evidence"]["oracle_in_column_mse_gap_vs_flat"], 0.10)
            self.assertGreater(summary["evidence"]["global_dictionary_mse_gap_vs_flat"], 0.10)
            self.assertGreater(summary["evidence"]["learned_sparse_ce_gap_vs_flat"], 0.02)
            closeout = {row["criterion"]: row for row in summary["closeout_rows"]}
            self.assertTrue(closeout["gpu_validation_blocked"]["passed"])
            self.assertTrue(closeout["nondeployable_in_column_sparse_ceiling_loses_to_flat_value"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertIn("post-dense-teacher sparse-dictionary branch selector", selected[0]["next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_diagnostic_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_dense_teacher_sparse_value_formulation_closeout(
                diagnostic_dir=root / "missing",
                source_assay_dir=root / "missing_assay",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_diagnostic(path: Path) -> None:
    rows = [
        {
            "arm": "same_router_flat_value_control",
            "ce": 1.406017,
            "teacher_residual_reconstruction_mse": 0.864636,
            "oracle_value_code_non_deployable": False,
            "uses_future_hidden_or_delta": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        {
            "arm": "oracle_support_oracle_value_code_sparse",
            "ce": 1.634448,
            "teacher_residual_reconstruction_mse": 1.97803,
            "oracle_value_code_non_deployable": True,
        },
        {
            "arm": "learned_support_learned_value_code_sparse",
            "ce": 1.715321,
            "teacher_residual_reconstruction_mse": 2.54473,
            "oracle_value_code_non_deployable": False,
            "uses_future_hidden_or_delta": False,
            "uses_task_id": False,
            "uses_teacher_labels_in_deployable_router": False,
        },
        {
            "arm": "global_oracle_support_value_code_sparse",
            "ce": 1.575778,
            "teacher_residual_reconstruction_mse": 1.112734,
            "oracle_value_code_non_deployable": True,
        },
        {
            "arm": "random_support_oracle_value_code_null",
            "ce": 1.87925,
            "teacher_residual_reconstruction_mse": 2.530691,
            "oracle_value_code_non_deployable": True,
        },
    ]
    axes = [
        {"axis": "value_code_selection_regret", "delta": 0.494082},
        {"axis": "sparse_formulation_gap_vs_flat_value", "delta": 1.113394},
        {"axis": "learned_sparse_gap_vs_flat_value", "delta": 0.309304},
        {"axis": "in_column_gap_vs_global_dictionary_upper_bound", "delta": 0.865296},
    ]
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_sparse_value_selection_diagnostic_recorded",
            "claim_status": "sparse_value_formulation_and_code_selection_block_gpu",
            "base_holdout_ce": 1.592115,
            "dense_teacher_holdout_ce": 1.387565,
            "oracle_value_code_non_deployable": True,
            "diagnostic_rows": rows,
            "failure_axis_rows": axes,
            "git_commit": "diagnostic-commit",
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
