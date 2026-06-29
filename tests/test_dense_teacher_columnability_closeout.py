from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_columnability_closeout import (
    CLOSEOUT_DECISION,
    INSUFFICIENT_EVIDENCE_DECISION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_columnability_closeout,
)


class DenseTeacherColumnabilityCloseoutTest(unittest.TestCase):
    def test_closes_branch_and_selects_failure_localization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_primary = root / "dense_primary"
            distillation = root / "distillation"
            gate = root / "gate"
            review = root / "latest-review.md"
            _write_dense_primary(dense_primary)
            _write_distillation(distillation)
            _write_gate(gate)
            _write_review(review, major=True)

            summary = run_dense_teacher_columnability_closeout(
                dense_primary_dir=dense_primary,
                distillation_dir=distillation,
                gate_dir=gate,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], CLOSEOUT_DECISION)
            self.assertEqual(
                summary["claim_status"],
                "dense_teacher_sparse_columnability_not_established_current_branch_retired",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            self.assertEqual(summary["direction_shift"]["recommendation_disposition"], "accepted")
            self.assertIn("dense_teacher_failure_localization.py", summary["selected_next_step"])
            criteria = {row["criterion"]: row for row in summary["closeout_criteria"]}
            self.assertTrue(criteria["distillation_gate_failed_closed"]["passed"])
            self.assertTrue(criteria["norm_budget_rescue_not_supported"]["passed"])
            self.assertEqual(summary["evidence"]["dense_primary_git_commit"], "dense-primary-commit")
            self.assertEqual(summary["evidence"]["distillation_git_commit"], "distillation-commit")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_distillation_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_primary = root / "dense_primary"
            gate = root / "gate"
            _write_dense_primary(dense_primary)
            _write_gate(gate)

            summary = run_dense_teacher_columnability_closeout(
                dense_primary_dir=dense_primary,
                distillation_dir=root / "missing_distillation",
                gate_dir=gate,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE_DECISION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_dense_primary(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_primary_mechanism_assay_selected",
            "claim_status": "dense_or_mlp_control_selected_as_primary_mechanism_assay",
            "primary_arm": "parameter_matched_causal_mlp_control",
            "git_commit": "dense-primary-commit",
        },
    )


def _write_distillation(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "fail",
            "decision": "dense_teacher_residual_distillation_pilot_not_supported",
            "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
            "git_commit": "distillation-commit",
            "base_ce_loss": 4.0,
            "dense_teacher_ce_loss": 0.5,
            "gate_status": {"passes_dense_teacher_distillation_gate": False},
            "failures": [
                {"criterion": "acsr_beats_token_position_and_shuffled_distillation_nulls"},
                {"criterion": "calibrated_teacher_scale_gate"},
            ],
            "variant_rows": [
                {
                    "arm": "acsr_predicted_future_support",
                    "teacher_scale": 1.0,
                    "ce_loss": 2.8,
                    "teacher_logit_mse": 10.4,
                },
                {
                    "arm": "promoted_contextual_topk2_ce_mse_distill",
                    "teacher_scale": 1.0,
                    "ce_loss": 2.84,
                    "teacher_logit_mse": 10.47,
                },
                {
                    "arm": "promoted_contextual_topk2_mse_only_distill",
                    "teacher_scale": 1.0,
                    "ce_loss": 2.97,
                },
                {
                    "arm": "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
                    "teacher_scale": 1.0,
                    "ce_loss": 3.70,
                    "teacher_logit_mse": 12.2,
                    "residual_norm_ratio": 0.17,
                },
            ],
        },
    )


def _write_gate(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "dense_teacher_columnability_scaffold_ready_for_local_validation",
            "scientific_gate": "ready_for_local_validation",
            "git_commit": "gate-commit",
        },
    )


def _write_review(path: Path, *, major: bool) -> None:
    path.write_text(
        "\n".join(
            [
                f"strategic_change_level: {'major' if major else 'minor'}",
                f"notify_ben: {'true' if major else 'false'}",
                "recommended_next_action: Close out branch and run failure-localization audit.",
                "verdict: PIVOT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
