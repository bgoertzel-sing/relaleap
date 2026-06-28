from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_columnability_gate import (
    REQUIRED_ARTIFACTS,
    run_dense_teacher_columnability_gate,
)


class DenseTeacherColumnabilityGateTest(unittest.TestCase):
    def test_missing_contract_blocks_gpu_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_primary = root / "dense_primary"
            distillation = root / "distillation"
            review = root / "latest-review.md"
            _write_dense_primary(dense_primary, complete=False)
            _write_distillation(distillation, complete=False)
            _write_review(review)

            summary = run_dense_teacher_columnability_gate(
                dense_primary_dir=dense_primary,
                distillation_dir=distillation,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(
                summary["decision"],
                "dense_teacher_columnability_scaffold_blocked_missing_contract",
            )
            self.assertFalse(summary["requires_gpu_now"])
            criteria = {row["criterion"]: row for row in summary["criteria"]}
            self.assertTrue(criteria["dense_primary_selected_parameter_matched_mlp_teacher"]["passed"])
            self.assertFalse(criteria["teacher_residual_exports_available"]["passed"])
            self.assertFalse(criteria["required_sparse_student_and_null_arms_declared"]["passed"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_complete_synthetic_contract_is_ready_for_local_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_primary = root / "dense_primary"
            distillation = root / "distillation"
            review = root / "latest-review.md"
            _write_dense_primary(dense_primary, complete=True)
            _write_distillation(distillation, complete=True)
            _write_review(review)

            summary = run_dense_teacher_columnability_gate(
                dense_primary_dir=dense_primary,
                distillation_dir=distillation,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["scientific_gate"], "ready_for_local_validation")
            self.assertEqual(
                summary["decision"],
                "dense_teacher_columnability_scaffold_ready_for_local_validation",
            )
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])


def _write_dense_primary(path: Path, *, complete: bool) -> None:
    path.mkdir(parents=True)
    teacher = _complete_row("parameter_matched_causal_mlp_control") if complete else {
        "arm": "parameter_matched_causal_mlp_control",
        "active_params": 113078,
        "active_rank_or_topk": 2,
        "residual_l2": 4.4,
        "ce_loss": 2.87,
        "anchor_kl_or_logit_mse": 0.15,
        "functional_churn": 0.75,
        "intervention_fingerprint_purity": 1.0,
    }
    scorecard = [teacher]
    if complete:
        scorecard.extend(
            _complete_row(arm)
            for arm in (
                "dense_teacher_parameter_matched_mlp",
                "dense_rank_norm_control",
                "rank_matched_contextual_topk1",
                "random_support_topk2",
                "fixed_support_topk2",
                "shuffled_teacher_target_topk2",
            )
        )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "dense_primary_mechanism_assay_selected",
                "primary_arm": "parameter_matched_causal_mlp_control",
                "candidate_scorecard": scorecard,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_distillation(path: Path, *, complete: bool) -> None:
    path.mkdir(parents=True)
    rows = []
    if complete:
        rows = [
            _complete_row("promoted_contextual_topk2_ce_mse_distill", variant="promoted_contextual_topk2_ce_mse_distill"),
            _complete_row("promoted_contextual_topk2_mse_only_distill", variant="promoted_contextual_topk2_mse_only_distill"),
            _complete_row("token_position_only_router_topk2", variant="token_position_only_router_topk2"),
            _complete_row("shuffled_feature_router_topk2", variant="shuffled_feature_router_topk2"),
        ]
    else:
        rows = [
            {"variant": "promoted_contextual_router_support", "ce_loss": 2.84, "teacher_logit_mse": 10.4},
            {"variant": "token_position_only_predicted_support", "ce_loss": 2.93, "teacher_logit_mse": 10.5},
        ]
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                "variant_rows": rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _complete_row(arm: str, *, variant: str | None = None) -> dict[str, object]:
    row = {
        "arm": arm,
        "stored_params": 100,
        "active_params": 50,
        "active_rank_or_topk": 2,
        "residual_l2": 1.0,
        "residual_norm_ratio": 1.0,
        "flops_estimate": 200,
        "ce_loss": 2.0,
        "anchor_kl_or_logit_mse": 0.01,
        "functional_churn": 0.1,
        "intervention_fingerprint_purity": 1.0,
        "support_regret": 0.02,
        "commutator_norm": 0.03,
        "teacher_hidden_residual_export": "teacher_hidden_residual.pt",
        "teacher_logit_residual_export": "teacher_logit_residual.pt",
        "teacher_residual_mse": 0.1,
        "teacher_residual_r2": 0.8,
        "teacher_residual_cosine": 0.9,
    }
    if variant is not None:
        row["variant"] = variant
    return row


def _write_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: Convert dense MLP branch into a columnability audit.",
                "verdict: FIX",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
