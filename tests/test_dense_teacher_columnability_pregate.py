from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_columnability_pregate import (
    REQUIRED_ARTIFACTS,
    run_dense_teacher_columnability_pregate,
)


class DenseTeacherColumnabilityPregateTest(unittest.TestCase):
    def test_complete_fixture_passes_local_pregate(self) -> None:
        torch = _require_torch(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            branch = root / "branch.json"
            distillation = root / "distillation"
            review = root / "latest-review.md"
            _write_branch_inventory(branch)
            _write_strategy_review(review)
            _write_distillation_fixture(torch, distillation, pass_gate=True)

            summary = run_dense_teacher_columnability_pregate(
                branch_inventory_path=branch,
                distillation_dir=distillation,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "ready_for_local_columnability_validation")
            self.assertEqual(
                summary["decision"],
                "dense_teacher_columnability_pregate_ready_for_local_validation",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for row in summary["target_tensors"]:
                self.assertTrue(row["present"], row)
                self.assertTrue(row["loadable"], row)
                self.assertTrue(row["finite"], row)
            norm_budget = {
                row["arm"]: row for row in summary["arm_rows"]
            }["norm_budgeted_promoted_contextual_topk2_ce_mse_distill"]
            self.assertGreaterEqual(norm_budget["residual_norm_ratio"], 0.2)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_negative_fixture_blocks_without_gpu(self) -> None:
        torch = _require_torch(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            branch = root / "branch.json"
            distillation = root / "distillation"
            review = root / "latest-review.md"
            _write_branch_inventory(branch)
            _write_strategy_review(review)
            _write_distillation_fixture(torch, distillation, pass_gate=False)

            summary = run_dense_teacher_columnability_pregate(
                branch_inventory_path=branch,
                distillation_dir=distillation,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(summary["decision"], "dense_teacher_columnability_pregate_blocked")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("sparse_columns_reconstruct_teacher_residual", failed)
            self.assertIn("norm_budget_accounting_present", failed)
            self.assertIn("learned_router_beats_null_targets", failed)
            self.assertIn("interference_budget_plausible", failed)
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_missing_branch_inventory_fails_closed(self) -> None:
        torch = _require_torch(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            distillation = root / "distillation"
            review = root / "latest-review.md"
            _write_strategy_review(review)
            _write_distillation_fixture(torch, distillation, pass_gate=True)

            summary = run_dense_teacher_columnability_pregate(
                branch_inventory_path=root / "missing.json",
                distillation_dir=distillation,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["scientific_gate"], "blocked")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("branch_inventory_selected_pregate", failed)


def _require_torch(testcase: unittest.TestCase):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        testcase.skipTest(f"torch unavailable: {exc}")
    return torch


def _write_branch_inventory(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "mechanism_branch_inventory_recorded",
            "selected_next_action": "start_dense_teacher_columnability_pregate_before_gpu",
        },
    )


def _write_strategy_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: major",
                "notify_ben: true",
                "recommended_next_action: Start a local dense-teacher columnability/continual-interference pregate with matched nulls.",
                "verdict: PIVOT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_distillation_fixture(torch, path: Path, *, pass_gate: bool) -> None:
    path.mkdir(parents=True)
    base_hidden = torch.ones(2, 3, 4)
    teacher_hidden_residual = torch.ones(2, 3, 4) * 0.5
    base_logits = torch.ones(2, 3, 5)
    teacher_logit_residual = torch.ones(2, 3, 5) * 0.25
    torch.save(base_hidden, path / "base_hidden.pt")
    torch.save(teacher_hidden_residual, path / "teacher_hidden_residual.pt")
    torch.save(base_logits, path / "base_logits.pt")
    torch.save(teacher_logit_residual, path / "teacher_logit_residual.pt")

    if pass_gate:
        primary_r2 = 0.35
        primary_mse = 0.40
        norm_ratio = 0.55
        churn = 0.20
    else:
        primary_r2 = 0.06
        primary_mse = 0.90
        norm_ratio = 0.05
        churn = 0.94

    variant_rows = [
        {
            "arm": "parameter_matched_causal_mlp_control",
            "teacher_scale": 1.0,
            "ce_loss": 1.0,
            "teacher_residual_r2": 1.0,
            "teacher_residual_mse": 0.0,
            "residual_norm_ratio": 1.0,
            "functional_churn": 0.0,
            "commutator_norm": 0.0,
        },
        _student_row(
            "promoted_contextual_topk2_ce_mse_distill",
            primary_r2,
            primary_mse,
            norm_ratio,
            churn,
        ),
        _student_row("promoted_contextual_topk2_mse_only_distill", primary_r2 - 0.02, primary_mse + 0.04, norm_ratio, churn),
        _student_row(
            "norm_budgeted_promoted_contextual_topk2_ce_mse_distill",
            primary_r2 - 0.01,
            primary_mse + 0.02,
            norm_ratio,
            churn,
            include_budget=True,
        ),
        _student_row("rank_matched_contextual_topk1", 0.22 if pass_gate else 0.04, 0.55, 0.40, 0.35),
    ]
    null_mse = 0.75 if pass_gate else 0.70
    for arm in (
        "token_position_only_router_topk2",
        "random_support_topk2",
        "fixed_support_topk2",
        "shuffled_feature_router_topk2",
        "shuffled_teacher_target_topk2",
    ):
        variant_rows.append(_student_row(arm, 0.01, null_mse, 0.25, 0.25))
        null_mse += 0.01

    _write_json(
        path / "summary.json",
        {
            "status": "fail",
            "decision": "dense_teacher_residual_distillation_pilot_not_supported",
            "base_ce_loss": 4.0,
            "dense_teacher_ce_loss": 1.0,
            "dense_teacher_ce_improvement": 3.0,
            "variant_rows": variant_rows,
        },
    )


def _student_row(
    arm: str,
    r2: float,
    mse: float,
    norm_ratio: float,
    churn: float,
    *,
    include_budget: bool = False,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "arm": arm,
        "teacher_scale": 1.0,
        "ce_loss": 2.0,
        "teacher_residual_r2": r2,
        "teacher_residual_mse": mse,
        "teacher_logit_mse": mse,
        "residual_norm_ratio": norm_ratio,
        "functional_churn": churn,
        "commutator_norm": 0.02,
        "support_regret": 0.01,
        "intervention_fingerprint_purity": 0.95,
    }
    if include_budget:
        row["residual_norm_budget"] = 1.0
        row["residual_norm_budget_error"] = 0.05
        row["residual_norm_budget_overuse"] = 0.0
    return row


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
