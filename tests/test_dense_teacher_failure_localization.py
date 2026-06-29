from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_failure_localization import (
    CONTRACT_RECORDED,
    EVALUATOR_RECORDED,
    INSUFFICIENT_EVIDENCE,
    PARTIAL_EVALUATOR_RECORDED,
    REQUIRED_ARMS,
    REQUIRED_TENSORS,
    DECODER_EXPORTED_PREGATE_NEXT_STEP,
    run_dense_teacher_failure_localization_contract,
)


class DenseTeacherFailureLocalizationContractTest(unittest.TestCase):
    def test_records_failure_localization_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout"
            distillation = root / "distillation"
            _write_closeout(closeout)
            _write_distillation(distillation)
            _write_required_tensors(distillation)

            summary = run_dense_teacher_failure_localization_contract(
                closeout_dir=closeout,
                distillation_dir=distillation,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], EVALUATOR_RECORDED)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["no_gpu_pregate_status"], "fail")
            self.assertFalse(summary["composer_train_holdout_split_recorded"])
            self.assertFalse(summary["composer_uses_true_frozen_decoder_for_ce"])
            self.assertEqual(summary["composer_ce_metric_path"], "linearized_logit_surrogate")
            self.assertEqual(summary["selected_next_step"], DECODER_EXPORTED_PREGATE_NEXT_STEP)
            self.assertEqual(summary["required_arms"], list(REQUIRED_ARMS))
            self.assertEqual(
                [row["tensor"] for row in summary["tensor_inventory"]],
                [str(spec["tensor"]) for spec in REQUIRED_TENSORS],
            )
            self.assertTrue(all(row["present"] for row in summary["tensor_inventory"]))
            self.assertIn("teacher_hidden_residual_mse", summary["metric_fields"])
            self.assertEqual(
                summary["filled_evaluator_arms"],
                [
                    "learned_support_sparse_student",
                    "oracle_support_trained_values",
                    "retrained_oracle_support_values",
                    "oracle_support_gated_value_pair_composer",
                    "dense_teacher",
                    "dense_rank_norm_control",
                    "random_support_null",
                    "fixed_support_null",
                    "token_position_router_null",
                    "shuffled_teacher_target_null",
                ],
            )
            self.assertEqual(summary["pending_evaluator_arms"], [])
            self.assertNotIn("retrained_oracle_support_values", summary["pending_evaluator_arms"])
            self.assertNotIn("dense_teacher", summary["pending_evaluator_arms"])
            evaluator_by_arm = {row["arm"]: row for row in summary["evaluator_rows"]}
            no_gpu_by_criterion = {
                row["criterion"]: row for row in summary["no_gpu_pregate_rows"]
            }
            self.assertFalse(
                no_gpu_by_criterion["composer_train_holdout_split_recorded"]["passed"]
            )
            self.assertFalse(
                no_gpu_by_criterion["composer_uses_true_frozen_decoder_for_ce"]["passed"]
            )
            self.assertTrue(no_gpu_by_criterion["composer_surrogate_caveat_recorded"]["passed"])
            self.assertEqual(
                evaluator_by_arm["learned_support_sparse_student"]["availability"],
                "filled",
            )
            self.assertEqual(
                evaluator_by_arm["oracle_support_trained_values"]["availability"],
                "filled",
            )
            self.assertEqual(
                evaluator_by_arm["retrained_oracle_support_values"]["availability"],
                "filled",
            )
            self.assertEqual(
                evaluator_by_arm["oracle_support_gated_value_pair_composer"]["availability"],
                "filled",
            )
            self.assertLessEqual(
                evaluator_by_arm["oracle_support_trained_values"]["teacher_logit_residual_mse"],
                evaluator_by_arm["learned_support_sparse_student"]["teacher_logit_residual_mse"],
            )
            self.assertLessEqual(
                evaluator_by_arm["retrained_oracle_support_values"]["teacher_hidden_residual_mse"],
                evaluator_by_arm["learned_support_sparse_student"]["teacher_hidden_residual_mse"],
            )
            self.assertLessEqual(
                evaluator_by_arm["oracle_support_gated_value_pair_composer"][
                    "teacher_logit_residual_mse"
                ],
                evaluator_by_arm["retrained_oracle_support_values"]["teacher_logit_residual_mse"],
            )
            self.assertEqual(evaluator_by_arm["random_support_null"]["availability"], "filled")
            self.assertEqual(evaluator_by_arm["random_support_null"]["ce_loss"], 4.0)
            self.assertEqual(evaluator_by_arm["dense_teacher"]["ce_loss"], 0.25)
            contract_by_arm = {row["arm"]: row for row in summary["contract_rows"]}
            self.assertEqual(
                contract_by_arm["oracle_support_gated_value_pair_composer"]["availability"],
                "implemented_local_evaluator",
            )
            self.assertEqual(
                contract_by_arm["retrained_oracle_support_values"]["availability"],
                "implemented_local_evaluator",
            )
            self.assertEqual(
                contract_by_arm["random_support_null"]["availability"],
                "available_from_distillation_summary",
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "tensor_inventory.csv").is_file())
            self.assertTrue((root / "out" / "pregate_rows.csv").is_file())
            self.assertTrue((root / "out" / "no_gpu_pregate_rows.csv").is_file())
            self.assertTrue((root / "out" / "contract_rows.csv").is_file())
            self.assertTrue((root / "out" / "evaluator_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_fails_closed_without_retired_closeout_and_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout"
            distillation = root / "distillation"
            _write_closeout(closeout, decision="not_retired")
            _write_distillation(distillation)

            summary = run_dense_teacher_failure_localization_contract(
                closeout_dir=closeout,
                distillation_dir=distillation,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("closeout_branch_retired", failed)
            self.assertIn("teacher_residual_tensors_present", failed)
            self.assertIn("per_column_evaluator_tensors_present", failed)
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_fails_closed_when_per_column_exports_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout"
            distillation = root / "distillation"
            _write_closeout(closeout)
            _write_distillation(distillation)
            _write_required_tensors(
                distillation,
                skip={"per_column_hidden_contributions", "per_column_logit_contributions"},
            )

            summary = run_dense_teacher_failure_localization_contract(
                closeout_dir=closeout,
                distillation_dir=distillation,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("per_column_evaluator_tensors_present", failed)
            inventory = {row["tensor"]: row for row in summary["tensor_inventory"]}
            self.assertEqual(
                inventory["per_column_hidden_contributions"]["status"],
                "missing_required_export",
            )
            self.assertEqual(
                inventory["per_column_logit_contributions"]["status"],
                "missing_required_export",
            )


def _write_closeout(path: Path, *, decision: str | None = None) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": decision
            or "dense_teacher_sparse_columnability_branch_closed_for_failure_localization",
            "claim_status": "dense_teacher_sparse_columnability_not_established_current_branch_retired",
            "git_commit": "closeout-commit",
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
            "variant_rows": [
                _variant_row("promoted_contextual_topk2_ce_mse_distill", ce_loss=2.5),
                _variant_row("parameter_matched_causal_mlp_control", ce_loss=0.25),
                _variant_row("dense_rank_norm_control", ce_loss=0.5),
                _variant_row("random_support_topk2", ce_loss=4.0),
                _variant_row("fixed_support_topk2", ce_loss=4.1),
                _variant_row("token_position_only_router_topk2", ce_loss=2.7),
                _variant_row("shuffled_teacher_target_topk2", ce_loss=4.2),
            ],
        },
    )


def _variant_row(arm: str, *, ce_loss: float) -> dict[str, float | str]:
    return {
        "arm": arm,
        "teacher_scale": 1.0,
        "active_params": 10.0,
        "stored_params": 20.0,
        "flops_estimate": 30.0,
        "ce_loss": ce_loss,
        "teacher_ce_loss": 0.25,
        "teacher_residual_mse": ce_loss + 1.0,
        "teacher_residual_r2": 0.1,
        "teacher_residual_cosine": 0.2,
        "teacher_logit_mse": ce_loss + 2.0,
        "support_regret": ce_loss - 0.25,
        "functional_churn": 0.3,
        "anchor_kl_or_logit_mse": 0.4,
        "residual_norm_ratio": 0.5,
    }


def _write_required_tensors(path: Path, *, skip: set[str] | None = None) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc

    skipped = skip or set()
    inputs = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 0, 1]], dtype=torch.long)
    base_hidden = torch.zeros(1, 4, 2)
    base_logits = torch.zeros(1, 4, 3)
    teacher_logit_residual = torch.zeros(1, 4, 3)
    teacher_logit_residual[:, :, 0] = 1.0
    teacher_hidden_residual = torch.zeros(1, 4, 2)
    teacher_hidden_residual[:, :, 0] = 1.0
    teacher_logits = base_logits + teacher_logit_residual
    learned_support = torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=torch.long)
    learned_scores = torch.zeros(1, 4, 2)
    per_column_hidden = torch.zeros(1, 4, 3, 2)
    per_column_hidden[:, :, 0, 0] = 2.0
    per_column_hidden[:, :, 1, 0] = 0.5
    per_column_hidden[:, :, 2, 1] = 0.5
    per_column_logits = torch.zeros(1, 4, 3, 3)
    per_column_logits[:, :, 0, 0] = 2.0
    per_column_logits[:, :, 1, 0] = 0.25
    per_column_logits[:, :, 2, 1] = 0.25
    values = {
        "inputs": inputs,
        "targets": targets,
        "base_hidden": base_hidden,
        "base_logits": base_logits,
        "teacher_logits": teacher_logits,
        "teacher_hidden_residual": teacher_hidden_residual,
        "teacher_logit_residual": teacher_logit_residual,
        "learned_support_indices": learned_support,
        "learned_support_scores": learned_scores,
        "per_column_hidden_contributions": per_column_hidden,
        "per_column_logit_contributions": per_column_logits,
        "sparse_column_value_state": {
            "top_k": 2,
            "num_columns": 3,
            "atoms_per_column": 1,
            "atom_logits": torch.zeros(3, 1),
            "atom_values": torch.zeros(3, 1, 2),
            "column_values": torch.zeros(3, 2),
            "support_router": "contextual_mlp",
        },
    }
    for spec in REQUIRED_TENSORS:
        if spec["tensor"] in skipped:
            continue
        torch.save(values[str(spec["tensor"])], path / str(spec["filename"]))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
