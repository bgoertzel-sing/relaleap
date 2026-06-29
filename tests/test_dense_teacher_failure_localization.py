from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_failure_localization import (
    CONTRACT_RECORDED,
    INSUFFICIENT_EVIDENCE,
    REQUIRED_ARMS,
    REQUIRED_TENSORS,
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
            self.assertEqual(summary["decision"], CONTRACT_RECORDED)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["required_arms"], list(REQUIRED_ARMS))
            self.assertEqual(
                [row["tensor"] for row in summary["tensor_inventory"]],
                [str(spec["tensor"]) for spec in REQUIRED_TENSORS],
            )
            self.assertTrue(all(row["present"] for row in summary["tensor_inventory"]))
            self.assertIn("teacher_hidden_residual_mse", summary["metric_fields"])
            self.assertIn("oracle_support_trained_values", summary["selected_next_step"])
            contract_by_arm = {row["arm"]: row for row in summary["contract_rows"]}
            self.assertEqual(
                contract_by_arm["oracle_support_gated_value_pair_composer"]["availability"],
                "required_pending",
            )
            self.assertEqual(
                contract_by_arm["random_support_null"]["availability"],
                "available_from_distillation_summary",
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "tensor_inventory.csv").is_file())
            self.assertTrue((root / "out" / "pregate_rows.csv").is_file())
            self.assertTrue((root / "out" / "contract_rows.csv").is_file())
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
                {"arm": "promoted_contextual_topk2_ce_mse_distill"},
                {"arm": "random_support_topk2"},
                {"arm": "fixed_support_topk2"},
                {"arm": "token_position_only_router_topk2"},
                {"arm": "shuffled_teacher_target_topk2"},
            ],
        },
    )


def _write_required_tensors(path: Path, *, skip: set[str] | None = None) -> None:
    skipped = skip or set()
    for spec in REQUIRED_TENSORS:
        if spec["tensor"] in skipped:
            continue
        (path / str(spec["filename"])).write_bytes(str(spec["tensor"]).encode("utf-8"))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
