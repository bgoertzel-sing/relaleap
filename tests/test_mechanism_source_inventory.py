from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mechanism_source_inventory import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_NEXT_ACTION,
    STRATEGY_REFRESH_ACTION,
    run_mechanism_source_inventory,
)


class MechanismSourceInventoryTest(unittest.TestCase):
    def test_selects_broader_gate_and_records_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reconciliation = root / "reconciliation.json"
            _write_json(
                reconciliation,
                {
                    "status": "pass",
                    "selected_next_action": "run_local_mechanism_source_inventory_before_new_branch",
                    "decision": "post_negative_loop_reconciliation_recorded",
                    "claim_status": "negative_loop_reconciled_no_gpu_or_promotion",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )
            _write_default_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement a local low-churn MLP residual-control pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_mechanism_source_inventory(
                reconciliation_path=reconciliation,
                strategy_review_path=review,
                out_dir=root / "out",
                source_root=root,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "mechanism_source_inventory_recorded")
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(
                summary["strategy_response"]["disposition"],
                "deferred_as_already_completed",
            )
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], SELECTED_NEXT_ACTION)
            gaps = {row["gap"]: row for row in summary["evidence_gap_rows"]}
            self.assertTrue(gaps["broader_mechanism_gate_missing"]["gap_present"])
            duplicates = {row["item"]: row for row in summary["duplicate_work_rows"]}
            self.assertEqual(
                duplicates["latest_review_low_churn_recommendation"]["disposition"],
                "defer_as_completed",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "source_inventory.csv").open(newline="", encoding="utf-8") as handle:
                source_rows = list(csv.DictReader(handle))
            self.assertIn("post_negative_loop_reconciliation", {row["source"] for row in source_rows})
            with (root / "out" / "candidate_actions.csv").open(newline="", encoding="utf-8") as handle:
                action_rows = list(csv.DictReader(handle))
            self.assertIn(SELECTED_NEXT_ACTION, {row["candidate_action"] for row in action_rows})

    def test_missing_required_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_mechanism_source_inventory(
                reconciliation_path=root / "missing.json",
                out_dir=root / "out",
                source_root=root,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_completed_failed_broader_gate_selects_strategy_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reconciliation = root / "reconciliation.json"
            _write_json(
                reconciliation,
                {
                    "status": "pass",
                    "selected_next_action": "run_local_mechanism_source_inventory_before_new_branch",
                    "decision": "post_negative_loop_reconciliation_recorded",
                    "claim_status": "negative_loop_reconciled_no_gpu_or_promotion",
                },
            )
            _write_default_sources(root)
            _write_json(
                root / "results/audits/acsr_broader_mechanism_gate_local/summary.json",
                {
                    "status": "fail",
                    "decision": "acsr_broader_mechanism_gate_failed_closed",
                    "claim_status": "acsr_anticipation_specific_claim_blocked_no_default_change",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                },
            )

            summary = run_mechanism_source_inventory(
                reconciliation_path=reconciliation,
                out_dir=root / "out",
                source_root=root,
                urgent_review_status="timeout",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], STRATEGY_REFRESH_ACTION)
            self.assertEqual(summary["urgent_review_status"], "timeout")
            self.assertFalse(summary["requires_gpu_now"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], STRATEGY_REFRESH_ACTION)


def _write_default_sources(root: Path) -> None:
    sources = {
        "results/reports/core_periphery_negative_evidence_closeout/summary.json": {
            "status": "pass",
            "decision": "core_periphery_negative_evidence_closeout_branch_selected",
            "selected_next_action": "demote_current_core_periphery_mechanism_to_diagnostic_status",
            "claim_status": "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
            "requires_gpu_now": False,
        },
        "results/reports/dense_teacher_pair_composer_pregate_closeout/summary.json": {
            "status": "pass",
            "decision": "dense_teacher_pair_composer_pregate_closed_negative",
            "selected_next_action": "redirect_to_core_periphery_predictive_coding_column_design",
            "claim_status": "dense_teacher_pair_composer_negative_local_evidence_no_gpu",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        "results/reports/low_churn_mlp_residual_control_pilot/summary.json": {
            "status": "pass",
            "decision": "low_churn_mlp_residual_control_pilot_completed",
            "scientific_gate": "blocked",
            "claim_status": "low_churn_mlp_no_budgeted_advancement_claim",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        "results/reports/sparse_dense_mlp_matched_intervention_decision/summary.json": {
            "status": "pass",
            "decision": "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail",
            "scientific_gate": "blocked",
            "claim_status": "mlp_or_sparse_advantage_not_decisive_after_ce_l2_churn_matching",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
        "results/reports/acsr_common_causal_residual_benchmark/summary.json": {
            "status": "fail",
            "decision": "acsr_common_causal_residual_benchmark_failed_gate",
            "claim_status": "sparse_support_specific_effect_not_separated_from_common_dense_controls",
            "requires_gpu_now": False,
        },
        "results/reports/acsr_dense_residual_transfer_control/summary.json": {
            "status": "fail",
            "decision": "acsr_dense_residual_transfer_control_failed_gate",
            "claim_status": "sparse_transfer_not_separated_from_dense_control",
            "requires_gpu_now": False,
        },
    }
    for rel_path, payload in sources.items():
        _write_json(root / rel_path, payload)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
