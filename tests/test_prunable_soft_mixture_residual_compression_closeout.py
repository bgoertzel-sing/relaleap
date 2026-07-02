from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.prunable_soft_mixture_residual_compression_closeout import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    SELECTED_NEXT_ACTION,
    run_prunable_soft_mixture_residual_compression_closeout,
)


class PrunableSoftMixtureResidualCompressionCloseoutTests(unittest.TestCase):
    def test_closes_flat_dominated_soft_mixture_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = root / "pilot.json"
            review = root / "latest-review.md"
            _write_json(pilot, _pilot_payload())
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: keep continuous/soft-mixture claims non-promotional",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_prunable_soft_mixture_residual_compression_closeout(
                pilot_path=pilot,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(summary["claim_status"], "prunable_soft_mixture_retired_before_gpu")
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            evidence = summary["evidence"]
            self.assertGreater(evidence["ce_gap_soft_minus_flat"], 0.002)
            self.assertGreater(evidence["mse_gap_soft_minus_flat"], 0.02)
            self.assertEqual(evidence["best_half_prune_ce_gain_retention"], 0.0)
            closeout = {row["criterion"]: row for row in summary["closeout_rows"]}
            self.assertTrue(closeout["flat_ce_control_blocks_soft_mixture"]["passed"])
            self.assertTrue(closeout["flat_mse_control_blocks_soft_mixture"]["passed"])
            self.assertTrue(closeout["pruning_retention_failed"]["passed"])
            self.assertTrue(closeout["gpu_validation_blocked"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertIn("scale-constrained", selected[0]["next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_pilot_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_prunable_soft_mixture_residual_compression_closeout(
                pilot_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _pilot_payload() -> dict[str, object]:
    objective_rows = []
    for objective in ("ce_only", "mse_only", "ce_mse_combined"):
        for family in (
            "same_objective_flat",
            "soft_mixture_unpruned",
            "prunable_soft_mixture_entropy_l1",
            "scale_only_residual_null",
            "shuffled_target_soft_mixture_null",
        ):
            for variant in ("raw", "norm_matched"):
                objective_rows.append(
                    {
                        "objective": objective,
                        "family": family,
                        "variant": variant,
                        "ce": 1.98 if (objective, family, variant) == ("ce_only", "prunable_soft_mixture_entropy_l1", "norm_matched") else 1.43,
                        "teacher_residual_reconstruction_mse": 1.47
                        if (objective, family, variant) == ("mse_only", "prunable_soft_mixture_entropy_l1", "norm_matched")
                        else 0.73,
                        "intervention_selectivity_proxy": 0.23 if family == "prunable_soft_mixture_entropy_l1" else 0.31,
                        "finite_update_commutator_proxy": 8.8 if family == "prunable_soft_mixture_entropy_l1" else 10.2,
                    }
                )
    mixture_rows = [
        {
            "objective": objective,
            "family": family,
            "weight_near_zero_fraction": 0.80,
            "effective_component_count_mean": 1.4,
        }
        for objective in ("ce_only", "mse_only", "ce_mse_combined")
        for family in ("soft_mixture_unpruned", "prunable_soft_mixture_entropy_l1", "shuffled_target_soft_mixture_null")
    ]
    pruning_rows = [
        {
            "objective": objective,
            "family": family,
            "prune_rule": rule,
            "ce_gain_retention_fraction": 0.0,
            "pruned_at_least_half_components": True,
        }
        for objective in ("ce_only", "mse_only", "ce_mse_combined")
        for family in ("soft_mixture_unpruned", "prunable_soft_mixture_entropy_l1")
        for rule in ("top1", "top2", "top4", "threshold_0p10", "threshold_0p20")
    ]
    return {
        "status": "pass",
        "decision": "prunable_soft_mixture_residual_compression_pilot_recorded",
        "claim_status": "prunable_soft_mixture_flat_ce_control_blocks_gpu",
        "selected_next_step": "close or redesign soft-mixture compression before GPU; same-objective flat controls still dominate",
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": True,
        "base_holdout_ce": 1.592115,
        "dense_teacher_holdout_ce": 1.387565,
        "objective_rows": objective_rows,
        "mixture_rows": mixture_rows,
        "pruning_rows": pruning_rows,
        "gate_rows": [
            {"criterion": "soft_mixture_beats_flat_ce_same_objective", "passed": False},
            {"criterion": "soft_mixture_not_worse_than_flat_mse", "passed": False},
            {"criterion": "pruning_retains_function_after_halving", "passed": False},
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
