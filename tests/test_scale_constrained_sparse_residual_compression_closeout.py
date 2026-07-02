from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.scale_constrained_sparse_residual_compression_closeout import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    SELECTED_NEXT_ACTION,
    run_scale_constrained_sparse_residual_compression_closeout,
)


class ScaleConstrainedSparseResidualCompressionCloseoutTests(unittest.TestCase):
    def test_closes_flat_dominated_scale_constrained_pilot_and_blocks_gpu(self) -> None:
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
                        "recommended_next_action: Implement a local executable scale-constrained sparse residual-compression pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_scale_constrained_sparse_residual_compression_closeout(
                pilot_path=pilot,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(
                summary["claim_status"], "scale_constrained_sparse_residual_compression_retired_before_gpu"
            )
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            evidence = summary["evidence"]
            self.assertGreater(evidence["ce_gap_sparse_minus_flat"], 0.002)
            self.assertGreater(evidence["mse_gap_sparse_minus_flat"], 0.02)
            closeout = {row["criterion"]: row for row in summary["closeout_rows"]}
            self.assertTrue(closeout["flat_ce_control_blocks_sparse"]["passed"])
            self.assertTrue(closeout["flat_mse_control_blocks_sparse"]["passed"])
            self.assertTrue(closeout["gpu_validation_blocked"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertIn("mechanism-factorized", selected[0]["next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_pilot_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_scale_constrained_sparse_residual_compression_closeout(
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
    arm_rows = []
    for objective in ("ce_only", "mse_only", "ce_mse_combined"):
        for arm in (
            "sparse_topr_norm_controller",
            "same_controller_flat_residual",
            "same_controller_dense_mlp",
            "scale_only_null",
            "shuffled_target_sparse_null",
            "random_support_sparse_null",
            "position_support_sparse_null",
        ):
            arm_rows.append(
                {
                    "objective": objective,
                    "arm": arm,
                    "ce": 1.54 if (objective, arm) == ("ce_only", "sparse_topr_norm_controller") else 1.37,
                    "teacher_residual_reconstruction_mse": 1.25
                    if (objective, arm) == ("mse_only", "sparse_topr_norm_controller")
                    else 0.98,
                    "intervention_selectivity_proxy": 0.24 if arm == "sparse_topr_norm_controller" else 0.27,
                    "finite_update_commutator_proxy": 8.58 if arm == "sparse_topr_norm_controller" else 8.93,
                    "active_component_fraction": 0.25 if arm == "sparse_topr_norm_controller" else 1.0,
                }
            )
    pruning_rows = [
        {
            "objective": objective,
            "arm": "sparse_topr_norm_controller",
            "prune_rule": rule,
            "ce_gain_retention_fraction": 0.85,
            "pruned_at_least_half_components": True,
        }
        for objective in ("ce_only", "mse_only", "ce_mse_combined")
        for rule in ("top1", "top2", "threshold_0p15", "threshold_0p25")
    ]
    return {
        "status": "pass",
        "decision": "scale_constrained_sparse_residual_compression_pilot_recorded",
        "claim_status": "scale_constrained_sparse_flat_control_blocks_gpu",
        "selected_next_step": "close or redesign scale-constrained sparse residual compression before GPU; local gates did not clear",
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": True,
        "base_holdout_ce": 1.592115,
        "dense_teacher_holdout_ce": 1.387565,
        "arm_metrics": arm_rows,
        "pruning_rows": pruning_rows,
        "gate_rows": [
            {"criterion": "sparse_matches_or_beats_flat_ce", "passed": False},
            {"criterion": "sparse_not_worse_than_flat_mse", "passed": False},
            {"criterion": "mechanism_proxy_wins_at_least_two", "passed": True},
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
