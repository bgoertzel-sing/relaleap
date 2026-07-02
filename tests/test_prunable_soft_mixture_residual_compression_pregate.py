from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.prunable_soft_mixture_residual_compression_pregate import (
    DECISION,
    REQUIRED_ARTIFACTS,
    SELECTED_NEXT_ACTION,
    run_prunable_soft_mixture_residual_compression_pregate,
)


class PrunableSoftMixtureResidualCompressionPregateTests(unittest.TestCase):
    def test_records_design_pregate_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            adjudicator = root / "adjudicator.json"
            pregate = root / "pregate.json"
            review = root / "latest-review.md"
            _write_json(closeout, _closeout_payload())
            _write_json(adjudicator, {"status": "pass", "claim_status": "flat_control_blocks_no_gpu"})
            _write_json(pregate, {"status": "pass", "claim_status": "ce_mse_discordant_no_promotion"})
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Keep GPU blocked and finish local adjudication.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_prunable_soft_mixture_residual_compression_pregate(
                closeout_path=closeout,
                adjudicator_path=adjudicator,
                pregate_path=pregate,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            constraints = {row["constraint"] for row in summary["design_constraints"]}
            self.assertIn("same_objective_controls", constraints)
            self.assertIn("explicit_pruning_axis", constraints)
            arms = {row["arm"] for row in summary["pregate_arms"]}
            self.assertIn("same_objective_flat_value_control", arms)
            self.assertIn("prunable_soft_mixture_entropy_l1", arms)
            self.assertIn("pruned_soft_mixture_topr_sweep", arms)
            gates = {row["criterion"]: row for row in summary["advancement_gates"]}
            self.assertTrue(gates["closeout_selected_soft_mixture_pregate"]["passed"])
            self.assertTrue(gates["prior_gpu_validation_blocked"]["passed"])
            self.assertFalse(gates["soft_mixture_must_beat_flat_ce_same_objective"]["passed"])
            self.assertEqual(gates["soft_mixture_must_beat_flat_ce_same_objective"]["gate_type"], "future_scientific")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_when_closeout_did_not_select_soft_mixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = _closeout_payload()
            closeout["selected_next_action"] = "different_action"
            _write_json(root / "closeout.json", closeout)
            _write_json(root / "adjudicator.json", {"status": "pass"})
            _write_json(root / "pregate.json", {"status": "pass"})

            summary = run_prunable_soft_mixture_residual_compression_pregate(
                closeout_path=root / "closeout.json",
                adjudicator_path=root / "adjudicator.json",
                pregate_path=root / "pregate.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], "repair_soft_mixture_pregate_sources")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("closeout_selected_soft_mixture_pregate", failed)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _closeout_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "continuous_coefficient_branch_closed_no_gpu",
        "claim_status": "unconstrained_continuous_coefficients_retired_before_gpu",
        "selected_next_action": "design_prunable_soft_mixture_residual_compression_pregate",
        "selected_next_step": "design a local prunable soft-mixture residual compression pregate with same-objective flat/dense controls before GPU",
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
        "evidence": {
            "continuous_pregate_ce": 1.371034,
            "flat_pregate_ce": 1.410925,
            "continuous_pregate_mse": 1.230176,
            "flat_pregate_mse": 0.843461,
            "adjudicator_ce_continuous": 1.443184,
            "adjudicator_ce_flat": 1.444016,
            "adjudicator_mse_continuous": 1.262871,
            "adjudicator_mse_flat": 0.82918,
            "coeff_near_zero_fraction_min": 0.0,
            "dense_like_coefficients": True,
            "adjudicator_failed_gates": [
                "ce_objective_continuous_beats_flat_ce",
                "mse_objective_continuous_beats_flat_mse",
            ],
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
