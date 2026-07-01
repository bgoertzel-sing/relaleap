from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.continuous_coefficient_closeout import (
    DECISION,
    FAIL_DECISION,
    REQUIRED_ARTIFACTS,
    SELECTED_NEXT_ACTION,
    run_continuous_coefficient_closeout,
)


class ContinuousCoefficientCloseoutTests(unittest.TestCase):
    def test_closes_unconstrained_continuous_coefficients_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pregate = root / "pregate.json"
            adjudicator = root / "adjudicator.json"
            review = root / "latest-review.md"
            _write_json(pregate, _pregate_payload())
            _write_json(adjudicator, _adjudicator_payload())
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Close continuous coefficients locally before GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_continuous_coefficient_closeout(
                pregate_path=pregate,
                adjudicator_path=adjudicator,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertEqual(summary["claim_status"], "unconstrained_continuous_coefficients_retired_before_gpu")
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["direction_shift"]["ben_should_be_notified"])
            evidence = summary["evidence"]
            self.assertGreater(evidence["pregate_mse_gap_vs_flat"], 0.10)
            self.assertTrue(evidence["dense_like_coefficients"])
            closeout = {row["criterion"]: row for row in summary["closeout_rows"]}
            self.assertTrue(closeout["pregate_was_ce_mse_discordant"]["passed"])
            self.assertTrue(closeout["adjudicator_ce_objective_not_promotion_safe"]["passed"])
            self.assertTrue(closeout["adjudicator_mse_objective_flat_dominates"]["passed"])
            self.assertTrue(closeout["gpu_validation_blocked"]["passed"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertIn("prunable soft-mixture", selected[0]["next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_supports_current_objective_rows_schema(self) -> None:
        payload = _adjudicator_payload()
        payload["objective_rows"] = [
            {
                "objective": "ce_only",
                "family": "continuous_coeff",
                "variant": "norm_matched",
                "ce": 1.54,
                "teacher_residual_reconstruction_mse": 3.92,
            },
            {
                "objective": "ce_only",
                "family": "same_router_flat",
                "variant": "norm_matched",
                "ce": 1.44,
                "teacher_residual_reconstruction_mse": 1.63,
            },
            {
                "objective": "mse_only",
                "family": "continuous_coeff",
                "variant": "norm_matched",
                "ce": 1.41,
                "teacher_residual_reconstruction_mse": 1.19,
            },
            {
                "objective": "mse_only",
                "family": "same_router_flat",
                "variant": "norm_matched",
                "ce": 1.42,
                "teacher_residual_reconstruction_mse": 0.77,
            },
            {
                "objective": "ce_mse_combined",
                "family": "continuous_coeff",
                "variant": "norm_matched",
                "ce": 1.32,
                "teacher_residual_reconstruction_mse": 1.96,
            },
            {
                "objective": "ce_mse_combined",
                "family": "same_router_flat",
                "variant": "norm_matched",
                "ce": 1.37,
                "teacher_residual_reconstruction_mse": 0.80,
            },
        ]
        payload.pop("adjudication_rows")
        payload["coefficient_rows"] = [{"objective": "ce_only", "coeff_near_zero_fraction": 0.015}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "pregate.json", _pregate_payload())
            _write_json(root / "adjudicator.json", payload)

            summary = run_continuous_coefficient_closeout(
                pregate_path=root / "pregate.json",
                adjudicator_path=root / "adjudicator.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertTrue(summary["evidence"]["dense_like_coefficients"])

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_continuous_coefficient_closeout(
                pregate_path=root / "missing-pregate.json",
                adjudicator_path=root / "missing-adjudicator.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAIL_DECISION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _pregate_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "continuous_coefficient_sparse_value_pregate_recorded",
        "claim_status": "continuous_coeff_ce_mse_discordant_no_promotion",
        "selected_next_step": "run continuous CE/MSE discordance adjudicator with same-objective flat controls and no GPU",
        "ce_mse_discordant": True,
        "base_holdout_ce": 1.592115,
        "dense_teacher_holdout_ce": 1.387565,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": True,
        "arm_metrics": [
            {
                "arm": "continuous_coeff_learned_support",
                "ce": 1.371034,
                "teacher_residual_reconstruction_mse": 1.230176,
                "oracle_support_non_deployable": False,
            },
            {
                "arm": "same_router_flat_value_control",
                "ce": 1.410925,
                "teacher_residual_reconstruction_mse": 0.843461,
                "oracle_support_non_deployable": False,
            },
            {
                "arm": "continuous_coeff_oracle_support_ceiling",
                "ce": 1.133843,
                "teacher_residual_reconstruction_mse": 1.231513,
                "oracle_support_non_deployable": True,
            },
        ],
        "coefficient_rows": [{"row": "aggregate", "coeff_near_zero_fraction": 0.010417}],
    }


def _adjudicator_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "continuous_coefficient_ce_mse_discordance_adjudicator_recorded",
        "claim_status": "continuous_coeff_objective_parity_flat_control_blocks_no_gpu",
        "selected_next_step": "close continuous coefficient branch or move to prunable soft-mixture compression before GPU",
        "same_objective_flat_controls_present": True,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": True,
        "adjudication_rows": [
            {
                "objective": "ce",
                "arm": "continuous_coeff_learned_support",
                "residual_variant": "norm_matched",
                "ce": 1.443184,
                "teacher_residual_reconstruction_mse": 2.941063,
                "coefficient_near_zero_fraction": 0.010417,
            },
            {
                "objective": "ce",
                "arm": "same_objective_flat_value_control",
                "residual_variant": "norm_matched",
                "ce": 1.444016,
                "teacher_residual_reconstruction_mse": 1.574379,
            },
            {
                "objective": "mse",
                "arm": "continuous_coeff_learned_support",
                "residual_variant": "norm_matched",
                "ce": 1.338516,
                "teacher_residual_reconstruction_mse": 1.262871,
                "coefficient_near_zero_fraction": 0.010417,
            },
            {
                "objective": "mse",
                "arm": "same_objective_flat_value_control",
                "residual_variant": "norm_matched",
                "ce": 1.428102,
                "teacher_residual_reconstruction_mse": 0.82918,
            },
            {
                "objective": "ce_mse",
                "arm": "continuous_coeff_learned_support",
                "residual_variant": "norm_matched",
                "ce": 1.306411,
                "teacher_residual_reconstruction_mse": 1.515312,
                "coefficient_near_zero_fraction": 0.010417,
            },
            {
                "objective": "ce_mse",
                "arm": "same_objective_flat_value_control",
                "residual_variant": "norm_matched",
                "ce": 1.286361,
                "teacher_residual_reconstruction_mse": 0.689681,
            },
        ],
        "gate_rows": [
            {"criterion": "ce_objective_continuous_beats_flat_ce", "passed": False},
            {"criterion": "mse_objective_continuous_beats_flat_mse", "passed": False},
            {"criterion": "ce_objective_coefficients_sparse_enough", "passed": False},
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
