from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.norm_budgeted_churn_regularized_residual_pilot_design import (
    REQUIRED_ARTIFACTS,
    run_norm_budgeted_churn_regularized_residual_pilot_design,
)


class NormBudgetedChurnRegularizedResidualPilotDesignTest(unittest.TestCase):
    def test_records_design_from_blocked_matched_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matched = root / "matched"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_matched_sources(matched, blocked=True)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Define a local norm-budgeted pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_norm_budgeted_churn_regularized_residual_pilot_design(
                matched_decision_dir=matched,
                strategy_review_path=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "norm_budgeted_churn_regularized_residual_pilot_design_recorded",
            )
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertAlmostEqual(summary["residual_l2_budget"], 1.004, places=3)
            arms = {row["arm"] for row in summary["pilot_arms"]}
            self.assertIn("dense_rank24_norm_budgeted", arms)
            self.assertIn("sparse_contextual_topk2_norm_budgeted", arms)
            self.assertIn("bottleneck_gated_mlp_norm_budgeted", arms)
            terms = {row["term"] for row in summary["objective_terms"]}
            self.assertIn("residual_l2_budget_penalty", terms)
            self.assertIn("residual_l2_budget_floor_penalty", terms)
            self.assertIn("prediction_flip_churn_penalty", terms)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

    def test_fails_closed_when_matched_decision_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_norm_budgeted_churn_regularized_residual_pilot_design(
                matched_decision_dir=root / "missing",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "norm_budgeted_churn_regularized_residual_pilot_design_failed_closed",
            )
            self.assertTrue(
                any(row["criterion"] == "matched_decision_passed_artifacts" for row in summary["failures"])
            )
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_already_advancing_matched_decision_does_not_design_budgeted_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matched = root / "matched"
            review = root / "latest-review.md"
            _write_matched_sources(matched, blocked=False)
            review.write_text("strategic_change_level: minor\nnotify_ben: false\n", encoding="utf-8")

            summary = run_norm_budgeted_churn_regularized_residual_pilot_design(
                matched_decision_dir=matched,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(row["criterion"] == "matched_scientific_gate_blocked" for row in summary["failures"])
            )


def _write_matched_sources(path: Path, *, blocked: bool) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail",
                "scientific_gate": "blocked" if blocked else "pass",
                "advancement_row_count": 0 if blocked else 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "available_arms.csv",
        [
            {"arm": "dense_rank24_best_norm", "heldout_residual_update_l2": 1.004},
            {"arm": "dense_rank16_best_norm", "heldout_residual_update_l2": 1.0},
            {"arm": "sparse_contextual_topk2", "heldout_residual_update_l2": 1.0},
            {"arm": "sparse_rank_matched_topk1", "heldout_residual_update_l2": 1.16},
            {"arm": "sparse_frequency_matched_random_topk1", "heldout_residual_update_l2": 1.16},
            {"arm": "parameter_matched_causal_mlp_control", "heldout_residual_update_l2": 4.44},
        ],
    )
    _write_csv(
        path / "pareto_frontier.csv",
        [{"arm": "dense_rank24_best_norm", "lambda": 1.0, "ce_loss": 3.7, "residual_update_l2": 1.004}],
    )
    _write_csv(
        path / "domination_cases.csv",
        [{"challenger_arm": "parameter_matched_causal_mlp_control", "challenger_advances": "False"}],
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
