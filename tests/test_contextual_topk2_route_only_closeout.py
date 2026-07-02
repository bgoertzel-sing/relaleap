from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_topk2_route_only_closeout import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_contextual_topk2_route_only_closeout,
)


class ContextualTopk2RouteOnlyCloseoutTests(unittest.TestCase):
    def test_closes_route_only_branch_after_generated_candidate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate.json"
            selector = root / "selector.json"
            review = root / "latest-review.md"
            _write_json(pregate, _pregate_payload())
            _write_json(
                selector,
                {
                    "status": "pass",
                    "claim_status": "dense_mlp_mechanism_track_selected_no_gpu_or_default_change",
                    "selected_next_action": "continue_dense_mlp_mechanism_track_with_causal_router_diagnostics",
                    "next_step": "run a bounded local dense/MLP mechanism follow-up before any GPU validation",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement one local deployable support-candidate-generation pregate with a predeclared closeout if it cannot beat linear.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_contextual_topk2_route_only_closeout(
                pregate_path=pregate,
                branch_selector_path=selector,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "contextual_topk2_route_only_branch_closed")
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(
                summary["claim_status"],
                "contextual_topk2_route_only_redesign_closed_no_gpu",
            )
            self.assertIn(
                "generated_candidate_fails_regret_churn_null_same_student",
                {row["signal"] for row in summary["failure_matrix"] if row["passed"]},
            )
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], CLOSE_ACTION)
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_pregate_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_contextual_topk2_route_only_closeout(
                pregate_path=root / "missing-pregate.json",
                branch_selector_path=root / "missing-selector.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _pregate_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "contextual_topk2_support_quality_pregate_pilot_recorded",
        "claim_status": "route_only_contextual_topk2_support_quality_gate_failed_no_gpu",
        "selected_next_action": "record_contextual_topk2_route_only_closeout_no_gpu",
        "selected_next_step": "record a contextual top-k-2 route-only closeout and redirect the architecture loop to a non-router mechanism branch",
        "training_executed": True,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "gate_criteria": [
            {"criterion": "all_pair_one_swap_candidate_loss_coverage_present", "passed": False},
            {"criterion": "deployable_generated_candidate_beats_shuffled_control", "passed": False},
        ],
        "evidence": {
            "backend_summaries": {
                "local": {
                    "deployable_candidate_generation_present": True,
                    "all_pair_one_swap_candidate_loss_coverage_present": False,
                    "generated_candidate_accepted_one_swap_fraction": 0.0,
                    "generated_candidate_minus_linear_oracle_regret": -0.0000000002,
                    "generated_candidate_p90_minus_linear_oracle_regret": 0.02,
                    "generated_candidate_minus_linear_support_churn_proxy": 0.002,
                    "generated_candidate_loss_minus_shuffled_label_control": 0.24,
                    "generated_candidate_same_student_forced_regret_delta_vs_linear": 0.0,
                    "trained_pair_quality_candidate_accepted_one_swap_fraction": 0.0,
                }
            }
        },
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
