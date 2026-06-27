from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_post_stratified_null_closeout import (
    INSUFFICIENT_EVIDENCE,
    POST_STRATIFIED_NULL_CLOSEOUT,
    run_causal_contextual_router_post_stratified_null_closeout,
)


class CausalContextualRouterPostStratifiedNullCloseoutTest(unittest.TestCase):
    def test_closes_distillation_branch_under_stronger_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: use same-student retention controls",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_post_stratified_null_closeout(
                stratified_null_dir=paths["stratified"],
                same_student_dir=paths["same_student"],
                discriminative_dir=paths["discriminative"],
                support_audit_dir=paths["support"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POST_STRATIFIED_NULL_CLOSEOUT)
            self.assertEqual(
                summary["claim_statuses"]["causal_router_distillation"],
                "closed_not_functionally_established",
            )
            self.assertEqual(
                summary["selected_next_step"],
                "return_to_non_distillation_architecture_loop_with_causal_router_as_ce_baseline",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "closure_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_stratified_reversal_gate_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            _write_json(
                paths["stratified"] / "summary.json",
                {
                    "status": "pass",
                    "decision": "prior_distillation_mechanism_claim_superseded_by_stratified_null",
                    "gate_status": {"passes_reversal_gate": False},
                },
            )

            summary = run_causal_contextual_router_post_stratified_null_closeout(
                stratified_null_dir=paths["stratified"],
                same_student_dir=paths["same_student"],
                discriminative_dir=paths["discriminative"],
                support_audit_dir=paths["support"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                ("stratified_null_reversal", "passes_reversal_gate"),
                {
                    (failure.get("source"), failure.get("field"))
                    for failure in summary["failures"]
                },
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "stratified": root / "stratified",
        "same_student": root / "same_student",
        "discriminative": root / "discriminative",
        "support": root / "support",
    }
    for path in paths.values():
        path.mkdir()
    _write_json(
        paths["stratified"] / "summary.json",
        {
            "status": "pass",
            "decision": "prior_distillation_mechanism_claim_superseded_by_stratified_null",
            "gate_status": {"passes_reversal_gate": True},
            "seed_rows": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
        },
    )
    _write_json(
        paths["same_student"] / "summary.json",
        {
            "status": "pass",
            "decision": "same_student_token_position_null_discriminator_blocks_claim",
            "key_metrics": {
                "teacher_minus_token_position_null_gain_all_tokens": 0.001,
                "teacher_forced_gain_all_tokens": -0.02,
            },
        },
    )
    _write_json(
        paths["discriminative"] / "summary.json",
        {
            "status": "pass",
            "decision": "real_teacher_distilled_causal_router_preferred_over_control_family",
            "claim_status": "distilled_causal_router_discriminative_mechanism_supported_not_promoted",
        },
    )
    _write_json(
        paths["support"] / "summary.json",
        {
            "status": "pass",
            "decision": "causal_contextual_router_support_audit_blocks_promotion",
            "audit": {
                "aggregate_metrics": {
                    "causal_contextual_topk2": {
                        "mean_router_loss_delta_vs_linear": -0.5,
                        "mean_oracle_regret_delta_vs_linear": 0.06,
                        "mean_functional_churn_delta_vs_linear": 0.11,
                        "mean_oracle_support_regret": 0.08,
                    },
                    "linear_topk2": {
                        "mean_oracle_support_regret": 0.02,
                    },
                }
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
