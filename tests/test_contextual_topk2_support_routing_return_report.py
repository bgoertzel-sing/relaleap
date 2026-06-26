from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_topk2_support_routing_return_report import (
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    SUPPORT_ROUTING_RETURN_SELECTED,
    run_contextual_topk2_support_routing_return_report,
)
from relaleap.experiments.causal_audit_coverage_report import (
    RANK_MATCHED_TOPK1_ACTIVE_POST_STOP,
)
from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)


class ContextualTopk2SupportRoutingReturnReportTest(unittest.TestCase):
    def test_selects_contextual_router_shortcut_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement the contextual top-k-2 support-routing return report",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_contextual_topk2_support_routing_return_report(
                promotion_report_path=paths["promotion"],
                post_promotion_report_path=paths["post_promotion"],
                coverage_report_path=paths["coverage"],
                finite_update_report_path=paths["finite_update"],
                gate_suppression_audit_path=paths["gate_suppression"],
                post_stop_report_path=paths["post_stop"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SUPPORT_ROUTING_RETURN_SELECTED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertIn("contextual_router_shortcut_ablation", summary["next_command"])
            self.assertEqual(
                summary["claim_statuses"]["topk1_singleton_reuse"],
                "diagnostic_only_not_deployable",
            )
            self.assertEqual(
                summary["claim_statuses"]["topk2_causal_cooperation"],
                "blocked_pending_new_identified_controls",
            )
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(
                dispositions["matched_causal_control_intervention_matrix"],
                "already_satisfied_or_superseded",
            )
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "minor")
            self.assertFalse(summary["strategy_review"]["notify_ben"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_post_promotion_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["post_promotion"].unlink()

            summary = run_contextual_topk2_support_routing_return_report(
                promotion_report_path=paths["promotion"],
                post_promotion_report_path=paths["post_promotion"],
                coverage_report_path=paths["coverage"],
                finite_update_report_path=paths["finite_update"],
                gate_suppression_audit_path=paths["gate_suppression"],
                post_stop_report_path=paths["post_stop"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("post_promotion_promoted_default", "source_artifact"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    promotion = root / "promotion.json"
    post_promotion = root / "post_promotion.json"
    coverage = root / "coverage.json"
    finite_update = root / "finite_update.json"
    gate_suppression = root / "gate_suppression.json"
    post_stop = root / "post_stop.json"
    _write_json(
        promotion,
        {
            "status": "pass",
            "decision": "satisfy_contextual_support_router_promotion_or_repeat_gate",
        },
    )
    _write_json(
        post_promotion,
        {
            "status": "pass",
            "decision": "confirm_post_promotion_support_wide_promoted_default",
            "promoted_support_router_default_confirmed": True,
            "evidence": {
                "alpha0_best_run_count": 3,
                "run_count": 3,
                "accepted_nonzero_hep_run_count": 0,
            },
        },
    )
    _write_json(
        coverage,
        {
            "status": "pass",
            "decision": RANK_MATCHED_TOPK1_ACTIVE_POST_STOP,
            "coverage": {
                "missing_fields_for_deconfounded_no_training_audit": [],
                "missing_controls_for_deconfounded_matrix": [],
                "topk2_ce_deficit_vs_rank_matched_topk1": 0.04,
            },
        },
    )
    _write_json(
        finite_update,
        {
            "status": "pass",
            "decision": FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
            "metrics": {
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_commutator_anchor_residual_stream_l2": 5.1,
                "topk2_mean_commutator_anchor_support_churn": 0.9,
            },
        },
    )
    _write_json(
        gate_suppression,
        {
            "status": "pass",
            "decision": "deployable_context_gate_suppression_calibration_failed",
            "evidence": {
                "metrics": {
                    "deployable_retained_gain_fraction": 0.91,
                    "deployable_offcontext_harm_suppression_fraction": 0.13,
                },
            },
        },
    )
    _write_json(
        post_stop,
        {
            "status": "pass",
            "decision": "select_post_stop_rank_matched_topk1_causal_bracket",
            "rank_matched_topk1_default_causal_audit_bracket": True,
            "topk2_causal_cooperation_claim_supported": False,
        },
    )
    return {
        "promotion": promotion,
        "post_promotion": post_promotion,
        "coverage": coverage,
        "finite_update": finite_update,
        "gate_suppression": gate_suppression,
        "post_stop": post_stop,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
