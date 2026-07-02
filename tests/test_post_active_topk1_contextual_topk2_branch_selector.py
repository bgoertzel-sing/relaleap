from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_active_topk1_contextual_topk2_branch_selector import (
    FAILED_DECISION,
    SELECTED_ACTION,
    SELECTED_DECISION,
    run_post_active_topk1_contextual_topk2_branch_selector,
)


class PostActiveTopk1ContextualTopk2BranchSelectorTest(unittest.TestCase):
    def test_selects_support_quality_preserving_contextual_topk2_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement local scale-constrained sparse residual-compression pilot",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_active_topk1_contextual_topk2_branch_selector(
                active_topk1_synthesis_path=paths["active"],
                contextual_failure_path=paths["contextual"],
                topk2_value_closeout_path=paths["value_closeout"],
                topk2_return_path=paths["topk2_return"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SELECTED_DECISION)
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertIn("functional churn", summary["selected_next_step"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_active_topk1_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["active"].unlink()

            summary = run_post_active_topk1_contextual_topk2_branch_selector(
                active_topk1_synthesis_path=paths["active"],
                contextual_failure_path=paths["contextual"],
                topk2_value_closeout_path=paths["value_closeout"],
                topk2_return_path=paths["topk2_return"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], FAILED_DECISION)
            self.assertTrue(summary["failures"])


def _write_sources(root: Path) -> dict[str, Path]:
    active = root / "active.json"
    contextual = root / "contextual.json"
    value_closeout = root / "value_closeout.json"
    topk2_return = root / "topk2_return.json"
    _write_json(
        active,
        {
            "status": "pass",
            "decision": "causal_retention_claim_blocked_by_deployable_gate",
            "claim_status": "local_retention_bracket_with_context_gated_singleton_efficacy_only",
            "evidence": {
                "deployable_context_gate_failed": True,
                "local_retention_churn_bracket_supported": True,
            },
        },
    )
    _write_json(
        contextual,
        {
            "status": "pass",
            "decision": "contextual_router_regret_churn_failure_inspection_recorded",
            "claim_status": "causal_router_ce_win_is_not_support_quality_evidence",
            "evidence": {
                "all_folds_causal_ce_beats_linear": True,
                "all_folds_causal_regret_worse_than_linear": True,
                "all_folds_causal_churn_worse_than_linear": True,
            },
        },
    )
    _write_json(
        value_closeout,
        {
            "status": "pass",
            "decision": "promoted_topk2_value_router_family_closed",
            "claim_status": "topk2_value_router_mitigation_family_closed_no_promotion",
        },
    )
    _write_json(
        topk2_return,
        {
            "status": "pass",
            "decision": "contextual_topk2_support_routing_return_selected",
            "selected_next_action": "contextual_router_shortcut_ablation",
        },
    )
    return {
        "active": active,
        "contextual": contextual,
        "value_closeout": value_closeout,
        "topk2_return": topk2_return,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
