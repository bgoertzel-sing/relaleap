from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_mitigation_branch_selector import (
    INSUFFICIENT_EVIDENCE,
    MITIGATION_BRANCH_SELECTED,
    ORDER_AVERAGING_ACTION,
    ROUTER_POLICY_ACTION,
    run_promoted_topk2_mitigation_branch_selector,
)


class PromotedTopk2MitigationBranchSelectorTest(unittest.TestCase):
    def test_selects_explicit_order_averaging_branch(self) -> None:
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

            summary = run_promoted_topk2_mitigation_branch_selector(
                shortcut_decision_path=paths["shortcut"],
                finite_update_report_path=paths["finite_update"],
                value_mitigation_audit_path=paths["value_mitigation"],
                low_rank_value_audit_path=paths["low_rank_value"],
                commutator_value_penalty_audit_path=paths["commutator_value_penalty"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], MITIGATION_BRANCH_SELECTED)
            self.assertEqual(summary["selected_next_action"], ORDER_AVERAGING_ACTION)
            self.assertIn("explicit_order_averaging", summary["next_command"])
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(dispositions[ORDER_AVERAGING_ACTION], "selected")
            self.assertEqual(dispositions[ROUTER_POLICY_ACTION], "deferred")
            self.assertEqual(
                summary["claim_statuses"]["topk2_causal_cooperation"],
                "not_supported_pending_commutator_cleanliness",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_commutator_penalty_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["commutator_value_penalty"].unlink()

            summary = run_promoted_topk2_mitigation_branch_selector(
                shortcut_decision_path=paths["shortcut"],
                finite_update_report_path=paths["finite_update"],
                value_mitigation_audit_path=paths["value_mitigation"],
                low_rank_value_audit_path=paths["low_rank_value"],
                commutator_value_penalty_audit_path=paths["commutator_value_penalty"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("commutator_value_penalty_probe", "source_artifact"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "shortcut": root / "shortcut.json",
        "finite_update": root / "finite_update.json",
        "value_mitigation": root / "value_mitigation.json",
        "low_rank_value": root / "low_rank_value.json",
        "commutator_value_penalty": root / "commutator_value_penalty.json",
    }
    _write_json(
        paths["shortcut"],
        {
            "status": "pass",
            "decision": "contextual_router_shortcut_decision_selected",
            "selected_next_action": "commutator_aware_value_penalty_probe",
        },
    )
    _write_json(
        paths["finite_update"],
        {
            "status": "pass",
            "decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
            "metrics": {
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_commutator_anchor_support_churn": 0.91,
                "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": 0.25,
                "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": -0.05,
                "topk2_mean_order_averaged_anchor_logit_mse_to_forward": 0.06,
            },
        },
    )
    _write_json(
        paths["value_mitigation"],
        {
            "status": "pass",
            "decision": "value_mitigation_not_established",
            "metrics": {"best_value_mitigation_reduction_fraction": 0.13},
        },
    )
    _write_json(
        paths["low_rank_value"],
        {
            "status": "pass",
            "decision": "low_rank_value_not_established",
            "metrics": {"best_low_rank_reduction_fraction": 0.09},
        },
    )
    _write_json(
        paths["commutator_value_penalty"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "metrics": {
                "best_penalty_reduction_fraction": 0.23,
                "best_penalty_transfer_retention_fraction": 1.09,
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
