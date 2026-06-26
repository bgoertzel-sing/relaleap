from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_shortcut_decision_report import (
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    SHORTCUT_DECISION_SELECTED,
    run_contextual_router_shortcut_decision_report,
)


class ContextualRouterShortcutDecisionReportTest(unittest.TestCase):
    def test_selects_commutator_aware_value_penalty_probe(self) -> None:
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

            summary = run_contextual_router_shortcut_decision_report(
                shortcut_audit_path=paths["shortcut"],
                support_selection_report_path=paths["support_selection"],
                functional_churn_report_path=paths["functional_churn"],
                finite_update_report_path=paths["finite_update"],
                value_mitigation_audit_path=paths["value_mitigation"],
                low_rank_value_audit_path=paths["low_rank_value"],
                topk1_gate_audit_path=paths["topk1_gate"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SHORTCUT_DECISION_SELECTED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertIn("commutator_value_penalty_probe", summary["next_command"])
            self.assertEqual(
                summary["claim_statuses"]["contextual_shortcut_risk"],
                "bounded_in_fixed_batch_but_not_a_generalization_claim",
            )
            self.assertEqual(
                summary["claim_statuses"]["topk2_causal_cooperation"],
                "not_supported_pending_commutator_cleanliness",
            )
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(dispositions[SELECTED_NEXT_ACTION], "selected")
            self.assertEqual(
                dispositions["simple_value_scaling_or_low_rank_repeat"],
                "disqualified",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_shortcut_audit_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["shortcut"].unlink()

            summary = run_contextual_router_shortcut_decision_report(
                shortcut_audit_path=paths["shortcut"],
                support_selection_report_path=paths["support_selection"],
                functional_churn_report_path=paths["functional_churn"],
                finite_update_report_path=paths["finite_update"],
                value_mitigation_audit_path=paths["value_mitigation"],
                low_rank_value_audit_path=paths["low_rank_value"],
                topk1_gate_audit_path=paths["topk1_gate"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("contextual_router_shortcut_ablation", "source_artifact"),
                fields,
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "shortcut": root / "shortcut.json",
        "support_selection": root / "support_selection.json",
        "functional_churn": root / "functional_churn.json",
        "finite_update": root / "finite_update.json",
        "value_mitigation": root / "value_mitigation.json",
        "low_rank_value": root / "low_rank_value.json",
        "topk1_gate": root / "topk1_gate.json",
    }
    _write_json(
        paths["shortcut"],
        {
            "status": "ok",
            "decision": "contextual_router_shortcut_ablation_completed",
            "ablation": {
                "selected_variant": "full_context",
                "shortcut_interpretation": "full_context_features_best_supported",
                "router_oracle_gap": 0.0025,
                "variants": {
                    "full_context": {
                        "holdout": {
                            "intervention_oracle_gap_recovery_fraction": 1.0,
                        }
                    },
                    "position_only": {
                        "holdout": {
                            "intervention_oracle_gap_recovery_fraction": -449.0,
                        }
                    },
                    "context_only": {
                        "holdout": {
                            "intervention_oracle_gap_recovery_fraction": -2.2,
                        }
                    },
                },
            },
        },
    )
    _write_json(
        paths["support_selection"],
        {
            "status": "pass",
            "decision": "promoted_topk2_support_selection_quality_established",
        },
    )
    _write_json(
        paths["functional_churn"],
        {
            "status": "pass",
            "decision": "support_identity_churn_functional_impact_bounded_with_commutator_risk",
        },
    )
    _write_json(
        paths["finite_update"],
        {
            "status": "pass",
            "decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
            "metrics": {
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_commutator_anchor_residual_stream_l2": 5.1,
            },
        },
    )
    _write_json(
        paths["value_mitigation"],
        {"status": "pass", "decision": "value_mitigation_not_established"},
    )
    _write_json(
        paths["low_rank_value"],
        {"status": "pass", "decision": "low_rank_value_not_established"},
    )
    _write_json(
        paths["topk1_gate"],
        {
            "status": "pass",
            "decision": "deployable_context_gate_suppression_calibration_failed",
            "evidence": {
                "metrics": {
                    "deployable_retained_gain_fraction": 0.91,
                    "deployable_offcontext_harm_suppression_fraction": 0.13,
                }
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
