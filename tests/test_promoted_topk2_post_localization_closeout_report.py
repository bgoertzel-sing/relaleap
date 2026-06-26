from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_post_localization_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    POST_LOCALIZATION_CLOSEOUT,
    SELECTED_RETENTION_CAUSAL_AUDIT,
    run_promoted_topk2_post_localization_closeout_report,
)


class PromotedTopk2PostLocalizationCloseoutReportTest(unittest.TestCase):
    def test_closes_diffuse_value_router_family_and_selects_retention_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run active-rank-matched top-k-1 selection",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_promoted_topk2_post_localization_closeout_report(
                post_value_closeout_dir=paths["post_value"],
                pairwise_localization_dir=paths["localization"],
                active_topk1_selection_dir=paths["selection"],
                retention_followup_dir=paths["retention"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POST_LOCALIZATION_CLOSEOUT)
            self.assertEqual(
                summary["selected_next_branch"], SELECTED_RETENTION_CAUSAL_AUDIT
            )
            self.assertFalse(summary["selection_gate"]["requires_gpu_now"])
            self.assertFalse(summary["selection_gate"]["adds_new_mitigation_family"])
            self.assertTrue(
                summary["selection_gate"]["retention_followup_already_completed"]
            )
            self.assertEqual(
                summary["claim_statuses"]["value_router_mitigation_family"],
                "closed_not_established",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "closure_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_localization_is_not_diffuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            _write_json(
                paths["localization"] / "summary.json",
                {
                    "status": "pass",
                    "decision": "pairwise_value_interaction_localized_hub_family",
                    "localization_status": "hub_localized",
                },
            )

            summary = run_promoted_topk2_post_localization_closeout_report(
                post_value_closeout_dir=paths["post_value"],
                pairwise_localization_dir=paths["localization"],
                active_topk1_selection_dir=paths["selection"],
                retention_followup_dir=paths["retention"],
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
                ("pairwise_value_interaction_localization", "decision"), fields
            )


def _write_sources(root: Path) -> dict[str, Path]:
    post_value = root / "post_value"
    localization = root / "localization"
    selection = root / "selection"
    retention = root / "retention"
    for path in (post_value, localization, selection, retention):
        path.mkdir(parents=True)
    _write_json(
        post_value / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_mitigation_closeout_no_promotion",
            "selected_next_action": "pairwise_value_interaction_localization_audit",
        },
    )
    _write_json(
        localization / "summary.json",
        {
            "status": "pass",
            "decision": "pairwise_value_interaction_diffuse",
            "localization_status": "diffuse",
            "metrics": {
                "top3_pair_abs_synergy_share": 0.4,
                "dominant_column_abs_synergy_share": 0.5,
                "frequency_control_primary_denominator_count": 0,
            },
        },
    )
    _write_json(
        selection / "summary.json",
        {
            "status": "pass",
            "decision": "active_topk1_next_evidence_selected",
            "selected_experiment": "retention_churn",
            "signals": {"matched_control_coverage_adequate": True},
        },
    )
    _write_json(
        retention / "summary.json",
        {
            "status": "pass",
            "decision": "retention_functional_churn_bracket_supported",
            "aggregates": {
                "ce_guardrail_all_packets": True,
                "min_support_churn_advantage_topk1_vs_topk2": 0.8,
                "min_commutator_anchor_advantage_topk1_vs_topk2": 0.17,
                "min_transfer_advantage_topk1_vs_dense": 0.39,
            },
        },
    )
    return {
        "post_value": post_value,
        "localization": localization,
        "selection": selection,
        "retention": retention,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
