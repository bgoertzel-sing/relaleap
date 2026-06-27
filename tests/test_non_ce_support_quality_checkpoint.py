from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.non_ce_support_quality_checkpoint import (
    CHECKPOINT_SELECTED,
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    run_non_ce_support_quality_checkpoint,
)


class NonCESupportQualityCheckpointTest(unittest.TestCase):
    def test_selects_router_value_disentanglement_after_completed_closeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run closeout then retention/churn gate",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_non_ce_support_quality_checkpoint(
                causal_router_closeout_path=paths["causal_router_closeout"],
                post_localization_closeout_path=paths["post_localization"],
                retention_synthesis_path=paths["retention_synthesis"],
                shortcut_decision_path=paths["shortcut_decision"],
                commutator_value_penalty_path=paths["commutator_penalty"],
                support_frequency_blocker_path=paths["support_frequency"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], CHECKPOINT_SELECTED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertEqual(
                summary["claim_statuses"]["hub_pair_mitigation"],
                "deferred_rejected_diffuse_localization",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(dispositions[SELECTED_NEXT_ACTION], "selected")
            self.assertEqual(
                dispositions["hub_pair_or_order_averaging_mitigation"],
                "deferred_rejected",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_required_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["support_frequency"].unlink()

            summary = run_non_ce_support_quality_checkpoint(
                causal_router_closeout_path=paths["causal_router_closeout"],
                post_localization_closeout_path=paths["post_localization"],
                retention_synthesis_path=paths["retention_synthesis"],
                shortcut_decision_path=paths["shortcut_decision"],
                commutator_value_penalty_path=paths["commutator_penalty"],
                support_frequency_blocker_path=paths["support_frequency"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("support_frequency_blocker_diagnostic", "summary_json"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "causal_router_closeout": root / "causal_router_closeout.json",
        "post_localization": root / "post_localization.json",
        "retention_synthesis": root / "retention_synthesis.json",
        "shortcut_decision": root / "shortcut_decision.json",
        "commutator_penalty": root / "commutator_penalty.json",
        "support_frequency": root / "support_frequency.json",
    }
    _write_json(
        paths["causal_router_closeout"],
        {
            "status": "pass",
            "decision": "causal_contextual_router_distillation_branch_closed_no_promotion",
            "claim_status": "causal_router_ce_baseline_only_support_mechanism_not_established",
        },
    )
    _write_json(
        paths["post_localization"],
        {
            "status": "pass",
            "decision": "promoted_topk2_value_router_family_closed",
            "metrics": {
                "pairwise_localization_decision": "pairwise_value_interaction_diffuse"
            },
        },
    )
    _write_json(
        paths["retention_synthesis"],
        {
            "status": "pass",
            "decision": "causal_retention_claim_blocked_by_deployable_gate",
            "signals": {"causal_retention_claim_supported": False},
        },
    )
    _write_json(
        paths["shortcut_decision"],
        {
            "status": "pass",
            "decision": "contextual_router_shortcut_decision_selected",
            "selected_next_action": "commutator_aware_value_penalty_probe",
            "claim_statuses": {
                "topk2_causal_cooperation": "not_supported_pending_commutator_cleanliness"
            },
        },
    )
    _write_json(
        paths["commutator_penalty"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "metrics": {"best_penalty_reduction_fraction": 0.23},
        },
    )
    _write_json(
        paths["support_frequency"],
        {
            "status": "pass",
            "decision": "support_frequency_percentile_claim_remains_blocked_by_support_count_caliper",
            "evidence": {"claim_bearing": False},
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
