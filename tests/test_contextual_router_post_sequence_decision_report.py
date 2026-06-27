from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_post_sequence_decision_report import (
    INSUFFICIENT_EVIDENCE,
    POST_SEQUENCE_DECISION_RECORDED,
    run_contextual_router_post_sequence_decision_report,
)


class ContextualRouterPostSequenceDecisionReportTest(unittest.TestCase):
    def test_records_blocked_post_sequence_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence = root / "sequence.json"
            local_support = root / "local_support.json"
            runpod_support = root / "runpod_support.json"
            deconfounded = root / "deconfounded.json"
            coverage = root / "coverage.json"
            review = root / "latest-review.md"
            _write_sequence(sequence)
            _write_support(local_support, ce_delta=-0.57, regret=0.08, linear_regret=0.02)
            _write_support(runpod_support, ce_delta=-0.59, regret=0.08, linear_regret=0.03)
            _write_deconfounded(deconfounded)
            _write_coverage(coverage)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run sequence and non-CE controls.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_contextual_router_post_sequence_decision_report(
                sequence_closeout_path=sequence,
                local_support_audit_path=local_support,
                runpod_support_audit_path=runpod_support,
                deconfounded_intervention_audit_path=deconfounded,
                causal_coverage_report_path=coverage,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POST_SEQUENCE_DECISION_RECORDED)
            self.assertEqual(
                summary["claim_status"],
                "causal_feature_safe_router_not_promoted_support_quality_blocked",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertIn("oracle-regret", summary["selected_next_step"])
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(
                dispositions["inspect_causal_router_oracle_regret_and_churn_failure"],
                "selected",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_support_audit_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence = root / "sequence.json"
            runpod_support = root / "runpod_support.json"
            deconfounded = root / "deconfounded.json"
            coverage = root / "coverage.json"
            _write_sequence(sequence)
            _write_support(runpod_support, ce_delta=-0.59, regret=0.08, linear_regret=0.03)
            _write_deconfounded(deconfounded)
            _write_coverage(coverage)

            summary = run_contextual_router_post_sequence_decision_report(
                sequence_closeout_path=sequence,
                local_support_audit_path=root / "missing.json",
                runpod_support_audit_path=runpod_support,
                deconfounded_intervention_audit_path=deconfounded,
                causal_coverage_report_path=coverage,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                ("local_causal_support_audit", "source_artifact"),
                {
                    (failure.get("source"), failure.get("field"))
                    for failure in summary["failures"]
                },
            )


def _write_sequence(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "sequence_kfold_backend_validated",
            "claim_status": "causal_feature_safe_router_sequence_holdout_backend_validated",
            "evidence": {
                "causal_contextual_beats_linear_both_backends": True,
                "full_context_beats_causal_contextual_both_backends": True,
            },
        },
    )


def _write_support(path: Path, *, ce_delta: float, regret: float, linear_regret: float) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "causal_contextual_router_support_audit_blocks_promotion",
            "claim_status": "causal_contextual_router_ce_supported_support_quality_not_established",
            "audit": {
                "aggregate_metrics": {
                    "causal_contextual_topk2": {
                        "mean_router_loss_delta_vs_linear": ce_delta,
                        "mean_oracle_support_regret": regret,
                        "mean_functional_churn_logit_l1": 0.36,
                    },
                    "linear_topk2": {
                        "mean_oracle_support_regret": linear_regret,
                        "mean_functional_churn_logit_l1": 0.25,
                    },
                },
                "failures": [
                    {"criterion": "causal_reduces_oracle_support_regret_vs_linear"},
                    {"criterion": "causal_no_functional_churn_increase_vs_linear"},
                ],
            },
        },
    )


def _write_deconfounded(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "topk2_comparative_causal_cooperation_not_supported",
            "evidence": {
                "deconfounded_topk2_pair_synergy_positive_strata_fraction": 0.91,
                "topk2_incremental_pair_gain_positive_strata_fraction": 0.65,
                "topk2_fixed_support_cleaner_strata_fraction": 0.65,
                "topk2_functional_churn_cleaner_strata_fraction": 0.61,
            },
        },
    )


def _write_coverage(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "rank_matched_topk1_active_post_stop_bracket",
            "coverage": {
                "post_stop_rank_matched_topk1_active": True,
                "post_stop_topk2_claim_supported": False,
            },
        },
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
