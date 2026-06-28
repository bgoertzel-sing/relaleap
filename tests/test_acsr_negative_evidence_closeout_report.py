from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_negative_evidence_closeout_report import (
    DEMOTE_ACSR_ACTION,
    PREFIX_NORM_RESCUE_ACTION,
    REQUIRED_ARTIFACTS,
    run_acsr_negative_evidence_closeout_report,
)


class ACSRNegativeEvidenceCloseoutReportTest(unittest.TestCase):
    def test_demotes_acsr_when_mechanism_evidence_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            acsr = root / "acsr.json"
            dense = root / "dense.json"
            dense_rank_norm = root / "dense_rank_norm.json"
            mlp_churn = root / "mlp_churn.json"
            norm_budgeted = root / "norm_budgeted.json"
            commutator = root / "commutator.json"
            mechanism = root / "mechanism.json"
            review = root / "latest-review.md"
            _write_json(
                acsr,
                {
                    "status": "fail",
                    "decision": "acsr_broader_mechanism_gate_failed_closed",
                    "claim_status": "acsr_anticipation_specific_claim_blocked_no_default_change",
                    "gates": {
                        "acsr_beats_nulls_on_available_packets": True,
                        "acsr_beats_parameter_matched_causal_control": False,
                        "acsr_no_worse_retention_churn_than_contextual": False,
                        "acsr_no_worse_intervention_residual_l2_than_parameter_matched": True,
                    },
                },
            )
            _write_json(
                dense,
                {
                    "status": "fail",
                    "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                    "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
                    "dense_teacher_ce_loss": 0.25,
                    "variant_rows": [{"variant": "acsr_predicted_future_support", "ce_loss": 2.8}],
                    "gate_status": {
                        "criteria": [
                            {
                                "criterion": "acsr_ce_not_worse_than_teacher_by_large_margin",
                                "passed": False,
                            }
                        ]
                    },
                },
            )
            _write_json(
                dense_rank_norm,
                {
                    "status": "fail",
                    "decision": "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls",
                    "claim_status": "dense_rank16_24_controls_explain_ce_gain_threshold",
                    "comparison_metrics": [
                        {"metric": "minimal_dense_rank_beating_sparse", "value": 16},
                        {"metric": "rank24_delta_minus_sparse", "value": -0.05},
                    ],
                },
            )
            _write_json(
                mlp_churn,
                {
                    "status": "pass",
                    "decision": "mlp_churn_decision_recorded",
                    "claim_status": "mlp_or_sparse_advantage_not_decisive_after_ce_l2_churn_matching",
                    "candidate_actions": [
                        {
                            "candidate_action": "extract_sparse_acsr_per_token_churn_fingerprints",
                            "disposition": "selected",
                        }
                    ],
                },
            )
            _write_json(
                norm_budgeted,
                {
                    "status": "pass",
                    "decision": "post_norm_budgeted_branch_selected",
                    "claim_status": "sparse_branches_locally_blocked_dense_control_pivot_selected",
                    "selected_next_action": "pivot_to_dense_teacher_control_mechanism_assay",
                },
            )
            _write_json(
                commutator,
                {
                    "status": "fail",
                    "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
                    "claim_status": "finite_update_commutator_too_small_for_sparse_mechanism_claim",
                    "metrics": {"sparse_mean_logit_mse": 0.0006, "dense_mean_logit_mse": 0.008},
                },
            )
            _write_json(
                mechanism,
                {
                    "status": "pass",
                    "decision": "mechanism_factorized_cl_second_seed_repeat_recorded",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                    "topk2_tradeoff_repeat_status": "not_replicated",
                    "primary_result": {"topk2_tradeoff_supporting_seed_count": 1},
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run the local ACSR broader mechanism gate",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_negative_evidence_closeout_report(
                acsr_gate_path=acsr,
                dense_teacher_path=dense,
                dense_rank_norm_path=dense_rank_norm,
                mlp_churn_path=mlp_churn,
                norm_budgeted_path=norm_budgeted,
                commutator_path=commutator,
                mechanism_cl_path=mechanism,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], DEMOTE_ACSR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(
                summary["claim_status"],
                "acsr_promotion_path_demoted_to_diagnostic_no_default_change",
            )
            self.assertTrue(summary["evidence"]["dense24_or_rank_norm_blocks_sparse"])
            self.assertEqual(
                summary["evidence"]["mlp_selected_next_action"],
                "extract_sparse_acsr_per_token_churn_fingerprints",
            )
            self.assertEqual(
                summary["evidence"]["norm_budgeted_selected_next_action"],
                "pivot_to_dense_teacher_control_mechanism_assay",
            )
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], DEMOTE_ACSR_ACTION)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_selects_rescue_only_when_acsr_and_one_downstream_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            acsr = root / "acsr.json"
            dense = root / "dense.json"
            dense_rank_norm = root / "dense_rank_norm.json"
            mlp_churn = root / "mlp_churn.json"
            norm_budgeted = root / "norm_budgeted.json"
            commutator = root / "commutator.json"
            mechanism = root / "mechanism.json"
            _write_json(
                acsr,
                {
                    "status": "pass",
                    "decision": "acsr_broader_mechanism_gate_passed",
                    "claim_status": "acsr_broader_local_mechanism_gate_supported",
                    "gates": {
                        "acsr_beats_nulls_on_available_packets": True,
                        "acsr_beats_parameter_matched_causal_control": True,
                        "acsr_no_worse_retention_churn_than_contextual": True,
                        "acsr_no_worse_intervention_residual_l2_than_parameter_matched": True,
                    },
                },
            )
            _write_json(
                dense,
                {
                    "status": "pass",
                    "decision": "dense_teacher_residual_distillation_supported",
                    "claim_status": "dense_teacher_distillation_supported",
                    "variant_rows": [{"variant": "acsr_predicted_future_support", "ce_loss": 0.4}],
                },
            )
            _write_json(
                dense_rank_norm,
                {
                    "status": "pass",
                    "decision": "dense_rank_norm_does_not_block_sparse",
                    "claim_status": "dense_rank_norm_controls_do_not_explain_sparse_gain",
                },
            )
            _write_json(
                mlp_churn,
                {
                    "status": "pass",
                    "decision": "mlp_churn_decision_recorded",
                    "claim_status": "mlp_control_not_promoted",
                },
            )
            _write_json(
                norm_budgeted,
                {
                    "status": "pass",
                    "decision": "norm_budgeted_not_blocking",
                    "claim_status": "norm_budgeted_branch_not_blocking",
                },
            )
            _write_json(commutator, {"status": "fail", "decision": "tiny", "claim_status": "negative"})
            _write_json(
                mechanism,
                {
                    "status": "pass",
                    "decision": "repeat",
                    "claim_status": "mechanism_factorized_sparse_retention_not_established",
                    "topk2_tradeoff_repeat_status": "not_replicated",
                },
            )

            summary = run_acsr_negative_evidence_closeout_report(
                acsr_gate_path=acsr,
                dense_teacher_path=dense,
                dense_rank_norm_path=dense_rank_norm,
                mlp_churn_path=mlp_churn,
                norm_budgeted_path=norm_budgeted,
                commutator_path=commutator,
                mechanism_cl_path=mechanism,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], PREFIX_NORM_RESCUE_ACTION)

    def test_missing_required_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense = root / "dense.json"
            dense_rank_norm = root / "dense_rank_norm.json"
            mlp_churn = root / "mlp_churn.json"
            norm_budgeted = root / "norm_budgeted.json"
            commutator = root / "commutator.json"
            mechanism = root / "mechanism.json"
            _write_json(dense, {"status": "fail"})
            _write_json(dense_rank_norm, {"status": "fail"})
            _write_json(mlp_churn, {"status": "pass"})
            _write_json(norm_budgeted, {"status": "pass"})
            _write_json(commutator, {"status": "fail"})
            _write_json(mechanism, {"status": "pass"})

            summary = run_acsr_negative_evidence_closeout_report(
                acsr_gate_path=root / "missing-acsr.json",
                dense_teacher_path=dense,
                dense_rank_norm_path=dense_rank_norm,
                mlp_churn_path=mlp_churn,
                norm_budgeted_path=norm_budgeted,
                commutator_path=commutator,
                mechanism_cl_path=mechanism,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], "repair_missing_acsr_closeout_sources")
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
