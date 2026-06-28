from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.heldout_context_post_probe_decision_report import (
    DECISION_RECORDED,
    DENSE_BASELINE_ACTIVE,
    INSUFFICIENT_EVIDENCE,
    NEXT_BRANCH,
    REQUIRED_ARTIFACTS,
    run_heldout_context_post_probe_decision_report,
)


class HeldoutContextPostProbeDecisionReportTest(unittest.TestCase):
    def test_report_selects_synthetic_retention_branch_after_dense_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_heldout_context_post_probe_decision_report(
                probe_dir=paths["probe"],
                dense_teacher_dir=paths["dense_teacher"],
                commutator_dir=paths["commutator"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION_RECORDED)
            self.assertEqual(summary["claim_policy"], DENSE_BASELINE_ACTIVE)
            self.assertEqual(summary["selected_next_step"], NEXT_BRANCH)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertLess(summary["primary_metrics"]["mean_dense_minus_topk1_heldout_delta"], -0.1)
            self.assertGreater(
                summary["primary_metrics"]["mean_topk1_support_identity_advantage_vs_random"],
                0.0,
            )
            self.assertIn("task_a_retention_ce_delta_after_task_b_update", summary["next_branch_design"]["estimands"])
            self.assertTrue(
                any(
                    row["criterion"] == "dense_teacher_branch_already_failed_gate" and row["passed"]
                    for row in summary["decision_rows"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_report_fails_closed_without_probe_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["probe"] / "summary.json").unlink()

            summary = run_heldout_context_post_probe_decision_report(
                probe_dir=paths["probe"],
                dense_teacher_dir=paths["dense_teacher"],
                commutator_dir=paths["commutator"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertNotEqual(summary["selected_next_step"], NEXT_BRANCH)
            fields = {(failure["source"], failure["field"]) for failure in summary["failures"]}
            self.assertIn(("heldout_context_intervention_probe", "status_or_decision"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    probe = root / "probe"
    dense_teacher = root / "dense_teacher"
    commutator = root / "commutator"
    probe.mkdir()
    dense_teacher.mkdir()
    commutator.mkdir()
    _write_json(
        probe / "summary.json",
        {
            "status": "pass",
            "decision": "heldout_context_intervention_probe_passed",
            "claim_status": "rank_matched_topk1_remains_diagnostic_dense_baseline_active",
        },
    )
    (probe / "paired_deltas.csv").write_text(
        "\n".join(
            [
                "seed,comparison,left_arm,right_arm,left_present,right_present,dense_minus_topk1_heldout_delta,left_minus_right_heldout_delta",
                "seed1,primary_dense_minus_sparse,rank_flop_matched_causal_dense,sparse_rank_matched_topk1,True,True,-0.11,-0.11",
                "seed2,primary_dense_minus_sparse,rank_flop_matched_causal_dense,sparse_rank_matched_topk1,True,True,-0.15,-0.15",
                "seed1,topk1_minus_random_support_null,sparse_rank_matched_topk1,sparse_frequency_matched_random_topk1,True,True,,-0.27",
                "seed2,topk1_minus_random_support_null,sparse_rank_matched_topk1,sparse_frequency_matched_random_topk1,True,True,,-0.26",
                "seed1,causal_dense_minus_shuffled_context_null,rank_flop_matched_causal_dense,rank_flop_matched_shuffled_context_dense,True,True,,-0.37",
                "seed2,causal_dense_minus_ablated_context_null,rank_flop_matched_causal_dense,rank_flop_matched_ablated_context_dense,True,True,,-0.42",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (probe / "gate_criteria.csv").write_text(
        "\n".join(
            [
                "criterion,passed,severity,requirement,observed,failure_reason",
                "required_arms_and_nulls_present,True,hard,required,{},",
                "residual_norm_and_active_compute_accounting_present,True,hard,required,[],",
                "topk1_beats_causal_dense_on_heldout_ce,False,claim_blocker,required,\"[-0.11, -0.15]\",blocked",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        dense_teacher / "summary.json",
        {
            "status": "fail",
            "decision": "dense_teacher_residual_distillation_pilot_not_supported",
            "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
        },
    )
    _write_json(
        commutator / "summary.json",
        {
            "status": "fail",
            "decision": "acsr_finite_update_commutator_assay_tiny_commutator",
            "claim_status": "finite_update_commutator_too_small_for_sparse_mechanism_claim",
        },
    )
    review = root / "latest-review.md"
    review.write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: keep RunPod deferred after local null-controlled probe",
                "verdict: FIX",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "probe": probe,
        "dense_teacher": dense_teacher,
        "commutator": commutator,
        "review": review,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
