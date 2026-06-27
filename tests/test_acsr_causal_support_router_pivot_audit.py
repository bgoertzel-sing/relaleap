from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_causal_support_router_pivot_audit import (
    REQUIRED_ARTIFACTS,
    run_acsr_causal_support_router_pivot_audit,
)


class ACSRCausalSupportRouterPivotAuditTest(unittest.TestCase):
    def test_fails_closed_when_functional_token_position_null_blocks_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            acsr = root / "acsr.json"
            stratified = root / "stratified.json"
            same_student = root / "same_student.json"
            conditional = root / "conditional.json"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_json(
                acsr,
                {
                    "status": "fail",
                    "decision": "acsr_capacity_matched_causal_router_audit_failed_closed",
                    "claim_status": "acsr_as_anticipation_blocked_by_capacity_matched_causal_router",
                    "aggregate_metrics": {
                        "mean_acsr_minus_parameter_matched_ce_loss": 0.01,
                        "mean_high_support_match_acsr_minus_parameter_matched_ce_loss": 0.02,
                        "support_agreement_available": True,
                    },
                },
            )
            _write_json(
                stratified,
                {
                    "status": "pass",
                    "decision": "prior_distillation_mechanism_claim_superseded_by_stratified_null",
                    "claim_status": "distilled_causal_router_functional_mechanism_not_established_under_token_position_null",
                },
            )
            _write_json(
                same_student,
                {
                    "status": "pass",
                    "decision": "same_student_token_position_null_discriminator_blocks_claim",
                    "claim_status": "distilled_causal_router_functional_mechanism_not_established_under_same_student_token_position_null",
                    "key_metrics": {
                        "teacher_minus_token_position_null_gain_all_tokens": -0.01,
                    },
                },
            )
            _write_json(
                conditional,
                {
                    "status": "pass",
                    "decision": "conditional_permutation_assignment_signal_survives_functional_gate_blocks",
                    "claim_status": "teacher_support_assignment_exceeds_conditional_null_but_functional_claim_blocked",
                    "key_metrics": {
                        "student_exact_agreement_effect_vs_null_mean": 0.7,
                    },
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: pivot locally to a capacity-matched causal support-router mechanism audit",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_causal_support_router_pivot_audit(
                acsr_capacity_audit=acsr,
                stratified_null_report=stratified,
                same_student_report=same_student,
                conditional_resample_report=conditional,
                strategy_review=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["claim_status"],
                "direct_causal_support_router_mechanism_not_established",
            )
            self.assertIn("Ben should be notified: true", summary["direction_shift"])
            self.assertTrue(
                any(
                    failure["gate"] == "functional_token_position_null_support"
                    for failure in summary["failures"]
                )
            )
            self.assertTrue(
                any(
                    "dual-student value/support deconfounding" in boundary
                    for boundary in summary["claim_boundaries"]["not_supported"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)
            notes = (out_dir / "notes.md").read_text(encoding="utf-8")
            self.assertIn("teacher-minus-null gain=-0.01", notes)

    def test_missing_review_fails_closed_at_review_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_causal_support_router_pivot_audit(
                acsr_capacity_audit=root / "missing-acsr.json",
                stratified_null_report=root / "missing-stratified.json",
                same_student_report=root / "missing-same-student.json",
                conditional_resample_report=root / "missing-conditional.json",
                strategy_review=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(failure["gate"] == "strategy_review_consumed" for failure in summary["failures"])
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
