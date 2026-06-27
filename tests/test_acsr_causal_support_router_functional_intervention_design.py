from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_causal_support_router_functional_intervention_design import (
    REQUIRED_ARTIFACTS,
    run_acsr_causal_support_router_functional_intervention_design,
)


class ACSRCausalSupportRouterFunctionalInterventionDesignTest(unittest.TestCase):
    def test_records_dual_student_design_from_fail_closed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pivot = root / "pivot.json"
            capacity = root / "capacity.json"
            same_student = root / "same_student.json"
            matrix = root / "same_student_matrix.csv"
            router_value = root / "router_value.json"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_json(
                pivot,
                {
                    "status": "fail",
                    "decision": "causal_support_router_pivot_audit_failed_closed",
                    "claim_status": "direct_causal_support_router_mechanism_not_established",
                },
            )
            _write_json(
                capacity,
                {
                    "status": "fail",
                    "claim_status": "acsr_as_anticipation_blocked_by_capacity_matched_causal_router",
                    "aggregate_metrics": {
                        "mean_acsr_minus_parameter_matched_ce_loss": 0.001,
                    },
                },
            )
            _write_json(
                same_student,
                {
                    "status": "pass",
                    "claim_status": (
                        "distilled_causal_router_functional_mechanism_not_established_"
                        "under_same_student_token_position_null"
                    ),
                    "key_metrics": {
                        "teacher_minus_token_position_null_gain_all_tokens": 0.0009,
                        "teacher_forced_gain_all_tokens": -0.02,
                    },
                },
            )
            matrix.write_text(
                "\n".join(
                    [
                        "token_subset,intervention,gain_vs_student_router",
                        "all_tokens,oracle_best_support_for_student,0.012",
                        "teacher_student_disagreement_tokens,oracle_best_support_for_student,0.048",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                router_value,
                {
                    "status": "pass",
                    "claim_statuses": {
                        "router_value_disentanglement": (
                            "recorded_value_path_and_support_selection_entangled"
                        ),
                    },
                    "evidence": {
                        "value_only_fraction_of_full": 1.25,
                        "router_only_fraction_of_full": 0.23,
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

            summary = run_acsr_causal_support_router_functional_intervention_design(
                pivot_audit=pivot,
                capacity_audit=capacity,
                same_student_report=same_student,
                same_student_matrix=matrix,
                router_value_audit=router_value,
                strategy_review=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], "design_only_mechanism_not_established")
            self.assertIn("Ben should be notified: true", summary["direction_shift"])
            self.assertTrue(
                any(
                    row["intervention"] == "dual_student_cross_forcing"
                    and row["available_now"] is False
                    for row in summary["intervention_design"]
                )
            )
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_causal_support_router_functional_intervention_design(
                pivot_audit=root / "missing-pivot.json",
                capacity_audit=root / "missing-capacity.json",
                same_student_report=root / "missing-same-student.json",
                same_student_matrix=root / "missing-matrix.csv",
                router_value_audit=root / "missing-router-value.json",
                strategy_review=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(
                    row["criterion"] == "major_strategy_review_consumed"
                    for row in summary["failures"]
                )
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
