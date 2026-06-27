from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.router_value_disentanglement_audit import (
    AUDIT_RECORDED_NO_PROMOTION,
    INSUFFICIENT_EVIDENCE,
    run_router_value_disentanglement_audit,
)


class RouterValueDisentanglementAuditTest(unittest.TestCase):
    def test_records_no_training_disentanglement_without_promotion(self) -> None:
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

            summary = run_router_value_disentanglement_audit(
                out_dir=root / "audit",
                design_path=paths["design"],
                distillation_interventions_path=paths["interventions"],
                router_policy_rows_path=paths["router"],
                update_decomposition_rows_path=paths["decomposition"],
                value_mitigation_rows_path=paths["value"],
                commutator_value_penalty_rows_path=paths["penalty"],
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], AUDIT_RECORDED_NO_PROMOTION)
            self.assertEqual(
                summary["selected_next_action"],
                "local_same_student_retention_functional_churn_gate",
            )
            self.assertEqual(
                summary["claim_statuses"]["hub_pair_mitigation"],
                "rejected_diffuse_localization",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertAlmostEqual(summary["evidence"]["teacher_all_token_delta"], 0.1)
            self.assertAlmostEqual(summary["evidence"]["oracle_all_token_delta"], -0.02)
            self.assertAlmostEqual(summary["evidence"]["value_only_fraction_of_full"], 1.25)
            self.assertAlmostEqual(summary["evidence"]["router_only_fraction_of_full"], 0.23)
            self.assertEqual(len(summary["factor_rows"]), 14)
            self.assertTrue((root / "audit" / "summary.json").is_file())
            self.assertTrue((root / "audit" / "source_rows.csv").is_file())
            self.assertTrue((root / "audit" / "factor_rows.csv").is_file())
            self.assertTrue((root / "audit" / "notes.md").is_file())

    def test_fails_closed_when_design_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["design"].unlink()

            summary = run_router_value_disentanglement_audit(
                out_dir=root / "audit",
                design_path=paths["design"],
                distillation_interventions_path=paths["interventions"],
                router_policy_rows_path=paths["router"],
                update_decomposition_rows_path=paths["decomposition"],
                value_mitigation_rows_path=paths["value"],
                commutator_value_penalty_rows_path=paths["penalty"],
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("design", "source_artifact"), fields)
            self.assertIn(("design", "design_decision"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "design": root / "design.json",
        "interventions": root / "intervention_metrics.csv",
        "router": root / "router_policy_rows.csv",
        "decomposition": root / "decomposition_rows.csv",
        "value": root / "value_mitigation_rows.csv",
        "penalty": root / "commutator_value_penalty_rows.csv",
    }
    _write_json(
        paths["design"],
        {
            "status": "pass",
            "decision": "router_value_disentanglement_audit_design_recorded",
            "selected_next_action": "implement_no_training_router_value_disentanglement_audit",
        },
    )
    paths["interventions"].write_text(
        "\n".join(
            [
                "intervention,token_subset,fold,delta_vs_student_router_support,loss",
                "student_router_support,all_tokens,0,0.0,2.0",
                "teacher_support_forced_into_student,all_tokens,0,0.1,2.1",
                "oracle_best_support_for_student,all_tokens,0,-0.02,1.98",
                "linear_support_forced_into_student,all_tokens,0,1.2,3.2",
                "uniform_random_support,all_tokens,0,1.3,3.3",
                "teacher_support_forced_into_student,teacher_student_disagreement_tokens,0,0.4,2.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["router"].write_text(
        "\n".join(
            [
                "variant,commutator_anchor_logit_mse_reduction_fraction,anchor_ce_delta_vs_dynamic,passes_router_policy_gate",
                "dynamic_contextual_topk2,0.0,0.0,False",
                "pinned_forward_final_support_topk2,0.18,0.0,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["decomposition"].write_text(
        "\n".join(
            [
                "variant,transfer_update_group,commutator_anchor_fraction_of_full,transfer_retention_fraction",
                "router_only_transfer_topk2,router_only,0.23,0.1",
                "value_only_transfer_topk2,value_only,1.25,1.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["value"].write_text(
        "\n".join(
            [
                "variant,commutator_anchor_logit_mse_reduction_fraction,transfer_retention_fraction",
                "value_gradient_clipped_contextual_topk2,0.13,1.07",
                "value_update_scaled_contextual_topk2,-0.28,1.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["penalty"].write_text(
        "\n".join(
            [
                "variant,commutator_anchor_logit_mse_reduction_fraction,transfer_retention_fraction",
                "commutator_value_penalty_w010_contextual_topk2,0.23,1.09",
                "commutator_value_penalty_w100_contextual_topk2,-0.05,1.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
