from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.router_value_disentanglement_audit_design import (
    DESIGN_RECORDED,
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    run_router_value_disentanglement_audit_design,
)


class RouterValueDisentanglementAuditDesignTest(unittest.TestCase):
    def test_records_design_from_existing_support_and_value_artifacts(self) -> None:
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

            summary = run_router_value_disentanglement_audit_design(
                out_dir=root / "report",
                checkpoint_path=paths["checkpoint"],
                router_policy_probe_path=paths["router"],
                update_decomposition_audit_path=paths["decomposition"],
                value_mitigation_gate_path=paths["value"],
                commutator_value_penalty_probe_path=paths["penalty"],
                distillation_agreement_path=paths["distillation"],
                distillation_interventions_path=paths["interventions"],
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DESIGN_RECORDED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertEqual(
                summary["claim_statuses"]["hub_pair_mitigation"],
                "deferred_rejected_diffuse_localization",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(summary["evidence"]["support_swap_intervention_count"], 4)
            self.assertAlmostEqual(summary["evidence"]["teacher_support_mean_delta"], 0.1)
            self.assertEqual(len(summary["design_rows"]), 4)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "design_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_support_swap_rows_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["interventions"].unlink()

            summary = run_router_value_disentanglement_audit_design(
                out_dir=root / "report",
                checkpoint_path=paths["checkpoint"],
                router_policy_probe_path=paths["router"],
                update_decomposition_audit_path=paths["decomposition"],
                value_mitigation_gate_path=paths["value"],
                commutator_value_penalty_probe_path=paths["penalty"],
                distillation_agreement_path=paths["distillation"],
                distillation_interventions_path=paths["interventions"],
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("distillation_interventions", "source_artifact"), fields)
            self.assertIn(
                ("distillation_interventions", "support_swap_intervention_count"),
                fields,
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "checkpoint": root / "checkpoint.json",
        "router": root / "router.json",
        "decomposition": root / "decomposition.json",
        "value": root / "value.json",
        "penalty": root / "penalty.json",
        "distillation": root / "distillation.json",
        "interventions": root / "intervention_metrics.csv",
    }
    _write_json(
        paths["checkpoint"],
        {
            "status": "pass",
            "decision": "non_ce_support_quality_checkpoint_selected",
            "selected_next_action": "router_value_disentanglement_audit_design",
        },
    )
    _write_json(
        paths["router"],
        {
            "status": "pass",
            "decision": "value_composition_prioritized_over_router_policy",
            "router_policy_rows": [
                {
                    "variant": "dynamic_contextual_topk2",
                    "commutator_anchor_logit_mse_reduction_fraction": 0.0,
                },
                {
                    "variant": "pinned_forward_final_support_topk2",
                    "commutator_anchor_logit_mse_reduction_fraction": 0.18,
                },
            ],
        },
    )
    _write_json(
        paths["decomposition"],
        {
            "status": "pass",
            "decision": "value_update_dominated_order_sensitivity",
            "metrics": {
                "value_only_fraction_of_full": 1.25,
                "router_only_fraction_of_full": 0.23,
            },
        },
    )
    _write_json(
        paths["value"],
        {
            "status": "pass",
            "decision": "value_mitigation_not_established",
            "metrics": {"best_value_mitigation_reduction_fraction": 0.13},
        },
    )
    _write_json(
        paths["penalty"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "metrics": {"best_penalty_reduction_fraction": 0.23},
        },
    )
    _write_json(
        paths["distillation"],
        {
            "status": "pass",
            "decision": "causal_contextual_router_distillation_agreement_recorded",
        },
    )
    paths["interventions"].write_text(
        "\n".join(
            [
                "intervention,delta_vs_student_router_support,fold,token_subset",
                "student_router_support,0.0,0,all_tokens",
                "teacher_support_forced_into_student,0.1,0,all_tokens",
                "oracle_best_support_for_student,-0.02,0,all_tokens",
                "linear_support_forced_into_student,1.2,0,all_tokens",
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
