from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_router_policy_mitigation_probe import (
    INSUFFICIENT_EVIDENCE,
    VALUE_COMPOSITION_PRIORITIZED,
    run_promoted_topk2_router_policy_mitigation_probe,
)


class PromotedTopk2RouterPolicyMitigationProbeTest(unittest.TestCase):
    def test_prioritizes_value_composition_when_pinned_policy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run the planned router-policy mitigation probe",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_promoted_topk2_router_policy_mitigation_probe(
                config_path=root / "config.yaml",
                out_dir=root / "probe",
                order_averaging_probe_path=paths["order"],
                retention_mitigation_probe_path=paths["retention"],
                update_decomposition_audit_path=paths["decomposition"],
                value_mitigation_gate_path=paths["value"],
                commutator_value_penalty_probe_path=paths["commutator"],
                finite_update_report_path=paths["finite"],
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], VALUE_COMPOSITION_PRIORITIZED)
            self.assertEqual(
                summary["selected_next_action"],
                "post_value_router_mitigation_closeout_report",
            )
            self.assertIsNone(summary["next_command"])
            self.assertFalse(summary["router_policy_rows"][0]["passes_router_policy_gate"])
            self.assertEqual(
                summary["claim_statuses"]["router_policy_mitigation"],
                "not_established",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "source_rows.csv").is_file())
            self.assertTrue((root / "probe" / "router_policy_rows.csv").is_file())
            self.assertTrue((root / "probe" / "interpretation_rows.csv").is_file())
            self.assertTrue((root / "probe" / "notes.md").is_file())

    def test_fails_closed_when_required_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["finite"].unlink()

            summary = run_promoted_topk2_router_policy_mitigation_probe(
                config_path=root / "config.yaml",
                out_dir=root / "probe",
                order_averaging_probe_path=paths["order"],
                retention_mitigation_probe_path=paths["retention"],
                update_decomposition_audit_path=paths["decomposition"],
                value_mitigation_gate_path=paths["value"],
                commutator_value_penalty_probe_path=paths["commutator"],
                finite_update_report_path=paths["finite"],
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("finite_update_order_control", "source_artifact"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "order": root / "order.json",
        "retention": root / "retention.json",
        "decomposition": root / "decomposition.json",
        "value": root / "value.json",
        "commutator": root / "commutator.json",
        "finite": root / "finite.json",
    }
    _write_json(
        paths["order"],
        {
            "status": "pass",
            "decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
        },
    )
    _write_json(
        paths["retention"],
        {
            "status": "pass",
            "decision": "retention_mitigation_not_established",
            "mitigation_rows": [
                {
                    "variant": "router_frozen_transfer_topk2",
                    "commutator_anchor_logit_mse": 0.195,
                    "baseline_commutator_anchor_logit_mse": 0.193,
                    "commutator_anchor_logit_mse_reduction_fraction": -0.014,
                    "transfer_retention_fraction": 1.06,
                    "support_usage_retention_fraction": 1.15,
                    "anchor_support_churn_after_transfer": 0.0,
                    "commutator_anchor_support_churn": 0.9,
                }
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
            "metrics": {
                "best_value_mitigation_reduction_fraction": 0.13,
            },
        },
    )
    _write_json(
        paths["commutator"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "metrics": {
                "best_penalty_reduction_fraction": 0.23,
            },
        },
    )
    _write_json(
        paths["finite"],
        {
            "status": "pass",
            "decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
            "metrics": {
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_commutator_anchor_residual_stream_l2": 5.1,
                "dense_mean_commutator_anchor_residual_stream_l2": 3.27,
                "topk1_mean_commutator_anchor_residual_stream_l2": 1.3,
                "random_fixed_topk2_mean_commutator_anchor_residual_stream_l2": 6.7,
                "topk2_mean_commutator_anchor_support_churn": 0.92,
                "topk2_to_dense_mean_commutator_anchor_logit_mse_ratio": 3.4,
                "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": 25.8,
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
