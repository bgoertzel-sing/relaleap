from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_explicit_order_averaging_mitigation_probe import (
    INSUFFICIENT_EVIDENCE,
    ORDER_AVERAGING_DIAGNOSTIC_CANDIDATE,
    run_promoted_topk2_explicit_order_averaging_mitigation_probe,
)


class PromotedTopk2ExplicitOrderAveragingMitigationProbeTest(unittest.TestCase):
    def test_records_diagnostic_candidate_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement the contextual top-k-2 support-routing return report",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_promoted_topk2_explicit_order_averaging_mitigation_probe(
                branch_selector_path=paths["branch_selector"],
                finite_update_report_path=paths["finite_update"],
                control_matrix_path=paths["control_matrix"],
                flat_value_report_path=paths["flat_value"],
                strategy_review_path=review,
                out_dir=root / "probe",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ORDER_AVERAGING_DIAGNOSTIC_CANDIDATE)
            self.assertEqual(
                summary["claim_statuses"]["order_averaging"],
                "diagnostic_upper_bound_not_deployable_not_promoted",
            )
            self.assertEqual(
                summary["selected_next_action"],
                "record_order_averaging_matched_control_closeout_no_gpu",
            )
            self.assertIsNone(summary["next_command"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["order_averaging_rows"][0]["passes_diagnostic_gate"])
            gates = {row["gate"]: row["passes"] for row in summary["gate_rows"]}
            self.assertTrue(gates["dense_control_present_and_matched"])
            self.assertTrue(gates["random_support_control_present"])
            self.assertFalse(gates["flat_value_order_averaging_control_present"])
            roles = {row["role"] for row in summary["control_rows"]}
            self.assertIn("dense_control", roles)
            self.assertIn("random_support_control", roles)
            self.assertIn("flat_value_control", roles)
            self.assertIn("no_update_null", roles)
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "source_rows.csv").is_file())
            self.assertTrue((root / "probe" / "order_averaging_rows.csv").is_file())
            self.assertTrue((root / "probe" / "control_rows.csv").is_file())
            self.assertTrue((root / "probe" / "gate_rows.csv").is_file())
            self.assertTrue((root / "probe" / "notes.md").is_file())

    def test_fails_closed_when_finite_update_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["finite_update"].unlink()

            summary = run_promoted_topk2_explicit_order_averaging_mitigation_probe(
                branch_selector_path=paths["branch_selector"],
                finite_update_report_path=paths["finite_update"],
                control_matrix_path=paths["control_matrix"],
                flat_value_report_path=paths["flat_value"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "probe",
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
        "branch_selector": root / "branch_selector.json",
        "finite_update": root / "finite_update.json",
        "control_matrix": root / "control_matrix.json",
        "flat_value": root / "flat_value.json",
    }
    _write_json(
        paths["branch_selector"],
        {
            "status": "pass",
            "decision": "promoted_topk2_mitigation_branch_selected",
            "selected_next_action": "explicit_order_averaging_mitigation_probe",
        },
    )
    _write_json(
        paths["finite_update"],
        {
            "status": "pass",
            "decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
            "signals": {"order_averaged_rows_available": True},
            "metrics": {
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_order_averaged_anchor_logit_mse_to_forward": 0.06,
                "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": 0.25,
                "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": -0.05,
                "topk2_mean_order_averaged_anchor_ce_delta_vs_forward": -0.06,
                "topk2_mean_same_order_ensemble_anchor_ce_delta_vs_best_endpoint": -0.13,
                "topk2_same_order_identical_anchor_logit_mse_to_commutator_ratio": 0.00005,
            },
        },
    )
    _write_json(
        paths["control_matrix"],
        {
            "status": "pass",
            "decision": "finite_update_control_matrix_ready",
            "matrix_rows": [
                {
                    "variant": "promoted_contextual_topk2",
                    "split": "all",
                    "mean_logit_mse": 0.24,
                    "mean_ce_abs_delta": 0.39,
                    "mean_residual_delta_l2": 5.1,
                    "mean_symmetric_kl": 0.25,
                    "support_churn_fraction": 0.92,
                    "row_count": 1008,
                },
                {
                    "variant": "norm_matched_dense_active_rank",
                    "split": "all",
                    "mean_logit_mse": 0.07,
                    "mean_ce_abs_delta": 0.06,
                    "mean_residual_delta_l2": 3.1,
                    "mean_symmetric_kl": 0.02,
                    "support_churn_fraction": None,
                    "row_count": 1008,
                },
                {
                    "variant": "random_fixed_topk2",
                    "split": "all",
                    "mean_logit_mse": 0.35,
                    "mean_ce_abs_delta": 0.44,
                    "mean_residual_delta_l2": 6.7,
                    "mean_symmetric_kl": 0.22,
                    "support_churn_fraction": 0.0,
                    "row_count": 1008,
                },
                {
                    "variant": "rank_matched_contextual_topk1",
                    "split": "all",
                    "mean_logit_mse": 0.009,
                    "mean_ce_abs_delta": 0.10,
                    "mean_residual_delta_l2": 1.2,
                    "mean_symmetric_kl": 0.006,
                    "support_churn_fraction": 0.0,
                    "row_count": 1008,
                },
            ],
        },
    )
    _write_json(
        paths["flat_value"],
        {
            "status": "pass",
            "decision": "same_router_flat_value_commutator_mitigation_probe_gpu_blocked",
            "claim_status": "flat_value_commutator_mitigation_not_established",
            "missing_required_variant_count": 2,
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
