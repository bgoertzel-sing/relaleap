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
                strategy_review_path=review,
                out_dir=root / "probe",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ORDER_AVERAGING_DIAGNOSTIC_CANDIDATE)
            self.assertEqual(
                summary["claim_statuses"]["order_averaging"],
                "diagnostic_only_not_promoted",
            )
            self.assertEqual(summary["selected_next_action"], "router_policy_mitigation_probe")
            self.assertIn("router_policy", summary["next_command"])
            self.assertTrue(summary["order_averaging_rows"][0]["passes_diagnostic_gate"])
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "source_rows.csv").is_file())
            self.assertTrue((root / "probe" / "order_averaging_rows.csv").is_file())
            self.assertTrue((root / "probe" / "notes.md").is_file())

    def test_fails_closed_when_finite_update_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["finite_update"].unlink()

            summary = run_promoted_topk2_explicit_order_averaging_mitigation_probe(
                branch_selector_path=paths["branch_selector"],
                finite_update_report_path=paths["finite_update"],
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
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
