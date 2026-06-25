from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_functional_churn_control_audit import (
    FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
    INSUFFICIENT_EVIDENCE,
    run_promoted_topk2_functional_churn_control_audit,
)


class PromotedTopk2FunctionalChurnControlAuditTest(unittest.TestCase):
    def test_bounds_support_identity_churn_but_keeps_commutator_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector = root / "selector"
            retention = root / "retention"
            fingerprint = root / "fingerprint"
            _write_selector(selector)
            _write_retention(retention)
            _write_fingerprint(fingerprint)

            summary = run_promoted_topk2_functional_churn_control_audit(
                selector_dir=selector,
                retention_reference_dir=retention,
                fingerprint_dir=fingerprint,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
            )
            self.assertTrue(summary["signals"]["support_churn_gap_high"])
            self.assertTrue(summary["signals"]["logit_churn_gap_low"])
            self.assertTrue(summary["signals"]["residual_drift_ratio_low"])
            self.assertTrue(summary["signals"]["finite_update_commutator_risk_high"])
            self.assertGreater(summary["metrics"]["mean_support_churn_gap"], 0.8)
            self.assertLess(summary["metrics"]["mean_logit_churn_gap"], 0.05)
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "packet_metrics.csv").is_file())
            self.assertTrue(
                (root / "out" / "fingerprint_functional_churn.csv").is_file()
            )
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_fails_closed_when_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_promoted_topk2_functional_churn_control_audit(
                selector_dir=root / "missing_selector",
                retention_reference_dir=root / "missing_retention",
                fingerprint_dir=root / "missing_fingerprint",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("summary_json", fields)
            self.assertIn("packet_rows", fields)
            self.assertIn("fingerprint_functional_churn_rows", fields)


def _write_selector(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "select_functional_churn_controls",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_retention(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for index, churn in enumerate((0.9, 0.82), 1):
        rows.append(
            {
                "packet": f"seed{index}",
                "config_path": f"configs/seed{index}.yaml",
                "probe_dir": "",
                "topk2_anchor_support_churn_after_transfer": churn,
                "topk1_anchor_support_churn_after_transfer": 0.01,
                "topk2_anchor_logit_mse_drift": 0.16,
                "topk1_anchor_logit_mse_drift": 0.14,
                "topk2_anchor_residual_stream_l2_drift": 4.6,
                "topk1_anchor_residual_stream_l2_drift": 4.5,
                "topk2_anchor_ce_drift": -0.9,
                "topk1_anchor_ce_drift": -0.91,
                "topk2_commutator_anchor_logit_mse": 0.24,
                "topk1_commutator_anchor_logit_mse": 0.01,
            }
        )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "promoted_topk2_router_default_retention_reference",
                "probe_rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fingerprint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "causal_column_fingerprint_established",
                "audit": {
                    "functional_churn": [
                        {
                            "variant": "baseline",
                            "load_balance_weight": 0.0,
                            "adjacent_support_identity_churn_fraction": 1.0,
                            "previous_support_changed_logit_mse_mean": 0.3,
                            "previous_support_changed_residual_l2_mean": 6.0,
                        },
                        {
                            "variant": "rank_matched_topk1_contextual",
                            "load_balance_weight": 0.0,
                            "adjacent_support_identity_churn_fraction": 1.0,
                            "previous_support_changed_logit_mse_mean": 0.32,
                            "previous_support_changed_residual_l2_mean": 6.7,
                        },
                    ]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
