from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_support_retention_gap_selector import (
    INSUFFICIENT_EVIDENCE,
    SELECT_FUNCTIONAL_CHURN_CONTROLS,
    run_promoted_topk2_support_retention_gap_selector,
)


class PromotedTopk2SupportRetentionGapSelectorTest(unittest.TestCase):
    def test_selects_functional_churn_controls_from_existing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_promoted_topk2_support_retention_gap_selector(
                retention_reference_dir=paths["retention"],
                support_quality_dir=paths["support_quality"],
                load_balance_closeout_dir=paths["load_balance"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SELECT_FUNCTIONAL_CHURN_CONTROLS)
            self.assertTrue(summary["signals"]["support_selection_good"])
            self.assertTrue(summary["signals"]["load_balance_closed"])
            self.assertTrue(summary["signals"]["support_churn_high"])
            self.assertTrue(summary["signals"]["logit_churn_gap_low"])
            self.assertTrue(summary["signals"]["commutator_gap_high"])
            self.assertGreater(
                summary["metrics"]["topk2_to_topk1_commutator_ratio"],
                20.0,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_retention_metric_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            summary_path = paths["retention"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            del summary["aggregates"]["mean_topk2_support_churn"]
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = run_promoted_topk2_support_retention_gap_selector(
                retention_reference_dir=paths["retention"],
                support_quality_dir=paths["support_quality"],
                load_balance_closeout_dir=paths["load_balance"],
                out_dir=root / "report",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                "metrics.mean_topk2_support_churn",
                {failure.get("field") for failure in report["failures"]},
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "retention": root / "retention",
        "support_quality": root / "support_quality",
        "load_balance": root / "load_balance",
    }
    for path in paths.values():
        path.mkdir()
    _write_json(
        paths["retention"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_router_default_retention_reference",
            "aggregates": {
                "mean_topk2_support_churn": 0.85,
                "mean_topk2_support_churn_minus_topk1": 0.84,
                "mean_topk2_logit_churn_minus_topk1": 0.02,
                "mean_topk2_commutator_anchor_logit_mse": 0.24,
            },
            "probe_rows": [
                {
                    "topk1_commutator_anchor_logit_mse": 0.01,
                    "topk1_anchor_residual_stream_l2_drift": 4.5,
                    "topk2_anchor_residual_stream_l2_drift": 4.7,
                },
                {
                    "topk1_commutator_anchor_logit_mse": 0.01,
                    "topk1_anchor_residual_stream_l2_drift": 4.6,
                    "topk2_anchor_residual_stream_l2_drift": 4.8,
                },
            ],
        },
    )
    _write_json(
        paths["support_quality"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_support_selection_quality_established",
            "metrics": {
                "oracle_support_regret": 0.002,
                "oracle_support_regret_positive_fraction": 0.05,
                "router_improvement_over_best_global_fixed_support": 1.1,
            },
        },
    )
    _write_json(
        paths["load_balance"] / "summary.json",
        {
            "status": "pass",
            "decision": "keep_load_balance_opt_in_branch_closed",
            "metrics": {
                "min_used_column_gain": 1,
                "max_mean_abs_force_delta_gain": 0.09,
            },
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
