from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_support_selection_quality_audit import (
    INSUFFICIENT_EVIDENCE,
    PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
    run_promoted_topk2_support_selection_quality_audit,
)


class PromotedTopk2SupportSelectionQualityAuditTest(unittest.TestCase):
    def test_establishes_support_selection_quality_from_existing_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_promoted_topk2_support_selection_quality_audit(
                exhaustive_audit_dir=paths["exhaustive"],
                exhaustive_report_dir=paths["exhaustive_report"],
                retention_reference_dir=paths["retention"],
                gate_audit_dir=paths["gate"],
                column_redundancy_dir=paths["column_redundancy"],
                dead_column_probe_dir=paths["dead_probe"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
            )
            self.assertTrue(summary["signals"]["oracle_regret_small"])
            self.assertTrue(summary["signals"]["positive_regret_fraction_low"])
            self.assertTrue(summary["signals"]["router_beats_best_global_fixed_pair"])
            self.assertTrue(summary["signals"]["oracle_contextual_selector_is_upper_bound"])
            self.assertTrue(
                summary["signals"]["deployable_contextual_support_head_positive_but_small"]
            )
            self.assertTrue(summary["signals"]["dead_columns_recruited_without_ce_hurt"])
            self.assertAlmostEqual(
                summary["metrics"]["router_improvement_over_best_global_fixed_support"],
                1.13,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_oracle_regret_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            summary_path = paths["exhaustive"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            del summary["audit"]["oracle_support_regret"]
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = run_promoted_topk2_support_selection_quality_audit(
                exhaustive_audit_dir=paths["exhaustive"],
                exhaustive_report_dir=paths["exhaustive_report"],
                retention_reference_dir=paths["retention"],
                gate_audit_dir=paths["gate"],
                column_redundancy_dir=paths["column_redundancy"],
                dead_column_probe_dir=paths["dead_probe"],
                out_dir=root / "report",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                "metrics.oracle_support_regret",
                {failure.get("field") for failure in report["failures"]},
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "exhaustive": root / "exhaustive",
        "exhaustive_report": root / "exhaustive_report",
        "retention": root / "retention",
        "gate": root / "gate",
        "column_redundancy": root / "column_redundancy",
        "dead_probe": root / "dead_probe",
    }
    for path in paths.values():
        path.mkdir()
    _write_json(
        paths["exhaustive"] / "summary.json",
        {
            "status": "ok",
            "audit": {
                "config_path": "configs/test.yaml",
                "dataset": "tiny_shakespeare_word",
                "num_columns": 24,
                "top_k": 2,
                "support_router": "contextual_mlp",
                "router_loss": 2.87,
                "oracle_loss": 2.865,
                "oracle_support_regret": 0.005,
                "oracle_support_regret_positive_fraction": 0.05,
                "best_global_fixed_support": "0,3",
                "best_global_fixed_support_loss": 4.0,
                "dominant_router_support": "14,18",
                "dominant_router_support_regret": 0.08,
                "support_audit": {
                    "used_columns": 20,
                    "dead_columns": 4,
                    "unique_support_sets": 50,
                },
                "router_oracle_target_contextual_diagnostic": {
                    "holdout": {
                        "oracle_target_accuracy": 0.97,
                        "oracle_gap_recovery_fraction": 1.0,
                        "selector_minus_router_loss": -0.002,
                    }
                },
                "router_oracle_target_diagnostic": {
                    "holdout": {"oracle_gap_recovery_fraction": -70.0}
                },
                "router_oracle_target_nonlinear_diagnostic": {
                    "holdout": {"oracle_gap_recovery_fraction": -73.0}
                },
                "contextual_router_support_head": {
                    "holdout": {
                        "oracle_gap_recovery_fraction": 0.19,
                        "intervention_minus_router_loss": -0.0004,
                    }
                },
                "contextual_router_support_intervention": {
                    "holdout": {
                        "oracle_gap_recovery_fraction": 1.0,
                        "intervention_minus_router_loss": -0.002,
                    }
                },
            },
        },
    )
    _write_json(
        paths["exhaustive_report"] / "decision_report.json",
        {"status": "pass", "decision": "diagnose_exhaustive_support_audit"},
    )
    _write_json(
        paths["retention"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_router_default_retention_reference",
        },
    )
    _write_json(
        paths["gate"] / "summary.json",
        {
            "status": "pass",
            "decision": "deployable_context_gate_suppression_calibration_failed",
        },
    )
    _write_json(paths["column_redundancy"] / "summary.json", {"status": "ok"})
    _write_json(
        paths["dead_probe"] / "summary.json",
        {
            "status": "ok",
            "probe": {"decision": {"status": "recruited_without_ce_hurt"}},
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
