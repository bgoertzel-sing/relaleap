from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_load_balance_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    KEEP_LOAD_BALANCE_OPT_IN_CLOSED,
    run_promoted_topk2_load_balance_closeout_report,
)


class PromotedTopk2LoadBalanceCloseoutReportTest(unittest.TestCase):
    def test_closes_load_balance_branch_as_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_promoted_topk2_load_balance_closeout_report(
                probe_dirs=(paths["probe1"], paths["probe2"]),
                causal_dirs=(paths["causal1"], paths["causal2"]),
                load_balance_report_dir=paths["load_balance_report"],
                support_quality_dir=paths["support_quality"],
                gate_audit_dir=paths["gate"],
                retention_reference_dir=paths["retention"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], KEEP_LOAD_BALANCE_OPT_IN_CLOSED)
            self.assertFalse(summary["promote_router_load_balance_default"])
            self.assertTrue(summary["signals"]["all_probes_recruited_without_ce_hurt"])
            self.assertTrue(summary["signals"]["prior_report_keeps_load_balance_opt_in"])
            self.assertTrue(summary["signals"]["topk2_support_selection_quality_established"])
            self.assertEqual(summary["metrics"]["min_used_column_gain"], 1)
            self.assertEqual(summary["metrics"]["max_used_column_gain"], 5)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_support_quality_context_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["support_quality"] / "summary.json").unlink()

            summary = run_promoted_topk2_load_balance_closeout_report(
                probe_dirs=(paths["probe1"], paths["probe2"]),
                causal_dirs=(paths["causal1"], paths["causal2"]),
                load_balance_report_dir=paths["load_balance_report"],
                support_quality_dir=paths["support_quality"],
                gate_audit_dir=paths["gate"],
                retention_reference_dir=paths["retention"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                "support_quality.decision",
                {failure.get("field") for failure in summary["failures"]},
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "probe1": root / "probe1",
        "probe2": root / "probe2",
        "causal1": root / "causal1",
        "causal2": root / "causal2",
        "load_balance_report": root / "load_balance_report",
        "support_quality": root / "support_quality",
        "gate": root / "gate",
        "retention": root / "retention",
    }
    for path in paths.values():
        path.mkdir()
    _write_probe(paths["probe1"], baseline_used=19, selected_used=24)
    _write_probe(paths["probe2"], baseline_used=23, selected_used=24)
    _write_causal(paths["causal1"])
    _write_causal(paths["causal2"])
    _write_json(
        paths["load_balance_report"] / "decision_report.json",
        {
            "status": "pass",
            "decision": "keep_router_load_balance_probe_opt_in",
            "promote_router_load_balance_default": False,
        },
    )
    _write_json(
        paths["support_quality"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_support_selection_quality_established",
        },
    )
    _write_json(
        paths["gate"] / "summary.json",
        {
            "status": "pass",
            "decision": "deployable_context_gate_suppression_calibration_failed",
        },
    )
    _write_json(
        paths["retention"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_router_default_retention_reference",
        },
    )
    return paths


def _write_probe(path: Path, *, baseline_used: int, selected_used: int) -> None:
    _write_json(
        path / "summary.json",
        {
            "status": "ok",
            "config_path": "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml",
            "probe": {
                "baseline_variant": "baseline",
                "decision": {
                    "status": "recruited_without_ce_hurt",
                    "baseline_alpha0_ce_loss": 2.9,
                    "selected_alpha0_ce_loss": 2.89,
                    "baseline_used_columns": baseline_used,
                    "selected_used_columns": selected_used,
                    "selected_variant": "load_balance_0.0125",
                },
                "variants": [
                    {
                        "variant": "baseline",
                        "load_balance_weight": 0.0,
                        "dead_columns": 24 - baseline_used,
                        "oracle_support_regret": 0.005,
                    },
                    {
                        "variant": "load_balance_0.0125",
                        "load_balance_weight": 0.0125,
                        "dead_columns": 24 - selected_used,
                        "oracle_support_regret": 0.001,
                    },
                ],
            },
        },
    )


def _write_causal(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "status": "ok",
            "audit": {
                "variant_summaries": [
                    {
                        "variant": "baseline",
                        "mean_abs_ablate_loss_delta": 0.05,
                        "mean_abs_force_loss_delta": 1.4,
                    },
                    {
                        "variant": "load_balance_0.0125",
                        "selected_for_load_balance_bracket": True,
                        "mean_abs_ablate_loss_delta": 0.051,
                        "mean_abs_force_loss_delta": 1.39,
                    },
                ]
            },
        },
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
