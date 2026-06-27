from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_margin_aware_transfer_objective_design import (
    REQUIRED_ARTIFACTS,
    run_acsr_margin_aware_transfer_objective_design,
)


class ACSRMarginAwareTransferObjectiveDesignTest(unittest.TestCase):
    def test_records_margin_aware_objective_from_supported_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "summary.json"
            values = root / "value_student_support_synthesis.csv"
            strata = root / "stratified_transfer_synthesis.csv"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_json(
                synthesis,
                {
                    "status": "pass",
                    "claim_status": "cross_value_support_transfer_supported_not_promoted",
                    "aggregate_metrics": {
                        "all_partner_beats_required_nulls": True,
                        "mean_partner_delta_vs_token_position_null": -0.11,
                        "mean_high_regret_partner_delta_vs_token_position_null": -0.46,
                        "mean_disagreement_partner_delta_vs_token_position_null": -0.08,
                        "mean_low_margin_partner_delta_vs_token_position_null": 0.0,
                        "mean_partner_delta_vs_token_position_null_per_residual_l2": -1.5,
                        "residual_norm_control_available": True,
                        "stratified_transfer_available": True,
                    },
                },
            )
            _write_csv(
                values,
                [
                    {"packet": "packet1", "value_student": "acsr_student", "status": "available"},
                    {
                        "packet": "packet1",
                        "value_student": "parameter_matched_direct_causal_mlp_student",
                        "status": "available",
                    },
                    {"packet": "packet2", "value_student": "acsr_student", "status": "available"},
                    {
                        "packet": "packet2",
                        "value_student": "parameter_matched_direct_causal_mlp_student",
                        "status": "available",
                    },
                ],
            )
            _write_csv(
                strata,
                [
                    _stratum("oracle_regret", "top_quartile_token_position_null_regret"),
                    _stratum("support_disagreement", "partner_vs_own"),
                    _stratum("support_disagreement", "partner_vs_token_position_null"),
                    _stratum("partner_support_margin_bin", "high_margin"),
                    _stratum("partner_support_margin_bin", "low_margin"),
                ],
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Design a bounded transfer objective.",
                        "verdict: GO",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_margin_aware_transfer_objective_design(
                synthesis=synthesis,
                value_synthesis=values,
                stratified_synthesis=strata,
                strategy_review=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "acsr_margin_aware_transfer_objective_design_recorded",
            )
            self.assertEqual(summary["claim_status"], "objective_design_only_not_promoted")
            terms = {row["term"]: row for row in summary["objective_terms"]}
            self.assertEqual(terms["high_regret_cross_value_focus"]["weight"], 1.0)
            self.assertEqual(terms["support_disagreement_focus"]["weight"], 0.5)
            self.assertEqual(terms["low_margin_suppression"]["weight"], 0.0)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertIn("local low-step transfer-objective probe", summary["selected_next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

    def test_missing_supported_synthesis_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_margin_aware_transfer_objective_design(
                synthesis=root / "missing-summary.json",
                value_synthesis=root / "missing-values.csv",
                stratified_synthesis=root / "missing-strata.csv",
                strategy_review=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_margin_aware_transfer_objective_design_failed_closed",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "cross_value_transfer_supported_not_promoted"
                    for row in summary["failures"]
                )
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _stratum(stratum_type: str, stratum_value: str) -> dict[str, object]:
    return {
        "packet": "packet1",
        "value_student": "acsr_student",
        "status": "available",
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "partner_delta_vs_token_position_null": -0.1,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
