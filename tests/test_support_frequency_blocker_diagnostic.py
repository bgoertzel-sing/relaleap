from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.support_frequency_blocker_diagnostic import (
    run_support_frequency_blocker_diagnostic,
)


class SupportFrequencyBlockerDiagnosticTest(unittest.TestCase):
    def test_reports_non_claim_blocker_and_nearest_distances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            audit_dir.mkdir()
            _write_csv(
                audit_dir / "support_frequency_candidate_controls.csv",
                [
                    _row("baseline", 0, "2,9", "0,1", 18, 1.5),
                    _row("baseline", 0, "2,9", "0,3", 17, 0.25),
                    _row("baseline", 1, "3,4", "0,2", 5, 0.75),
                ],
            )

            summary = run_support_frequency_blocker_diagnostic(
                audit_dir,
                root / "diagnostic",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "support_frequency_percentile_claim_remains_blocked_by_support_count_caliper",
            )
            evidence = summary["evidence"]
            self.assertFalse(evidence["claim_bearing"])
            self.assertEqual(evidence["candidate_row_count"], 3)
            self.assertEqual(evidence["anchor_count"], 2)
            self.assertEqual(evidence["calipered_candidate_row_count"], 0)
            self.assertEqual(
                evidence["failed_caliper_dimension_counts"],
                {"support_count_caliper": 3},
            )
            nearest = evidence["per_anchor_nearest_neighbor_summary"]
            self.assertEqual(nearest["support_count_abs_difference"]["min"], 5.0)
            self.assertEqual(nearest["support_count_abs_difference"]["max"], 17.0)
            relaxed = evidence["relaxed_support_count_caliper_diagnostics"]
            self.assertEqual(relaxed[0]["support_count_caliper"], 1)
            self.assertEqual(relaxed[0]["candidate_row_count"], 0)
            self.assertEqual(relaxed[3]["support_count_caliper"], 8)
            self.assertEqual(relaxed[3]["candidate_row_count"], 1)
            self.assertTrue((root / "diagnostic" / "summary.json").is_file())
            self.assertTrue(
                (root / "diagnostic" / "per_anchor_blockers.csv").is_file()
            )
            self.assertTrue((root / "diagnostic" / "notes.md").is_file())
            with (root / "diagnostic" / "per_anchor_blockers.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                anchor_rows = list(csv.DictReader(handle))
            self.assertEqual(len(anchor_rows), 2)
            self.assertIn("nearest_support_count_abs_difference", anchor_rows[0])

    def test_fails_without_candidate_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_support_frequency_blocker_diagnostic(
                root / "missing",
                root / "diagnostic",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "insufficient_artifacts_for_support_frequency_blocker_diagnostic",
            )
            self.assertIn(
                "support_frequency_candidate_controls_csv",
                [failure["field"] for failure in summary["evidence"]["failures"]],
            )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "load_balance_weight",
        "anchor_index",
        "anchor_intervention",
        "anchor_support",
        "candidate_support",
        "candidate_is_nonrouter",
        "support_count_caliper",
        "within_support_count_caliper",
        "candidate_match_status",
        "anchor_router_support_count",
        "candidate_router_support_count",
        "support_count_difference",
        "support_count_abs_difference",
        "candidate_pool_count",
        "exact_support_count_candidate_count",
        "near_support_count_candidate_count",
        "min_support_count_abs_difference_available",
        "calipered_candidate_count",
        "included_in_primary_percentile_denominator",
        "anchor_fixed_support_loss",
        "candidate_fixed_support_loss",
        "fixed_support_loss_difference",
        "anchor_pair_gain",
        "candidate_pair_gain",
        "pair_gain_difference",
        "anchor_singleton_gain_sum",
        "candidate_singleton_gain_sum",
        "singleton_gain_sum_difference",
        "anchor_pair_synergy",
        "candidate_pair_synergy",
        "pair_synergy_difference",
        "anchor_pair_value_norm",
        "candidate_pair_value_norm",
        "pair_value_norm_difference",
        "loss_match_rank_within_caliper",
        "singleton_gain_match_rank_within_caliper",
        "residual_norm_match_rank_within_caliper",
        "random_nonrouter_rank_within_caliper",
        "stable_random_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(
    variant: str,
    anchor_index: int,
    anchor_support: str,
    candidate_support: str,
    support_count_abs_difference: int,
    loss_difference: float,
) -> dict[str, object]:
    return {
        "variant": variant,
        "load_balance_weight": 0.0,
        "anchor_index": anchor_index,
        "anchor_intervention": "fixed_dominant_router_support",
        "anchor_support": anchor_support,
        "candidate_support": candidate_support,
        "candidate_is_nonrouter": True,
        "support_count_caliper": 1,
        "within_support_count_caliper": False,
        "candidate_match_status": "unmatched_excluded_by_support_count_caliper",
        "anchor_router_support_count": support_count_abs_difference,
        "candidate_router_support_count": 0,
        "support_count_difference": -support_count_abs_difference,
        "support_count_abs_difference": support_count_abs_difference,
        "candidate_pool_count": 3,
        "exact_support_count_candidate_count": 0,
        "near_support_count_candidate_count": 0,
        "min_support_count_abs_difference_available": support_count_abs_difference,
        "calipered_candidate_count": 0,
        "included_in_primary_percentile_denominator": False,
        "anchor_fixed_support_loss": 4.0,
        "candidate_fixed_support_loss": 4.0 + loss_difference,
        "fixed_support_loss_difference": loss_difference,
        "anchor_pair_gain": 0.1,
        "candidate_pair_gain": 0.05,
        "pair_gain_difference": -0.05,
        "anchor_singleton_gain_sum": -0.1,
        "candidate_singleton_gain_sum": -0.2,
        "singleton_gain_sum_difference": -0.1,
        "anchor_pair_synergy": 0.2,
        "candidate_pair_synergy": 0.25,
        "pair_synergy_difference": 0.05,
        "anchor_pair_value_norm": 6.0,
        "candidate_pair_value_norm": 5.75,
        "pair_value_norm_difference": -0.25,
        "loss_match_rank_within_caliper": "",
        "singleton_gain_match_rank_within_caliper": "",
        "residual_norm_match_rank_within_caliper": "",
        "random_nonrouter_rank_within_caliper": "",
        "stable_random_score": 1,
    }
