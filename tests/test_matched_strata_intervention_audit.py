from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.matched_strata_intervention_audit import (
    run_matched_strata_intervention_audit,
)


class MatchedStrataInterventionAuditTest(unittest.TestCase):
    def test_matched_strata_audit_prefers_rank_matched_topk1_when_ce_is_better(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audit_dir = tmp_path / "source"
            _write_source_audit(audit_dir)

            summary = run_matched_strata_intervention_audit(
                audit_dir,
                tmp_path / "matched",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "prefer_rank_matched_topk1_for_causal_audits",
            )
            evidence = summary["evidence"]
            self.assertEqual(evidence["matched_strata_count"], 2)
            self.assertTrue(
                evidence["rank_matched_topk1_router_ce_better_than_topk2"]
            )
            self.assertGreater(evidence["topk2_pair_synergy_mean_across_strata"], 0.0)
            self.assertTrue((tmp_path / "matched" / "summary.json").is_file())
            self.assertTrue((tmp_path / "matched" / "matched_strata.csv").is_file())
            self.assertTrue((tmp_path / "matched" / "notes.md").is_file())

            with (tmp_path / "matched" / "matched_strata.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("topk2_pair_synergy_mean", rows[0])
            self.assertIn("topk1_singleton_gain_mean", rows[0])

    def test_matched_strata_audit_fails_without_topk1_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audit_dir = tmp_path / "source"
            _write_source_audit(audit_dir, include_topk1_rows=False)

            summary = run_matched_strata_intervention_audit(
                audit_dir,
                tmp_path / "matched",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "insufficient_evidence")
            self.assertIn(
                "topk1_interventions",
                [failure["field"] for failure in summary["evidence"]["failures"]],
            )


def _write_source_audit(audit_dir: Path, *, include_topk1_rows: bool = True) -> None:
    audit_dir.mkdir(parents=True)
    summary = {
        "audit": {
            "variants": [
                {
                    "variant": "baseline",
                    "alpha0_ce_loss": 2.9,
                },
                {
                    "variant": "rank_matched_topk1_contextual",
                    "alpha0_ce_loss": 2.8,
                },
            ],
            "functional_churn": [
                {
                    "variant": "baseline",
                    "previous_support_changed_logit_mse_mean": 0.3,
                },
                {
                    "variant": "rank_matched_topk1_contextual",
                    "previous_support_changed_logit_mse_mean": 0.32,
                },
            ],
        }
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "router_loss",
        "fixed_support_loss_delta",
        "pair_gain",
        "singleton_left_gain",
        "pair_synergy",
        "pair_value_cosine",
    ]
    rows = [
        {
            "variant": "baseline",
            "intervention": "fixed_dominant_router_support",
            "position_bin": "all",
            "token_class": "all",
            "router_loss": 2.9,
            "fixed_support_loss_delta": 1.2,
            "pair_gain": 0.2,
            "singleton_left_gain": -0.1,
            "pair_synergy": 0.2,
            "pair_value_cosine": 0.1,
        },
        {
            "variant": "baseline",
            "intervention": "fixed_dominant_router_support",
            "position_bin": "even",
            "token_class": "all",
            "router_loss": 2.85,
            "fixed_support_loss_delta": 1.1,
            "pair_gain": 0.15,
            "singleton_left_gain": -0.05,
            "pair_synergy": 0.18,
            "pair_value_cosine": 0.2,
        },
    ]
    if include_topk1_rows:
        rows.extend(
            [
                {
                    "variant": "rank_matched_topk1_contextual",
                    "intervention": "fixed_dominant_router_singleton",
                    "position_bin": "all",
                    "token_class": "all",
                    "router_loss": 2.8,
                    "fixed_support_loss_delta": 1.0,
                    "pair_gain": "",
                    "singleton_left_gain": -0.08,
                    "pair_synergy": "",
                    "pair_value_cosine": 4.5,
                },
                {
                    "variant": "rank_matched_topk1_contextual",
                    "intervention": "fixed_dominant_router_singleton",
                    "position_bin": "even",
                    "token_class": "all",
                    "router_loss": 2.75,
                    "fixed_support_loss_delta": 1.05,
                    "pair_gain": "",
                    "singleton_left_gain": -0.02,
                    "pair_synergy": "",
                    "pair_value_cosine": 4.6,
                },
            ]
        )
    with (audit_dir / "pair_interventions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
