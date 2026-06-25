from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_causal_separability_audit import (
    ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_causal_separability_audit,
)


class ActiveTopk1CausalSeparabilityAuditTest(unittest.TestCase):
    def test_establishes_active_topk1_separability_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "deconfounded"
            _write_deconfounded_audit(audit_dir)

            summary = run_active_topk1_causal_separability_audit(
                audit_dir,
                root / "active_topk1",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                ACTIVE_TOPK1_SEPARABILITY_AUDIT_ESTABLISHED,
            )
            metrics = summary["evidence"]["metrics"]
            self.assertEqual(metrics["matched_deconfounded_strata_count"], 2)
            self.assertEqual(metrics["paired_exact_context_count"], 2)
            self.assertEqual(metrics["topk1_active_rank_proxy_values"], ["1"])
            self.assertTrue(summary["evidence"]["signals"]["topk1_ce_primary"])
            self.assertAlmostEqual(metrics["topk1_singleton_gain_mean"], 0.15)
            self.assertAlmostEqual(
                metrics["topk1_fixed_support_loss_delta_negative_strata_fraction"],
                0.5,
            )
            self.assertTrue((root / "active_topk1" / "summary.json").is_file())
            self.assertTrue(
                (
                    root
                    / "active_topk1"
                    / "topk1_separability_by_stratum.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "active_topk1"
                    / "topk1_separability_by_context.csv"
                ).is_file()
            )
            self.assertTrue((root / "active_topk1" / "notes.md").is_file())

    def test_fails_when_topk1_is_not_ce_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "deconfounded"
            _write_deconfounded_audit(audit_dir, topk1_ce=3.1, topk2_ce=3.0)

            summary = run_active_topk1_causal_separability_audit(
                audit_dir,
                root / "active_topk1",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(summary["evidence"]["signals"]["topk1_ce_primary"])

    def test_fails_without_active_rank_matched_topk1_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "deconfounded"
            _write_deconfounded_audit(audit_dir, topk1_active_rank="1,2")

            summary = run_active_topk1_causal_separability_audit(
                audit_dir,
                root / "active_topk1",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(
                summary["evidence"]["signals"]["active_rank_matched_topk1_rows_present"]
            )


def _write_deconfounded_audit(
    audit_dir: Path,
    *,
    topk1_ce: float = 2.8,
    topk2_ce: float = 2.9,
    topk1_active_rank: str = "1",
) -> None:
    audit_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "topk2_comparative_causal_cooperation_not_supported",
        "evidence": {
            "topk1_alpha0_ce_loss": topk1_ce,
            "topk2_alpha0_ce_loss": topk2_ce,
            "topk2_ce_deficit_vs_topk1": topk2_ce - topk1_ce,
            "matched_exact_context_count": 2,
            "matched_topk1_context_fraction": 1.0,
            "unmatched_topk1_context_count": 0,
            "topk2_incremental_pair_gain_positive_strata_fraction": 0.5,
            "topk2_fixed_support_cleaner_strata_fraction": 0.5,
            "topk2_functional_churn_cleaner_strata_fraction": 0.5,
            "deconfounded_topk2_pair_synergy_mean": 0.2,
        },
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        audit_dir / "matched_deconfounded_strata.csv",
        [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "support_count_bin",
            "matched_exact_context_count",
            "topk1_row_count",
            "topk1_active_rank_proxy",
            "topk1_router_loss_mean",
            "topk1_singleton_gain_mean",
            "topk1_fixed_support_loss_delta_mean",
            "topk1_fixed_support_logit_mse_mean",
            "topk1_residual_stream_l2_delta_mean",
            "topk2_incremental_pair_gain_minus_topk1_singleton",
            "topk2_pair_synergy_mean",
        ],
        [
            {
                "position_bin": "even",
                "token_class": "common",
                "residual_norm_bin": "low",
                "residual_gain_bin": "low",
                "support_count_bin": "low",
                "matched_exact_context_count": 1,
                "topk1_row_count": 2,
                "topk1_active_rank_proxy": topk1_active_rank,
                "topk1_router_loss_mean": 2.0,
                "topk1_singleton_gain_mean": 0.2,
                "topk1_fixed_support_loss_delta_mean": -0.1,
                "topk1_fixed_support_logit_mse_mean": 0.03,
                "topk1_residual_stream_l2_delta_mean": 1.0,
                "topk2_incremental_pair_gain_minus_topk1_singleton": 0.1,
                "topk2_pair_synergy_mean": 0.3,
            },
            {
                "position_bin": "odd",
                "token_class": "rare",
                "residual_norm_bin": "mid",
                "residual_gain_bin": "high",
                "support_count_bin": "mid",
                "matched_exact_context_count": 1,
                "topk1_row_count": 1,
                "topk1_active_rank_proxy": topk1_active_rank,
                "topk1_router_loss_mean": 2.5,
                "topk1_singleton_gain_mean": 0.1,
                "topk1_fixed_support_loss_delta_mean": 0.2,
                "topk1_fixed_support_logit_mse_mean": 0.04,
                "topk1_residual_stream_l2_delta_mean": 1.2,
                "topk2_incremental_pair_gain_minus_topk1_singleton": -0.1,
                "topk2_pair_synergy_mean": 0.1,
            },
        ],
    )
    _write_csv(
        audit_dir / "paired_exact_context_deltas.csv",
        [
            "batch_index",
            "position_index",
            "token_index",
            "target_token",
            "matched_context_stratum_count",
            "topk1_row_count",
            "topk1_router_loss_mean",
            "topk1_singleton_gain_mean",
            "topk2_fixed_delta_minus_topk1",
            "topk2_logit_mse_minus_topk1",
        ],
        [
            {
                "batch_index": 0,
                "position_index": 0,
                "token_index": 0,
                "target_token": 1,
                "matched_context_stratum_count": 1,
                "topk1_row_count": 2,
                "topk1_router_loss_mean": 2.0,
                "topk1_singleton_gain_mean": 0.2,
                "topk2_fixed_delta_minus_topk1": -0.1,
                "topk2_logit_mse_minus_topk1": -0.01,
            },
            {
                "batch_index": 0,
                "position_index": 1,
                "token_index": 1,
                "target_token": 2,
                "matched_context_stratum_count": 1,
                "topk1_row_count": 1,
                "topk1_router_loss_mean": 2.5,
                "topk1_singleton_gain_mean": 0.1,
                "topk2_fixed_delta_minus_topk1": 0.2,
                "topk2_logit_mse_minus_topk1": 0.02,
            },
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
