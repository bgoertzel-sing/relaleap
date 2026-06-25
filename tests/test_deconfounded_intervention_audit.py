from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.deconfounded_intervention_audit import (
    run_deconfounded_intervention_audit,
)


class DeconfoundedInterventionAuditTest(unittest.TestCase):
    def test_topk2_survives_deconfounding_inside_ce_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            _write_source_audit(audit_dir)

            summary = run_deconfounded_intervention_audit(
                audit_dir,
                root / "deconfounded",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "topk2_causal_metrics_survive_deconfounding_with_ce_guardrail",
            )
            evidence = summary["evidence"]
            self.assertEqual(evidence["matched_deconfounded_strata_count"], 2)
            self.assertEqual(evidence["matched_exact_context_count"], 2)
            self.assertTrue(evidence["ce_guardrail_passed"])
            self.assertLess(evidence["topk2_fixed_delta_minus_topk1_mean"], 0.0)
            self.assertLess(evidence["topk2_logit_mse_minus_topk1_mean"], 0.0)
            self.assertGreater(
                evidence[
                    "topk2_incremental_pair_gain_minus_topk1_singleton_mean"
                ],
                0.0,
            )
            self.assertEqual(
                evidence["topk2_incremental_pair_gain_positive_strata_fraction"],
                1.0,
            )
            self.assertFalse(evidence["per_token_pair_synergy_available"])
            self.assertTrue((root / "deconfounded" / "summary.json").is_file())
            self.assertTrue(
                (root / "deconfounded" / "matched_deconfounded_strata.csv").is_file()
            )
            self.assertTrue(
                (root / "deconfounded" / "paired_exact_context_deltas.csv").is_file()
            )
            self.assertTrue((root / "deconfounded" / "notes.md").is_file())

    def test_topk2_not_supported_after_deconfounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            _write_source_audit(audit_dir, topk2_cleaner=False)

            summary = run_deconfounded_intervention_audit(
                audit_dir,
                root / "deconfounded",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "topk2_comparative_causal_cooperation_not_supported",
            )
            self.assertLess(
                summary["evidence"]["topk2_fixed_support_cleaner_strata_fraction"],
                0.8,
            )

    def test_pair_synergy_can_survive_when_cleanliness_bar_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            _write_source_audit(
                audit_dir,
                topk2_cleaner=False,
                include_per_token_synergy=True,
            )

            summary = run_deconfounded_intervention_audit(
                audit_dir,
                root / "deconfounded",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "topk2_pair_synergy_survives_deconfounding_but_cleanliness_bar_fails",
            )
            evidence = summary["evidence"]
            self.assertTrue(evidence["per_token_pair_synergy_available"])
            self.assertGreater(evidence["deconfounded_topk2_pair_synergy_mean"], 0.0)
            self.assertEqual(
                evidence["deconfounded_topk2_pair_synergy_positive_strata_fraction"],
                1.0,
            )
            self.assertEqual(
                evidence["topk2_incremental_pair_gain_positive_strata_fraction"],
                1.0,
            )

    def test_fails_without_required_per_token_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            _write_source_audit(audit_dir, include_active_rank=False)

            summary = run_deconfounded_intervention_audit(
                audit_dir,
                root / "deconfounded",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "insufficient_evidence")
            self.assertIn(
                "per_token_matching_fields",
                [failure["field"] for failure in summary["evidence"]["failures"]],
            )


def _write_source_audit(
    audit_dir: Path,
    *,
    topk2_cleaner: bool = True,
    include_active_rank: bool = True,
    include_per_token_synergy: bool = False,
) -> None:
    audit_dir.mkdir(parents=True)
    summary = {
        "audit": {
            "variants": [
                {"variant": "baseline", "alpha0_ce_loss": 2.94},
                {
                    "variant": "rank_matched_topk1_contextual",
                    "alpha0_ce_loss": 2.9,
                },
            ]
        }
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    pair_fields = [
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "pair_synergy",
    ]
    _write_csv(
        audit_dir / "pair_interventions.csv",
        pair_fields,
        [
            {
                "variant": "baseline",
                "intervention": "fixed_dominant_router_support",
                "position_bin": "all",
                "token_class": "all",
                "pair_synergy": "0.2",
            }
        ],
    )
    fields = [
        "variant",
        "intervention",
        "batch_index",
        "position_index",
        "token_index",
        "target_token",
        "position_bin",
        "token_class",
        "router_support_count",
        "router_loss",
        "pair_gain",
        "singleton_left_gain",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
        "residual_norm_bin",
        "residual_gain_bin",
    ]
    if include_active_rank:
        fields.append("active_rank_proxy")
    if include_per_token_synergy:
        fields.append("pair_synergy")
    topk2_delta = 0.2 if topk2_cleaner else 0.8
    topk2_mse = 0.1 if topk2_cleaner else 0.5
    rows = [
        _per_token_row(
            "baseline",
            "fixed_dominant_router_support",
            "even",
            "common_target",
            "low",
            "low",
            10,
            0.5,
            0.1,
            topk2_delta,
            topk2_mse,
            2,
            batch_index=0,
            position_index=0,
            token_index=100,
            target_token="a",
            include_active_rank=include_active_rank,
            pair_synergy=0.15 if include_per_token_synergy else None,
        ),
        _per_token_row(
            "rank_matched_topk1_contextual",
            "fixed_dominant_router_singleton",
            "even",
            "common_target",
            "low",
            "low",
            11,
            None,
            0.1,
            0.5,
            0.3,
            1,
            batch_index=0,
            position_index=0,
            token_index=100,
            target_token="a",
            include_active_rank=include_active_rank,
            pair_synergy=None,
        ),
        _per_token_row(
            "baseline",
            "fixed_dominant_router_support",
            "odd",
            "rare_target",
            "mid",
            "high",
            20,
            0.7,
            0.2,
            topk2_delta,
            topk2_mse,
            2,
            batch_index=0,
            position_index=1,
            token_index=101,
            target_token="b",
            include_active_rank=include_active_rank,
            pair_synergy=0.25 if include_per_token_synergy else None,
        ),
        _per_token_row(
            "rank_matched_topk1_contextual",
            "fixed_dominant_router_singleton",
            "odd",
            "rare_target",
            "mid",
            "high",
            20,
            None,
            0.2,
            0.6,
            0.4,
            1,
            batch_index=0,
            position_index=1,
            token_index=101,
            target_token="b",
            include_active_rank=include_active_rank,
            pair_synergy=None,
        ),
    ]
    _write_csv(audit_dir / "per_token_pair_interventions.csv", fields, rows)


def _per_token_row(
    variant: str,
    intervention: str,
    position_bin: str,
    token_class: str,
    residual_norm_bin: str,
    residual_gain_bin: str,
    router_support_count: int,
    pair_gain: float | None,
    singleton_left_gain: float,
    fixed_delta: float,
    logit_mse: float,
    active_rank_proxy: int,
    *,
    batch_index: int,
    position_index: int,
    token_index: int,
    target_token: str,
    include_active_rank: bool,
    pair_synergy: float | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "intervention": intervention,
        "batch_index": batch_index,
        "position_index": position_index,
        "token_index": token_index,
        "target_token": target_token,
        "position_bin": position_bin,
        "token_class": token_class,
        "router_support_count": router_support_count,
        "router_loss": 2.0,
        "pair_gain": "" if pair_gain is None else pair_gain,
        "singleton_left_gain": singleton_left_gain,
        "fixed_support_loss_delta": fixed_delta,
        "fixed_support_logit_mse": logit_mse,
        "fixed_support_residual_stream_l2_delta": 1.0,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": residual_gain_bin,
    }
    if include_active_rank:
        row["active_rank_proxy"] = active_rank_proxy
    if pair_synergy is not None:
        row["pair_synergy"] = pair_synergy
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
