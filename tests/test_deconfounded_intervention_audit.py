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
            self.assertTrue(evidence["ce_guardrail_passed"])
            self.assertLess(evidence["topk2_fixed_delta_minus_topk1_mean"], 0.0)
            self.assertLess(evidence["topk2_logit_mse_minus_topk1_mean"], 0.0)
            self.assertFalse(evidence["per_token_pair_synergy_available"])
            self.assertTrue((root / "deconfounded" / "summary.json").is_file())
            self.assertTrue(
                (root / "deconfounded" / "matched_deconfounded_strata.csv").is_file()
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
                "topk2_coarse_synergy_not_supported_after_deconfounding",
            )
            self.assertLess(
                summary["evidence"]["topk2_fixed_support_cleaner_strata_fraction"],
                0.8,
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
        "position_bin",
        "token_class",
        "router_support_count",
        "router_loss",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
        "residual_norm_bin",
        "residual_gain_bin",
    ]
    if include_active_rank:
        fields.append("active_rank_proxy")
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
            topk2_delta,
            topk2_mse,
            2,
            include_active_rank=include_active_rank,
        ),
        _per_token_row(
            "rank_matched_topk1_contextual",
            "fixed_dominant_router_singleton",
            "even",
            "common_target",
            "low",
            "low",
            11,
            0.5,
            0.3,
            1,
            include_active_rank=include_active_rank,
        ),
        _per_token_row(
            "baseline",
            "fixed_dominant_router_support",
            "odd",
            "rare_target",
            "mid",
            "high",
            20,
            topk2_delta,
            topk2_mse,
            2,
            include_active_rank=include_active_rank,
        ),
        _per_token_row(
            "rank_matched_topk1_contextual",
            "fixed_dominant_router_singleton",
            "odd",
            "rare_target",
            "mid",
            "high",
            20,
            0.6,
            0.4,
            1,
            include_active_rank=include_active_rank,
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
    fixed_delta: float,
    logit_mse: float,
    active_rank_proxy: int,
    *,
    include_active_rank: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "intervention": intervention,
        "position_bin": position_bin,
        "token_class": token_class,
        "router_support_count": router_support_count,
        "router_loss": 2.0,
        "fixed_support_loss_delta": fixed_delta,
        "fixed_support_logit_mse": logit_mse,
        "fixed_support_residual_stream_l2_delta": 1.0,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": residual_gain_bin,
    }
    if include_active_rank:
        row["active_rank_proxy"] = active_rank_proxy
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
