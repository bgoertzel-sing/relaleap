from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_synergy_null_audit import (
    run_causal_synergy_null_audit,
)


class CausalSynergyNullAuditTest(unittest.TestCase):
    def test_pair_synergy_supported_against_local_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_artifacts(audit_dir, deconfounded_dir)

            summary = run_causal_synergy_null_audit(
                audit_dir,
                root / "null_audit",
                deconfounded_dir=deconfounded_dir,
                bootstrap_samples=200,
                seed=3,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "pair_synergy_supported_against_local_nulls_but_cleaner_bracket_fails",
            )
            evidence = summary["evidence"]
            self.assertTrue(evidence["pair_synergy_supported"])
            self.assertTrue(evidence["sign_flip_synergy_supported"])
            self.assertTrue(evidence["artifact_control_supported"])
            self.assertFalse(evidence["cleaner_causal_bracket_supported"])
            self.assertGreater(evidence["observed_minus_control_synergy_mean"], 0.0)
            diagnostics = evidence["control_match_diagnostics"]
            self.assertEqual(diagnostics["matched_strata_overlap_fraction"], 1.0)
            self.assertAlmostEqual(
                diagnostics["support_count_difference_mean"],
                -1.0,
            )
            self.assertAlmostEqual(
                diagnostics["fixed_support_loss_difference_mean"],
                0.02,
            )
            self.assertTrue((root / "null_audit" / "summary.json").is_file())
            self.assertTrue(
                (root / "null_audit" / "matched_synergy_null_strata.csv").is_file()
            )
            with (root / "null_audit" / "matched_synergy_null_strata.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                strata_rows = list(csv.DictReader(handle))
            self.assertIn(
                "control_support_count_difference_mean",
                strata_rows[0],
            )
            self.assertIn(
                "control_fixed_support_loss_difference_mean",
                strata_rows[0],
            )
            self.assertTrue((root / "null_audit" / "notes.md").is_file())

    def test_fails_without_pair_synergy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_artifacts(audit_dir, deconfounded_dir, include_synergy=False)

            summary = run_causal_synergy_null_audit(
                audit_dir,
                root / "null_audit",
                deconfounded_dir=deconfounded_dir,
                bootstrap_samples=20,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "insufficient_evidence")
            self.assertIn(
                "per_token_pair_synergy_fields",
                [failure["field"] for failure in summary["evidence"]["failures"]],
            )


def _write_artifacts(
    audit_dir: Path,
    deconfounded_dir: Path,
    *,
    include_synergy: bool = True,
) -> None:
    audit_dir.mkdir(parents=True)
    deconfounded_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "topk2_pair_synergy_survives_deconfounding_but_cleanliness_bar_fails",
        "evidence": {
            "topk2_ce_deficit_vs_topk1": 0.04,
            "topk2_fixed_support_cleaner_strata_fraction": 0.7,
            "topk2_functional_churn_cleaner_strata_fraction": 0.75,
        },
    }
    (deconfounded_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    strata = [
        ("even", "common_target", "low", 4, "low"),
        ("odd", "rare_target", "mid", 6, "low"),
        ("even", "rare_target", "high", 8, "low"),
        ("odd", "common_target", "low", 10, "mid"),
        ("even", "common_target", "mid", 12, "mid"),
        ("odd", "rare_target", "high", 14, "high"),
    ]
    _write_csv(
        deconfounded_dir / "matched_deconfounded_strata.csv",
        [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "support_count_bin",
        ],
        [
            _matched_row(position_bin, token_class, norm_bin, "high", support_bin)
            for position_bin, token_class, norm_bin, _support_count, support_bin in strata
        ],
    )
    fields = [
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "router_support_count",
        "anchor_support",
        "control_support",
        "support_count_difference",
        "fixed_support_loss_difference",
        "pair_gain",
        "singleton_left_gain",
        "singleton_right_gain",
        "residual_norm_bin",
        "residual_gain_bin",
    ]
    if include_synergy:
        fields.append("pair_synergy")
    rows = []
    for position_bin, token_class, norm_bin, support_count, _support_bin in strata:
        rows.append(
            _per_token_row(
                "baseline",
                "fixed_dominant_router_support",
                position_bin,
                token_class,
                norm_bin,
                support_count,
                support_count_difference=0,
                fixed_support_loss_difference=0.0,
                pair_gain=0.3,
                singleton_left_gain=0.05,
                singleton_right_gain=0.05,
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _per_token_row(
                "baseline",
                "fixed_best_support_swap",
                position_bin,
                token_class,
                norm_bin,
                support_count,
                support_count_difference=-1,
                fixed_support_loss_difference=0.02,
                pair_gain=0.11,
                singleton_left_gain=0.05,
                singleton_right_gain=0.05,
                include_synergy=include_synergy,
            )
        )
    _write_csv(audit_dir / "per_token_pair_interventions.csv", fields, rows)


def _matched_row(
    position_bin: str,
    token_class: str,
    residual_norm_bin: str,
    residual_gain_bin: str,
    support_count_bin: str,
) -> dict[str, str]:
    return {
        "position_bin": position_bin,
        "token_class": token_class,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": residual_gain_bin,
        "support_count_bin": support_count_bin,
    }


def _per_token_row(
    variant: str,
    intervention: str,
    position_bin: str,
    token_class: str,
    residual_norm_bin: str,
    router_support_count: int,
    *,
    support_count_difference: int,
    fixed_support_loss_difference: float,
    pair_gain: float,
    singleton_left_gain: float,
    singleton_right_gain: float,
    include_synergy: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "intervention": intervention,
        "position_bin": position_bin,
        "token_class": token_class,
        "router_support_count": router_support_count,
        "anchor_support": "0,1",
        "control_support": "2,3",
        "support_count_difference": support_count_difference,
        "fixed_support_loss_difference": fixed_support_loss_difference,
        "pair_gain": pair_gain,
        "singleton_left_gain": singleton_left_gain,
        "singleton_right_gain": singleton_right_gain,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": "high",
    }
    if include_synergy:
        row["pair_synergy"] = pair_gain - singleton_left_gain - singleton_right_gain
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
