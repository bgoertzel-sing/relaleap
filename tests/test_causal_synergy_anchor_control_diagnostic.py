from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_synergy_anchor_control_diagnostic import (
    run_causal_synergy_anchor_control_diagnostic,
)


class CausalSynergyAnchorControlDiagnosticTest(unittest.TestCase):
    def test_anchor_control_diagnostic_separates_control_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_artifacts(audit_dir, deconfounded_dir)

            summary = run_causal_synergy_anchor_control_diagnostic(
                audit_dir,
                root / "anchor_diagnostic",
                deconfounded_dir=deconfounded_dir,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "selection_controls_pass_but_loss_matched_bound_fails",
            )
            evidence = summary["evidence"]
            self.assertEqual(evidence["anchor_count"], 2)
            self.assertEqual(evidence["paired_control_token_count"], 14)
            support_frequency = evidence["control_summaries"][
                "fixed_support_frequency_matched_control"
            ]
            self.assertTrue(support_frequency["supported"])
            self.assertEqual(support_frequency["control_role"], "selection_control")
            self.assertEqual(support_frequency["anchor_count"], 2)
            self.assertAlmostEqual(
                support_frequency["control_supports_per_anchor_mean"],
                2.0,
            )
            random_nonrouter = evidence["control_summaries"][
                "fixed_random_nonrouter_control"
            ]
            self.assertTrue(random_nonrouter["supported"])
            self.assertEqual(random_nonrouter["control_role"], "selection_control")
            singleton_gain = evidence["control_summaries"][
                "fixed_singleton_gain_matched_nonrouter_control"
            ]
            self.assertTrue(singleton_gain["supported"])
            self.assertEqual(singleton_gain["control_role"], "selection_control")
            residual_norm = evidence["control_summaries"][
                "fixed_residual_norm_matched_nonrouter_control"
            ]
            self.assertTrue(residual_norm["supported"])
            self.assertEqual(residual_norm["control_role"], "selection_control")
            best_swap = evidence["control_summaries"]["fixed_best_support_swap"]
            self.assertTrue(best_swap["supported"])
            self.assertEqual(best_swap["control_role"], "selection_control")
            loss_matched = evidence["control_summaries"][
                "fixed_loss_matched_nonrouter_control"
            ]
            self.assertFalse(loss_matched["supported"])
            self.assertEqual(
                loss_matched["control_role"],
                "outcome_proximal_loss_matched_secondary_bound",
            )
            self.assertTrue((root / "anchor_diagnostic" / "summary.json").is_file())
            self.assertTrue(
                (
                    root / "anchor_diagnostic" / "per_anchor_control_deltas.csv"
                ).is_file()
            )
            self.assertTrue((root / "anchor_diagnostic" / "notes.md").is_file())
            with (root / "anchor_diagnostic" / "per_anchor_control_deltas.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 14)
            self.assertIn("observed_minus_control_pair_synergy", rows[0])
            self.assertIn("control_match_status", rows[0])

    def test_fails_without_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_artifacts(audit_dir, deconfounded_dir, include_synergy=False)

            summary = run_causal_synergy_anchor_control_diagnostic(
                audit_dir,
                root / "anchor_diagnostic",
                deconfounded_dir=deconfounded_dir,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "insufficient_evidence")
            self.assertIn(
                "per_token_anchor_control_fields",
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
            _matched_row("even", "common_target", "low", "high", "low"),
            _matched_row("odd", "rare_target", "mid", "high", "mid"),
        ],
    )
    (deconfounded_dir / "summary.json").write_text(
        json.dumps({"status": "pass"}) + "\n",
        encoding="utf-8",
    )
    fields = [
        "variant",
        "intervention",
        "support",
        "anchor_support",
        "control_support",
        "anchor_router_support_count",
        "control_router_support_count",
        "support_count_difference",
        "fixed_support_loss_difference",
        "control_match_rank",
        "control_match_status",
        "token_index",
        "batch_index",
        "position_index",
        "position_bin",
        "token_class",
        "residual_norm_bin",
        "residual_gain_bin",
        "router_support_count",
        "pair_gain",
    ]
    if include_synergy:
        fields.append("pair_synergy")

    rows: list[dict[str, object]] = []
    anchors = [
        ("0,1", "0", "even", "common_target", "low", 4, 0.20),
        ("2,3", "1", "odd", "rare_target", "mid", 5, 0.16),
    ]
    for anchor, token_index, position_bin, token_class, norm_bin, support_count, synergy in anchors:
        rows.append(
            _row(
                "fixed_dominant_router_support",
                anchor,
                "",
                "",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy,
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_support_frequency_matched_control",
                "4,5",
                anchor,
                "4,5",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.05,
                support_count_difference=0,
                fixed_support_loss_difference=0.01,
                control_match_rank=0,
                control_match_status="exact_support_count_candidate_available",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_support_frequency_matched_control",
                "4,6",
                anchor,
                "4,6",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.03,
                support_count_difference=1,
                fixed_support_loss_difference=0.02,
                control_match_rank=1,
                control_match_status="near_support_count_candidate_available",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_loss_matched_nonrouter_control",
                "5,6",
                anchor,
                "5,6",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy + 0.02,
                support_count_difference=-2,
                fixed_support_loss_difference=0.0,
                control_match_rank=0,
                control_match_status="loss_matched_nonrouter_candidate",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_random_nonrouter_control",
                "6,7",
                anchor,
                "6,7",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.04,
                support_count_difference=-support_count,
                fixed_support_loss_difference=0.03,
                control_match_rank=1,
                control_match_status="random_nonrouter_candidate",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_singleton_gain_matched_nonrouter_control",
                "3,7",
                anchor,
                "3,7",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.035,
                support_count_difference=-1,
                fixed_support_loss_difference=0.015,
                control_match_rank=1,
                control_match_status="singleton_gain_matched_nonrouter_candidate",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_residual_norm_matched_nonrouter_control",
                "1,6",
                anchor,
                "1,6",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.025,
                support_count_difference=-1,
                fixed_support_loss_difference=0.025,
                control_match_rank=1,
                control_match_status="residual_norm_matched_nonrouter_candidate",
                include_synergy=include_synergy,
            )
        )
        rows.append(
            _row(
                "fixed_best_support_swap",
                "0,2",
                anchor,
                "0,2",
                token_index,
                position_bin,
                token_class,
                norm_bin,
                support_count,
                synergy - 0.01,
                support_count_difference=-1,
                fixed_support_loss_difference=-0.02,
                control_match_rank=1,
                control_match_status="best_support_swap_candidate",
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


def _row(
    intervention: str,
    support: str,
    anchor_support: str,
    control_support: str,
    token_index: str,
    position_bin: str,
    token_class: str,
    residual_norm_bin: str,
    router_support_count: int,
    pair_synergy: float,
    *,
    support_count_difference: int | str = "",
    fixed_support_loss_difference: float | str = "",
    control_match_rank: int | str = "",
    control_match_status: str = "",
    include_synergy: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": "baseline",
        "intervention": intervention,
        "support": support,
        "anchor_support": anchor_support,
        "control_support": control_support,
        "anchor_router_support_count": router_support_count,
        "control_router_support_count": router_support_count,
        "support_count_difference": support_count_difference,
        "fixed_support_loss_difference": fixed_support_loss_difference,
        "control_match_rank": control_match_rank,
        "control_match_status": control_match_status,
        "token_index": token_index,
        "batch_index": "0",
        "position_index": token_index,
        "position_bin": position_bin,
        "token_class": token_class,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": "high",
        "router_support_count": router_support_count,
        "pair_gain": 0.3,
    }
    if include_synergy:
        row["pair_synergy"] = pair_synergy
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
