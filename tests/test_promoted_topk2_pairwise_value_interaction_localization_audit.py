from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_pairwise_value_interaction_localization_audit import (
    INSUFFICIENT_EVIDENCE,
    PAIRWISE_VALUE_INTERACTION_LOCALIZED,
    run_promoted_topk2_pairwise_value_interaction_localization_audit,
)


class PromotedTopk2PairwiseValueInteractionLocalizationAuditTest(unittest.TestCase):
    def test_localizes_hub_pair_family_from_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_promoted_topk2_pairwise_value_interaction_localization_audit(
                out_dir=root / "report",
                fingerprint_dir=paths["fingerprint"],
                update_decomposition_audit_path=paths["decomposition"],
                finite_update_report_path=paths["finite"],
                closeout_report_path=paths["closeout"],
                strategy_review_path=paths["review"],
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], PAIRWISE_VALUE_INTERACTION_LOCALIZED)
            self.assertEqual(summary["localization_status"], "hub_localized")
            self.assertEqual(summary["metrics"]["dominant_column"], 2)
            self.assertGreaterEqual(
                summary["metrics"]["dominant_column_abs_synergy_share"],
                0.60,
            )
            self.assertGreaterEqual(
                summary["metrics"]["top3_pair_abs_synergy_share"],
                0.35,
            )
            self.assertEqual(
                summary["claim_statuses"]["topk2_causal_cooperation"],
                "not_supported",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "localization_rows.csv").is_file())
            self.assertTrue(
                (root / "report" / "column_localization_rows.csv").is_file()
            )
            self.assertTrue((root / "report" / "stratum_rows.csv").is_file())
            saved = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(saved["decision"], summary["decision"])

    def test_fails_closed_when_closeout_does_not_select_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            _write_json(
                paths["closeout"],
                {
                    "status": "pass",
                    "decision": "promoted_topk2_mitigation_closeout_no_promotion",
                    "selected_next_action": "different_action",
                },
            )

            summary = run_promoted_topk2_pairwise_value_interaction_localization_audit(
                out_dir=root / "report",
                fingerprint_dir=paths["fingerprint"],
                update_decomposition_audit_path=paths["decomposition"],
                finite_update_report_path=paths["finite"],
                closeout_report_path=paths["closeout"],
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("post_value_router_mitigation_closeout", "selected_next_action"),
                fields,
            )


def _write_sources(root: Path) -> dict[str, Path]:
    fingerprint = root / "fingerprint"
    fingerprint.mkdir()
    _write_csv(
        fingerprint / "per_token_pair_interventions.csv",
        [
            "variant",
            "intervention",
            "support",
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "pair_synergy",
            "pair_gain",
            "singleton_left_gain",
            "singleton_right_gain",
            "fixed_support_loss_delta",
            "fixed_support_logit_mse",
            "fixed_support_residual_stream_l2_delta",
            "residual_norm",
        ],
        [
            _token_row("2,9", "even", 0.50),
            _token_row("2,9", "odd", 0.40),
            _token_row("2,15", "even", 0.35),
            _token_row("2,15", "odd", 0.30),
            _token_row("2,6", "even", 0.25),
            _token_row("2,6", "odd", 0.20),
            _token_row("1,3", "even", 0.05),
            _token_row("1,3", "odd", 0.04),
        ],
    )
    _write_csv(
        fingerprint / "pair_interventions.csv",
        [
            "variant",
            "intervention",
            "support",
            "position_bin",
            "token_class",
            "router_support_count",
            "pair_synergy",
            "pair_value_cosine",
        ],
        [
            _pair_row("2,9", 18, 0.45, 0.10),
            _pair_row("2,15", 12, 0.33, -0.10),
            _pair_row("2,6", 16, 0.23, 0.03),
            _pair_row("1,3", 2, 0.045, 0.01),
        ],
    )
    _write_csv(
        fingerprint / "column_fingerprints.csv",
        [
            "variant",
            "column",
            "router_support_count",
            "router_support_fraction",
            "column_value_norm",
            "force_loss_delta",
            "ablate_loss_delta",
        ],
        [
            _column_row(2, 144, 0.28, 4.1, 1.2, 0.06),
            _column_row(9, 18, 0.03, 4.0, 1.1, 0.04),
            _column_row(15, 12, 0.02, 4.3, 1.0, 0.03),
            _column_row(6, 16, 0.03, 4.2, 1.0, 0.03),
            _column_row(1, 8, 0.01, 4.2, 1.0, 0.03),
            _column_row(3, 4, 0.01, 4.2, 1.0, 0.03),
        ],
    )
    _write_csv(
        fingerprint / "support_frequency_candidate_controls.csv",
        [
            "anchor_support",
            "candidate_support",
            "included_in_primary_percentile_denominator",
            "candidate_pair_synergy",
        ],
        [
            _control_row("2,9", "0,1", 0.05),
            _control_row("2,9", "0,3", 0.08),
            _control_row("2,9", "0,4", 0.10),
            _control_row("2,15", "1,4", 0.07),
            _control_row("2,15", "1,5", 0.09),
            _control_row("2,6", "3,4", 0.11),
        ],
    )
    decomposition = root / "decomposition.json"
    finite = root / "finite.json"
    closeout = root / "closeout.json"
    review = root / "latest-review.md"
    _write_json(
        decomposition,
        {
            "status": "pass",
            "decision": "value_update_dominated_order_sensitivity",
            "metrics": {
                "value_only_fraction_of_full": 1.25,
                "router_only_fraction_of_full": 0.22,
            },
        },
    )
    _write_json(
        finite,
        {
            "status": "pass",
            "decision": "finite_update_order_sensitivity_ce_bounded_but_residual_material",
            "metrics": {
                "topk2_mean_commutator_anchor_support_churn": 0.91,
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_mean_commutator_anchor_residual_stream_l2": 5.0,
            },
        },
    )
    _write_json(
        closeout,
        {
            "status": "pass",
            "decision": "promoted_topk2_mitigation_closeout_no_promotion",
            "selected_next_action": "pairwise_value_interaction_localization_audit",
        },
    )
    review.write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: Run the planned router-policy mitigation probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "fingerprint": fingerprint,
        "decomposition": decomposition,
        "finite": finite,
        "closeout": closeout,
        "review": review,
    }


def _token_row(support: str, position_bin: str, synergy: float) -> dict[str, object]:
    return {
        "variant": "baseline",
        "intervention": "fixed_dominant_router_support",
        "support": support,
        "position_bin": position_bin,
        "token_class": "rare_target",
        "residual_norm_bin": "mid",
        "residual_gain_bin": "high",
        "pair_synergy": synergy,
        "pair_gain": 0.1,
        "singleton_left_gain": -0.05,
        "singleton_right_gain": -0.05,
        "fixed_support_loss_delta": 1.0,
        "fixed_support_logit_mse": 0.2,
        "fixed_support_residual_stream_l2_delta": 5.0,
        "residual_norm": 4.0,
    }


def _pair_row(
    support: str,
    router_count: int,
    synergy: float,
    cosine: float,
) -> dict[str, object]:
    return {
        "variant": "baseline",
        "intervention": "fixed_dominant_router_support",
        "support": support,
        "position_bin": "all",
        "token_class": "all",
        "router_support_count": router_count,
        "pair_synergy": synergy,
        "pair_value_cosine": cosine,
    }


def _column_row(
    column: int,
    router_count: int,
    router_fraction: float,
    value_norm: float,
    force_delta: float,
    ablate_delta: float,
) -> dict[str, object]:
    return {
        "variant": "baseline",
        "column": column,
        "router_support_count": router_count,
        "router_support_fraction": router_fraction,
        "column_value_norm": value_norm,
        "force_loss_delta": force_delta,
        "ablate_loss_delta": ablate_delta,
    }


def _control_row(
    anchor_support: str,
    candidate_support: str,
    candidate_pair_synergy: float,
) -> dict[str, object]:
    return {
        "anchor_support": anchor_support,
        "candidate_support": candidate_support,
        "included_in_primary_percentile_denominator": "True",
        "candidate_pair_synergy": candidate_pair_synergy,
    }


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
