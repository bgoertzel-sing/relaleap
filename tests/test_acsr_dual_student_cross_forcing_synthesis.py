from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_dual_student_cross_forcing_synthesis import (
    REQUIRED_ARTIFACTS,
    run_acsr_dual_student_cross_forcing_synthesis,
)


class ACSRDualStudentCrossForcingSynthesisTest(unittest.TestCase):
    def test_records_partner_transfer_against_null_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            review = root / "latest-review.md"
            _write_source(seed1, "seed1", acsr_partner_ce=2.80, direct_partner_ce=2.70)
            _write_source(seed2, "seed2", acsr_partner_ce=2.82, direct_partner_ce=2.72)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Interpret dual-student cross-forcing.",
                        "verdict: GO",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_dual_student_cross_forcing_synthesis(
                source_dirs=(seed1, seed2),
                strategy_review=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "acsr_dual_student_cross_forcing_synthesis_recorded",
            )
            self.assertEqual(
                summary["claim_status"],
                "cross_value_support_transfer_supported_not_promoted",
            )
            self.assertTrue(
                summary["aggregate_metrics"]["all_partner_beats_required_nulls"]
            )
            self.assertLess(
                summary["aggregate_metrics"][
                    "mean_partner_delta_vs_token_position_null"
                ],
                0.0,
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

            with (root / "report" / "value_student_support_synthesis.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("partner_oracle_headroom_recovered_fraction", rows[0])
            self.assertIn("partner_residual_update_l2_mean", rows[0])
            self.assertIn(
                "partner_per_token_delta_vs_token_position_null_improved_fraction",
                rows[0],
            )
            self.assertTrue(
                all(row["partner_beats_random_frequency_null"] == "True" for row in rows)
            )
            with (root / "report" / "stratified_transfer_synthesis.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                stratum_rows = list(csv.DictReader(handle))
            self.assertTrue(stratum_rows)
            self.assertIn("partner_delta_vs_token_position_null", stratum_rows[0])
            self.assertLess(
                summary["aggregate_metrics"][
                    "mean_high_regret_partner_delta_vs_token_position_null"
                ],
                0.0,
            )

    def test_fails_closed_when_required_null_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "seed1"
            _write_source(source, "seed1", acsr_partner_ce=2.80, direct_partner_ce=2.70)
            with (source / "dual_student_cross_forcing.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows = [
                row
                for row in rows
                if row["support_source"] != "random_frequency_matched_null"
            ]
            _write_csv(source / "dual_student_cross_forcing.csv", rows)

            summary = run_acsr_dual_student_cross_forcing_synthesis(
                source_dirs=(source,),
                strategy_review=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_dual_student_cross_forcing_synthesis_failed_closed",
            )
            self.assertTrue(
                any(
                    failure["gate"] == "required_value_student_rows_present"
                    for failure in summary["failures"]
                )
            )


def _write_source(
    path: Path,
    seed: str,
    *,
    acsr_partner_ce: float,
    direct_partner_ce: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_path": f"configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_{seed}.yaml",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    rows.extend(
        _value_rows(
            "acsr_student",
            own_variant="acsr_mlp_predicted_future",
            partner_variant="parameter_matched_causal_mlp_control",
            own_ce=2.81,
            partner_ce=acsr_partner_ce,
            token_ce=3.00,
            shuffled_ce=3.30,
            random_ce=3.90,
            oracle_ce=2.75,
            teacher_ce=2.76,
        )
    )
    rows.extend(
        _stratum_rows(
            "acsr_student",
            partner_variant="parameter_matched_causal_mlp_control",
            partner_ce=acsr_partner_ce,
            own_ce=2.81,
            token_ce=3.00,
            oracle_ce=2.75,
        )
    )
    rows.extend(
        _value_rows(
            "parameter_matched_direct_causal_mlp_student",
            own_variant="parameter_matched_causal_mlp_control",
            partner_variant="acsr_mlp_predicted_future",
            own_ce=2.73,
            partner_ce=direct_partner_ce,
            token_ce=2.95,
            shuffled_ce=3.20,
            random_ce=3.80,
            oracle_ce=2.68,
            teacher_ce=2.69,
        )
    )
    rows.extend(
        _stratum_rows(
            "parameter_matched_direct_causal_mlp_student",
            partner_variant="acsr_mlp_predicted_future",
            partner_ce=direct_partner_ce,
            own_ce=2.73,
            token_ce=2.95,
            oracle_ce=2.68,
        )
    )
    _write_csv(path / "dual_student_cross_forcing.csv", rows)


def _value_rows(
    value_student: str,
    *,
    own_variant: str,
    partner_variant: str,
    own_ce: float,
    partner_ce: float,
    token_ce: float,
    shuffled_ce: float,
    random_ce: float,
    oracle_ce: float,
    teacher_ce: float,
) -> list[dict[str, object]]:
    return [
        _row(value_student, "own", own_variant, own_ce, own_ce, token_ce, oracle_ce),
        _row(
            value_student,
            "partner",
            partner_variant,
            partner_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _row(
            value_student,
            "token_position_null",
            "token_position_only_predicted_features",
            token_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _row(
            value_student,
            "position_stratified_shuffled_null",
            "shuffled_predicted_features",
            shuffled_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _row(
            value_student,
            "random_frequency_matched_null",
            "random_frequency_matched_topk",
            random_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _row(
            value_student,
            "oracle_diagnostic",
            "oracle_best_per_token_topk",
            oracle_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _row(
            value_student,
            "full_context_teacher_diagnostic",
            "full_context_contextual_topk2_teacher",
            teacher_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
    ]


def _row(
    value_student: str,
    support_source: str,
    support_variant: str,
    ce_loss: float,
    own_ce: float,
    token_ce: float,
    oracle_ce: float,
) -> dict[str, object]:
    return {
        "forcing_type": "dual_student_cross_forcing",
        "status": "available",
        "eval_split": "source_batch_all_but_last_token",
        "value_student": value_student,
        "support_source": support_source,
        "support_variant": support_variant,
        "top_k": 2,
        "analysis_scope": "all_tokens",
        "stratum_type": "all_tokens",
        "stratum_value": "all",
        "stratum_token_count": 64,
        "ce_loss": ce_loss,
        "oracle_loss": oracle_ce,
        "oracle_regret": ce_loss - oracle_ce,
        "loss_delta_vs_own_support": ce_loss - own_ce,
        "loss_delta_vs_token_position_null": ce_loss - token_ce,
        "support_jaccard_with_own": 1.0 if support_source == "own" else 0.25,
        "topk_margin_bin": "high_margin",
        "residual_update_l2_mean": _residual_norm(support_source),
        "residual_update_l2_delta_vs_own": _residual_norm(support_source)
        - _residual_norm("own"),
        "residual_update_l2_delta_vs_token_position_null": _residual_norm(support_source)
        - _residual_norm("token_position_null"),
        "loss_delta_vs_own_per_residual_l2": (ce_loss - own_ce)
        / (_residual_norm(support_source) - _residual_norm("own") or 1.0),
        "loss_delta_vs_token_position_null_per_residual_l2": (ce_loss - token_ce)
        / (
            _residual_norm(support_source)
            - _residual_norm("token_position_null")
            or 1.0
        ),
        "per_token_delta_vs_own_mean": ce_loss - own_ce,
        "per_token_delta_vs_own_median": ce_loss - own_ce,
        "per_token_delta_vs_own_improved_fraction": 1.0 if ce_loss < own_ce else 0.0,
        "per_token_delta_vs_token_position_null_mean": ce_loss - token_ce,
        "per_token_delta_vs_token_position_null_median": ce_loss - token_ce,
        "per_token_delta_vs_token_position_null_improved_fraction": 1.0
        if ce_loss < token_ce
        else 0.0,
    }


def _residual_norm(support_source: str) -> float:
    return {
        "own": 0.50,
        "partner": 0.62,
        "token_position_null": 0.40,
        "position_stratified_shuffled_null": 0.70,
        "random_frequency_matched_null": 0.80,
        "oracle_diagnostic": 0.60,
        "full_context_teacher_diagnostic": 0.61,
    }[support_source]


def _stratum_rows(
    value_student: str,
    *,
    partner_variant: str,
    partner_ce: float,
    own_ce: float,
    token_ce: float,
    oracle_ce: float,
) -> list[dict[str, object]]:
    return [
        _stratum_row(
            value_student,
            partner_variant,
            "oracle_regret",
            "top_quartile_token_position_null_regret",
            partner_ce,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _stratum_row(
            value_student,
            partner_variant,
            "support_disagreement",
            "partner_vs_own",
            partner_ce + 0.01,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _stratum_row(
            value_student,
            partner_variant,
            "support_disagreement",
            "partner_vs_token_position_null",
            partner_ce + 0.02,
            own_ce,
            token_ce,
            oracle_ce,
        ),
        _stratum_row(
            value_student,
            partner_variant,
            "partner_support_margin_bin",
            "high_margin",
            partner_ce + 0.03,
            own_ce,
            token_ce,
            oracle_ce,
        ),
    ]


def _stratum_row(
    value_student: str,
    partner_variant: str,
    stratum_type: str,
    stratum_value: str,
    ce_loss: float,
    own_ce: float,
    token_ce: float,
    oracle_ce: float,
) -> dict[str, object]:
    row = _row(
        value_student,
        "partner",
        partner_variant,
        ce_loss,
        own_ce,
        token_ce,
        oracle_ce,
    )
    row.update(
        {
            "forcing_type": "dual_student_cross_forcing_stratum",
            "analysis_scope": "stratified_tokens",
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "stratum_token_count": 16,
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
