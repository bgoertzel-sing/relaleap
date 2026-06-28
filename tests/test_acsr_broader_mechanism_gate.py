from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_broader_mechanism_gate import (
    REQUIRED_ARTIFACTS,
    run_acsr_broader_mechanism_gate,
)


class ACSRBroaderMechanismGateTest(unittest.TestCase):
    def test_gate_writes_required_artifacts_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source_packet"
            out_dir = root / "gate"
            _write_source_packet(source)

            summary = run_acsr_broader_mechanism_gate(
                source_dirs=[source],
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"], "acsr_broader_mechanism_gate_failed_closed"
            )
            self.assertEqual(summary["loaded_packet_count"], 1)
            self.assertTrue(summary["gates"]["source_artifacts_present"])
            self.assertTrue(summary["gates"]["sequence_heldout_available"])
            self.assertTrue(summary["gates"]["dual_student_cross_forcing_available"])
            self.assertTrue(summary["gates"]["support_agreement_available"])
            self.assertTrue(summary["gates"]["retention_churn_available"])
            self.assertTrue(summary["gates"]["intervention_fingerprint_available"])
            self.assertTrue(summary["gates"]["leaky_positive_control_available"])
            self.assertTrue(
                summary["gates"]["parameter_matched_causal_control_available"]
            )
            self.assertTrue(
                summary["gates"][
                    "acsr_no_worse_retention_churn_than_contextual"
                ]
            )
            self.assertTrue(
                summary["gates"][
                    "acsr_no_worse_intervention_residual_l2_than_parameter_matched"
                ]
            )
            self.assertIn(
                "acsr_beats_nulls_on_available_packets", summary["aggregate_metrics"]
            )
            self.assertFalse(
                summary["gates"]["acsr_beats_parameter_matched_causal_control"]
            )
            self.assertTrue(
                any(
                    failure["reason"]
                    == "acsr_not_better_than_parameter_matched_causal_control"
                    for failure in summary["failures"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            variant_metrics = (out_dir / "variant_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("acsr_mlp_predicted_future", variant_metrics)
            self.assertIn("parameter_matched_causal_mlp_control", variant_metrics)
            self.assertIn("margin_gated_acsr_proxy", variant_metrics)
            cross_forcing = (out_dir / "same_student_cross_forcing.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("dual_student_cross_forcing", cross_forcing)
            support_agreement = (out_dir / "support_agreement.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "acsr_mlp_predicted_future_support_vs_parameter_matched_causal_mlp_control",
                support_agreement,
            )
            perturbation = (out_dir / "perturbation_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("leaky_future_positive", perturbation)
            intervention = (out_dir / "intervention_fingerprint.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("dual_student_cross_forcing", intervention)
            retention = (out_dir / "retention_churn_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("anchor_support_churn_after_transfer", retention)
            margin = (out_dir / "margin_fragility.csv").read_text(encoding="utf-8")
            self.assertIn("feature_noise_flip_rate", margin)
            parameter_counts = (out_dir / "parameter_counts.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("parameter_matched_causal_mlp_control", parameter_counts)
            notes = (out_dir / "notes.md").read_text(encoding="utf-8")
            self.assertIn("_score_from_features", notes)

    def test_missing_source_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "empty_source"
            source.mkdir()
            out_dir = root / "gate"

            summary = run_acsr_broader_mechanism_gate(
                source_dirs=[source],
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertFalse(summary["gates"]["source_artifacts_present"])
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertTrue(
                any(failure["gate"] == "source_artifact" for failure in summary["failures"])
            )


def _write_source_packet(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_path": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
                "hidden_dim": 16,
                "num_columns": 6,
                "top_k": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "router_metrics.csv",
        [
            _router_row("causal_feature_safe_contextual_topk2", 3.0, 0.2, 1.0),
            _router_row("acsr_mlp_predicted_future", 2.5, 0.1, 1.5),
            _router_row("shuffled_predicted_features", 3.5, 0.6, 1.1),
            _router_row("token_position_only_predicted_features", 3.1, 0.3, 1.2),
            _router_row("mean_predicted_features", 3.2, 0.4, 1.0),
            _router_row("zero_predicted_features", 3.0, 0.2, 1.0),
            _router_row("parameter_matched_causal_mlp_control", 2.45, 0.05, 1.3),
            _router_row("random_fixed_topk2", 4.0, 1.0, 0.0),
        ],
    )
    _write_csv(
        path / "same_student_metrics.csv",
        [
            {
                "comparison": "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features",
                "acsr_forced_ce_loss": 2.5,
                "control_forced_ce_loss": 3.5,
                "acsr_minus_control_ce_loss": -1.0,
            }
        ],
    )
    _write_csv(
        path / "dual_student_cross_forcing.csv",
        [
            {
                "forcing_type": "dual_student_cross_forcing",
                "status": "available",
                "eval_split": "source_batch_all_but_last_token",
                "value_student": "acsr_student",
                "support_source": "own",
                "support_variant": "acsr_mlp_predicted_future",
                "analysis_scope": "all_tokens",
                "stratum_type": "all_tokens",
                "stratum_value": "all",
                "residual_update_l2_mean": 4.0,
                "per_token_delta_vs_own_improved_fraction": 0.0,
            },
            {
                "forcing_type": "dual_student_cross_forcing",
                "status": "available",
                "eval_split": "source_batch_all_but_last_token",
                "value_student": "acsr_student",
                "support_source": "partner",
                "support_variant": "parameter_matched_causal_mlp_control",
                "analysis_scope": "all_tokens",
                "stratum_type": "all_tokens",
                "stratum_value": "all",
                "residual_update_l2_mean": 4.1,
                "per_token_delta_vs_own_improved_fraction": 0.1,
            },
        ],
    )
    _write_csv(
        path / "feature_perturbation.csv",
        [
            {
                "control_type": "future_perturbation_negative",
                "check": "future_positions_do_not_change_prefix_predictions_or_support",
                "perturb_start": 4,
                "checked_prefix_positions": 4,
                "max_predicted_feature_delta": 0.0,
                "max_router_score_delta": 0.0,
                "support_unchanged": True,
                "passed": True,
            },
            {
                "control_type": "leaky_future_positive",
                "check": "full_context_features_detect_future_perturbation",
                "perturb_start": 4,
                "checked_prefix_positions": 4,
                "max_predicted_feature_delta": 0.5,
                "max_router_score_delta": 0.1,
                "support_unchanged": False,
                "passed": True,
            }
        ],
    )
    _write_csv(
        path / "support_agreement.csv",
        [
            {
                "comparison": "acsr_mlp_predicted_future_support_vs_parameter_matched_causal_mlp_control",
                "status": "available",
                "top_k": 2,
                "slot_match_fraction": 0.75,
                "set_match_fraction": 0.8,
                "changed_support_fraction": 0.2,
            }
        ],
    )
    _write_csv(
        path / "sequence_heldout_metrics.csv",
        [
            {
                "split": "sequence_suffix_holdout",
                "variant": "acsr_mlp_predicted_future",
                "top_k": 2,
                "holdout_start": 4,
                "heldout_positions": 3,
                "ce_loss": 2.6,
                "oracle_loss": 2.4,
                "oracle_regret": 0.2,
            },
            {
                "split": "sequence_suffix_holdout",
                "variant": "parameter_matched_causal_mlp_control",
                "top_k": 2,
                "holdout_start": 4,
                "heldout_positions": 3,
                "ce_loss": 2.55,
                "oracle_loss": 2.4,
                "oracle_regret": 0.15,
            }
        ],
    )
    _write_csv(
        path / "margin_fragility.csv",
        [
            {
                "variant": "acsr_mlp_predicted_future",
                "top_k": 2,
                "noise_kind": "gaussian_feature_noise",
                "noise_scale_fraction": 0.01,
                "mean_topk_margin": 1.5,
                "p25_topk_margin": 0.7,
                "feature_noise_flip_rate": 0.1,
                "low_margin_feature_noise_flip_rate": 0.2,
            }
        ],
    )
    _write_csv(
        path / "parameter_counts.csv",
        [
            {
                "component": "residual_columns",
                "active_parameter_count": 64,
                "stored_parameter_count": 512,
                "basis": "unit",
                "status": "available",
            },
            {
                "component": "parameter_matched_causal_mlp_control",
                "active_parameter_count": 1024,
                "stored_parameter_count": 1024,
                "basis": "unit",
                "status": "available",
                "parameter_count_ratio_to_acsr_path": 1.0,
            },
        ],
    )
    _write_csv(
        path / "retention_churn_metrics.csv",
        [
            {
                "phase": "second_context_transfer",
                "variant": "causal_feature_safe_contextual_topk2",
                "anchor_support_churn_after_transfer": 0.2,
                "anchor_logit_mse_after_transfer": 0.02,
            },
            {
                "phase": "second_context_transfer",
                "variant": "acsr_mlp_predicted_future",
                "anchor_ce_drift": 0.01,
                "anchor_support_churn_after_transfer": 0.1,
                "anchor_logit_mse_after_transfer": 0.01,
            }
        ],
    )


def _router_row(
    variant: str,
    ce_loss: float,
    oracle_regret: float,
    mean_topk_margin: float,
) -> dict[str, object]:
    return {
        "variant": variant,
        "top_k": 2,
        "ce_loss": ce_loss,
        "used_columns": 5,
        "unique_support_sets": 7,
        "support_entropy": 1.5,
        "mean_topk_margin": mean_topk_margin,
        "oracle_regret": oracle_regret,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
