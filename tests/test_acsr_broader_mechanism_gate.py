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
            self.assertFalse(summary["gates"]["sequence_heldout_available"])
            self.assertFalse(summary["gates"]["dual_student_cross_forcing_available"])
            self.assertIn(
                "acsr_beats_nulls_on_available_packets", summary["aggregate_metrics"]
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            variant_metrics = (out_dir / "variant_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("acsr_mlp_predicted_future", variant_metrics)
            self.assertIn("margin_gated_acsr_proxy", variant_metrics)
            cross_forcing = (out_dir / "same_student_cross_forcing.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("dual_student_cross_forcing", cross_forcing)
            perturbation = (out_dir / "perturbation_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("leaky_future_positive", perturbation)
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
        path / "feature_perturbation.csv",
        [
            {
                "check": "future_positions_do_not_change_prefix_predictions_or_support",
                "perturb_start": 4,
                "checked_prefix_positions": 4,
                "max_predicted_feature_delta": 0.0,
                "max_router_score_delta": 0.0,
                "support_unchanged": True,
                "passed": True,
            }
        ],
    )
    _write_csv(
        path / "retention_churn_metrics.csv",
        [
            {
                "phase": "second_context_transfer",
                "variant": "acsr_mlp_predicted_future",
                "anchor_ce_drift": 0.01,
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
