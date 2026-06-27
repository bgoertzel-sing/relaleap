from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_causal_router_capacity_audit import (
    REQUIRED_ARTIFACTS,
    run_acsr_causal_router_capacity_audit,
)


class ACSRCausalRouterCapacityAuditTest(unittest.TestCase):
    def test_fails_closed_when_parameter_matched_control_is_competitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            gate = root / "gate"
            review = root / "latest-review.md"
            out_dir = root / "out"
            _write_source_packet(source)
            _write_gate(gate)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Freeze ACSR-as-anticipation promotion and GPU repeats, finalize the parameter-matched causal-MLP fail-closed gate with exact deltas, then pivot locally to a capacity-matched causal support-router mechanism audit.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_causal_router_capacity_audit(
                source_dirs=(source,),
                gate_dir=gate,
                strategy_review=review,
                out_dir=out_dir,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["claim_status"],
                "acsr_as_anticipation_blocked_by_capacity_matched_causal_router",
            )
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "major")
            self.assertEqual(summary["direction_shift"].count("Ben should be notified"), 1)
            self.assertEqual(summary["aggregate_metrics"]["paired_delta_count"], 2)
            self.assertFalse(
                summary["aggregate_metrics"]["dual_student_cross_forcing_available"]
            )
            self.assertTrue(
                any(
                    failure["reason"]
                    == "acsr_not_strictly_better_than_parameter_matched_causal_mlp"
                    for failure in summary["failures"]
                )
            )
            self.assertTrue(
                any(failure["gate"] == "support_agreement" for failure in summary["failures"])
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            deltas = (out_dir / "paired_capacity_deltas.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("sequence_suffix_holdout", deltas)
            self.assertIn("parameter_matched_ce_loss", deltas)
            missing = (out_dir / "missing_mechanism_evidence.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("dual_student_cross_forcing", missing)
            self.assertIn("support_agreement", missing)

    def test_missing_packet_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_acsr_causal_router_capacity_audit(
                source_dirs=(root / "missing",),
                gate_dir=root / "missing_gate",
                strategy_review=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(failure["gate"] == "source_packet" for failure in summary["failures"])
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_source_packet(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_path": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "router_metrics.csv",
        [
            _router_row("acsr_mlp_predicted_future", 2.80, 0.10),
            _router_row("parameter_matched_causal_mlp_control", 2.79, 0.09),
        ],
    )
    _write_csv(
        path / "sequence_heldout_metrics.csv",
        [
            _sequence_row("acsr_mlp_predicted_future", 2.90, 0.20),
            _sequence_row("parameter_matched_causal_mlp_control", 2.90, 0.20),
        ],
    )
    _write_csv(
        path / "same_student_metrics.csv",
        [
            {
                "comparison": "acsr_mlp_predicted_future_support_vs_parameter_matched_causal_mlp_control",
                "acsr_forced_ce_loss": 2.80,
                "control_forced_ce_loss": 2.79,
                "acsr_minus_control_ce_loss": 0.01,
            }
        ],
    )
    _write_csv(
        path / "margin_fragility.csv",
        [
            {
                "variant": "acsr_mlp_predicted_future",
                "mean_topk_margin": 1.0,
                "feature_noise_flip_rate": 0.1,
            },
            {
                "variant": "parameter_matched_causal_mlp_control",
                "mean_topk_margin": 1.1,
                "feature_noise_flip_rate": 0.05,
            },
        ],
    )
    _write_csv(
        path / "parameter_counts.csv",
        [
            {
                "component": "parameter_matched_causal_mlp_control",
                "status": "available",
                "stored_parameter_count": 1024,
                "active_parameter_count": 1024,
                "parameter_count_ratio_to_acsr_path": 1.0,
                "basis": "unit",
            }
        ],
    )


def _write_gate(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"gates": {"dual_student_cross_forcing_available": False}}) + "\n",
        encoding="utf-8",
    )


def _router_row(variant: str, ce_loss: float, regret: float) -> dict[str, object]:
    return {
        "variant": variant,
        "top_k": 2,
        "ce_loss": ce_loss,
        "oracle_regret": regret,
    }


def _sequence_row(variant: str, ce_loss: float, regret: float) -> dict[str, object]:
    return {
        "split": "sequence_suffix_holdout",
        "variant": variant,
        "top_k": 2,
        "holdout_start": 4,
        "ce_loss": ce_loss,
        "oracle_regret": regret,
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
