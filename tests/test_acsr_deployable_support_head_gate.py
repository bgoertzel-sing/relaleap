from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_deployable_support_head_gate import (
    REQUIRED_ARTIFACTS,
    run_acsr_deployable_support_head_gate,
)


class ACSRDeployableSupportHeadGateTest(unittest.TestCase):
    def test_blocks_claim_when_shuffled_feature_null_and_headroom_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit = root / "audit"
            gate = root / "gate"
            review = root / "latest-review.md"
            _write_source_audit(audit)
            _write_prior_gate(gate, retired=True)
            _write_review(review)

            summary = run_acsr_deployable_support_head_gate(
                source_audit_dir=audit,
                acsr_gate_dir=gate,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "deployable_support_head_gate_blocks_claim_pending_nulls_or_headroom",
            )
            self.assertEqual(
                summary["claim_status"],
                "deployable_support_discovery_not_established_sparse_identity_retired",
            )
            self.assertIn("do not run RunPod", summary["selected_next_step"])
            self.assertIn("Ben should be notified", summary["direction_shift"])
            blocker_names = {row["criterion"] for row in summary["claim_blockers"]}
            self.assertIn("shuffled_feature_support_head_null_present", blocker_names)
            self.assertIn("oracle_support_headroom_positive", blocker_names)
            self.assertIn("learned_head_recovers_oracle_gap", blocker_names)
            self.assertFalse(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "support_head_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            learned = next(row for row in rows if row["component"] == "learned_contextual_support_head")
            self.assertEqual(learned["present"], "True")
            self.assertEqual(learned["holdout_intervention_minus_router_loss"], "-0.0004782676696777344")

    def test_fails_closed_when_prior_gate_does_not_retire_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit = root / "audit"
            gate = root / "gate"
            review = root / "latest-review.md"
            _write_source_audit(audit)
            _write_prior_gate(gate, retired=False)
            _write_review(review)

            summary = run_acsr_deployable_support_head_gate(
                source_audit_dir=audit,
                acsr_gate_dir=gate,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "deployable_support_head_gate_failed_closed")
            self.assertTrue(any(row["criterion"] == "identity_claim_retired" for row in summary["failures"]))


def _write_source_audit(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "audit": {
            "contextual_router_support_head": {
                "selector": "mlp_contextual_support_head_ce_minimizer",
                "training_objective": "expected_fixed_batch_support_ce",
                "train_split": "even flattened token positions",
                "holdout_split": "odd flattened token positions",
                "holdout": {
                    "router_loss": 2.874990463256836,
                    "oracle_loss": 2.8724637031555176,
                    "intervention_loss": 2.874512195587158,
                    "intervention_minus_router_loss": -0.0004782676696777344,
                    "intervention_oracle_regret": 0.002048492431640625,
                    "oracle_gap_recovery_fraction": 0.1892809964144178,
                },
                "all": {
                    "oracle_gap_recovery_fraction": 0.1868929750117869,
                },
            },
            "contextual_router_support_intervention": {
                "selector": "mlp_contextual_hidden_to_oracle_pair",
                "train_split": "train even positions",
                "holdout_split": "holdout odd positions",
                "holdout": {
                    "router_loss": 2.874990463256836,
                    "oracle_loss": 2.8724637031555176,
                    "intervention_loss": 2.8724637031555176,
                    "intervention_minus_router_loss": -0.0025267601013183594,
                    "intervention_oracle_regret": 0.0,
                    "oracle_gap_recovery_fraction": 1.0,
                },
                "all": {
                    "oracle_gap_recovery_fraction": 1.0,
                },
            },
            "contextual_router_support_sequence_head": {
                "selector": "mlp_contextual_support_head_ce_minimizer",
                "training_objective": "expected_fixed_batch_support_ce",
                "holdout": {
                    "intervention_minus_router_loss": 0.016779184341430664,
                    "oracle_gap_recovery_fraction": -6.655035460992908,
                },
                "all": {
                    "oracle_gap_recovery_fraction": -5.47996228194248,
                },
            },
        }
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "router_support_intervention.csv").write_text(
        "split,positions,router_loss,oracle_loss,intervention_loss\n"
        "holdout_odd_positions,126,2.874990463256836,2.8724637031555176,2.8724637031555176\n",
        encoding="utf-8",
    )


def _write_prior_gate(path: Path, *, retired: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "claim_status": (
            "deployable_support_discovery_not_established_sparse_identity_retired"
            if retired
            else "sparse_support_identity_not_retired"
        ),
        "aggregate_metrics": {
            "sparse_oracle_minus_sparse_default_heldout_ce_delta": -0.0023670196533203125,
        },
        "null_controls": [
            {
                "control": "token_position_support_null",
                "present": True,
                "heldout_delta_vs_base_ce": -0.03207588195800781,
                "gap_vs_reference_heldout_ce_delta": 0.2847728729248047,
            }
        ],
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: major",
                "notify_ben: true",
                "recommended_next_action: keep support discovery local before RunPod",
                "verdict: PIVOT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
