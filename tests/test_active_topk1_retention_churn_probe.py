from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from relaleap.experiments.active_topk1_retention_churn_probe import (
    ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_retention_churn_probe,
)


class ActiveTopk1RetentionChurnProbeTest(unittest.TestCase):
    def test_probe_establishes_low_churn_active_topk1_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            _write_config(config_path)
            separability_dir = root / "separability"
            _write_separability_packet(separability_dir)

            with mock.patch(
                "relaleap.experiments.active_topk1_retention_churn_probe"
                ".run_retention_churn_microtest",
                side_effect=_write_microtest_packet,
            ):
                summary = run_active_topk1_retention_churn_probe(
                    config_path=config_path,
                    separability_dir=separability_dir,
                    out_dir=root / "probe",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                ACTIVE_TOPK1_RETENTION_CHURN_PROBE_ESTABLISHED,
            )
            metrics = summary["evidence"]["metrics"]
            signals = summary["evidence"]["signals"]
            self.assertTrue(signals["required_variants_present"])
            self.assertTrue(signals["topk1_support_churn_lower_than_topk2"])
            self.assertTrue(signals["topk1_logit_churn_not_higher_than_topk2"])
            self.assertTrue(signals["topk1_transfer_improvement_beats_dense"])
            self.assertEqual(metrics["random_fixed_topk2_transfer_ce_improvement"], 0.25)
            self.assertLess(
                metrics["topk1_anchor_support_churn_after_transfer"],
                metrics["topk2_anchor_support_churn_after_transfer"],
            )
            self.assertEqual(
                metrics["source_separability_decision"],
                "active_topk1_causal_separability_audit_established",
            )
            self.assertTrue(signals["finite_update_commutator_present"])
            self.assertEqual(metrics["topk1_commutator_anchor_logit_mse"], 0.04)
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "notes.md").is_file())

    def test_probe_fails_without_active_separability_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            _write_config(config_path)
            separability_dir = root / "separability"
            _write_separability_packet(separability_dir, decision="wrong_decision")

            summary = run_active_topk1_retention_churn_probe(
                config_path=config_path,
                separability_dir=separability_dir,
                out_dir=root / "probe",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertEqual(
                summary["evidence"]["failures"][0]["field"],
                "separability.decision",
            )


def _write_config(path: Path) -> None:
    path.write_text(
        """
run:
  experiment_id: test_active_topk1_retention_churn_probe
  seed: 1
  max_steps: 2

data:
  dataset: tiny_shakespeare_char
  seq_len: 16

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 32
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_separability_packet(
    out_dir: Path,
    *,
    status: str = "pass",
    decision: str = "active_topk1_causal_separability_audit_established",
) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": status,
        "decision": decision,
        "evidence": {
            "metrics": {
                "topk1_singleton_gain_mean": -0.04,
                "context_level_topk1_singleton_gain_mean": -0.15,
            }
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _write_microtest_packet(config_path: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "audit": {
            "variants": [
                {
                    "variant": "promoted_contextual_topk2",
                    "anchor_support_churn_after_transfer": 0.9,
                    "anchor_logit_mse_drift": 0.16,
                    "anchor_residual_stream_l2_drift": 4.6,
                    "anchor_ce_drift": -0.89,
                    "transfer_ce_improvement": 0.90,
                    "commutator_anchor_logit_mse": 0.08,
                    "commutator_transfer_logit_mse": 0.07,
                    "commutator_anchor_residual_stream_l2": 0.8,
                    "commutator_transfer_residual_stream_l2": 0.7,
                },
                {
                    "variant": "random_fixed_topk2",
                    "anchor_support_churn_after_transfer": 0.0,
                    "anchor_logit_mse_drift": 0.20,
                    "anchor_residual_stream_l2_drift": 4.8,
                    "anchor_ce_drift": 0.04,
                    "transfer_ce_improvement": 0.25,
                    "commutator_anchor_logit_mse": 0.30,
                    "commutator_transfer_logit_mse": 0.29,
                    "commutator_anchor_residual_stream_l2": 3.0,
                    "commutator_transfer_residual_stream_l2": 2.9,
                },
                {
                    "variant": "rank_matched_contextual_topk1",
                    "anchor_support_churn_after_transfer": 0.01,
                    "anchor_logit_mse_drift": 0.14,
                    "anchor_residual_stream_l2_drift": 4.5,
                    "anchor_ce_drift": -0.88,
                    "transfer_ce_improvement": 0.92,
                    "commutator_anchor_logit_mse": 0.04,
                    "commutator_transfer_logit_mse": 0.03,
                    "commutator_anchor_residual_stream_l2": 0.4,
                    "commutator_transfer_residual_stream_l2": 0.3,
                },
                {
                    "variant": "norm_matched_dense_active_rank",
                    "anchor_ce_drift": -0.42,
                    "transfer_ce_improvement": 0.42,
                    "commutator_anchor_logit_mse": 0.06,
                    "commutator_transfer_logit_mse": 0.05,
                },
            ]
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    unittest.main()
