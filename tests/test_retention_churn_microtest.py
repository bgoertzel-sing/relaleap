from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.retention_churn_microtest import (
    run_retention_churn_microtest,
)


class RetentionChurnMicrotestTest(unittest.TestCase):
    def test_retention_churn_microtest_writes_control_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_retention_churn
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

            summary = run_retention_churn_microtest(config_path, tmp_path / "audit")

            self.assertEqual(summary["status"], "ok")
            audit = summary["audit"]
            self.assertEqual(audit["training_steps_per_slice"], 2)
            variants = {row["variant"]: row for row in audit["variants"]}
            self.assertEqual(
                sorted(variants),
                [
                    "norm_matched_dense_active_rank",
                    "promoted_contextual_topk2",
                    "random_fixed_topk2",
                    "rank_matched_contextual_topk1",
                ],
            )
            self.assertEqual(variants["promoted_contextual_topk2"]["top_k"], 2)
            self.assertEqual(variants["rank_matched_contextual_topk1"]["top_k"], 1)
            self.assertEqual(variants["random_fixed_topk2"]["kind"], "sparse_fixed")
            self.assertEqual(
                variants["random_fixed_topk2"][
                    "anchor_support_churn_after_transfer"
                ],
                0.0,
            )
            self.assertEqual(variants["norm_matched_dense_active_rank"]["kind"], "dense")
            self.assertIn("anchor_ce_drift", variants["promoted_contextual_topk2"])
            self.assertIn(
                "anchor_support_churn_after_transfer",
                variants["rank_matched_contextual_topk1"],
            )
            self.assertEqual(
                variants["norm_matched_dense_active_rank"][
                    "anchor_support_churn_after_transfer"
                ],
                "",
            )

            saved = json.loads((tmp_path / "audit" / "summary.json").read_text())
            self.assertEqual(len(saved["audit"]["variants"]), 4)
            self.assertTrue((tmp_path / "audit" / "variant_metrics.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "phase_metrics.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "per_token_commutator.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "notes.md").is_file())

            with (tmp_path / "audit" / "variant_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 4)
            self.assertIn("anchor_logit_mse_drift", metric_rows[0])
            self.assertIn("transfer_ce_improvement", metric_rows[0])
            self.assertIn("commutator_anchor_logit_mse", metric_rows[0])
            self.assertIn("commutator_transfer_logit_mse", metric_rows[0])
            self.assertIn("order_averaged_anchor_ce_loss", metric_rows[0])
            self.assertIn("order_averaged_anchor_ce_delta_vs_forward", metric_rows[0])
            self.assertIn("order_averaged_anchor_logit_mse_to_forward", metric_rows[0])
            self.assertIn(
                "order_averaged_anchor_residual_stream_l2_to_forward",
                metric_rows[0],
            )
            self.assertIn("same_order_ensemble_anchor_ce_loss", metric_rows[0])
            self.assertIn(
                "same_order_ensemble_anchor_ce_delta_vs_best_endpoint",
                metric_rows[0],
            )
            self.assertIn(
                "same_order_ensemble_anchor_logit_mse_to_primary",
                metric_rows[0],
            )
            self.assertIn(
                "same_order_identical_replay_nonperturbation_pass",
                metric_rows[0],
            )
            promoted = next(
                row
                for row in metric_rows
                if row["variant"] == "promoted_contextual_topk2"
            )
            self.assertEqual(
                promoted["same_order_identical_replay_nonperturbation_pass"],
                "True",
            )
            self.assertLessEqual(
                float(promoted["same_order_identical_anchor_logit_mse_to_primary"]),
                1e-12,
            )
            self.assertLessEqual(
                float(promoted["same_order_identical_transfer_logit_mse_to_primary"]),
                1e-12,
            )
            topk1 = next(
                row
                for row in metric_rows
                if row["variant"] == "rank_matched_contextual_topk1"
            )
            self.assertNotEqual(topk1["commutator_anchor_support_churn"], "")

            with (tmp_path / "audit" / "phase_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                phase_rows = list(csv.DictReader(handle))
            self.assertEqual(len(phase_rows), 16)
            self.assertIn("residual_norm_mean", phase_rows[0])

            with (tmp_path / "audit" / "per_token_commutator.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                token_rows = list(csv.DictReader(handle))
            self.assertEqual(len(token_rows), 4 * 2 * 4 * 15)
            self.assertIn("ce_delta_forward_minus_reverse", token_rows[0])
            self.assertIn("symmetric_kl", token_rows[0])
            self.assertIn("logit_mse", token_rows[0])
            self.assertIn("residual_delta_l2", token_rows[0])
            self.assertIn("support_churn", token_rows[0])
            self.assertIn("forward_support", token_rows[0])

    def test_retention_churn_microtest_requires_contextual_topk_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: bad_retention_churn
  seed: 1
  max_steps: 1

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
    top_k: 1
    support_router: contextual_mlp
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "top_k: 2"):
                run_retention_churn_microtest(config_path, tmp_path / "audit")


if __name__ == "__main__":
    unittest.main()
