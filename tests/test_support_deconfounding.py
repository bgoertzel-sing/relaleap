from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.support_deconfounding import run_support_deconfounding


class SupportDeconfoundingTest(unittest.TestCase):
    def test_support_deconfounding_writes_control_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_support_deconfounding
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
    support_stress: true
    support_stress_preset: false
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_support_deconfounding(config_path, tmp_path / "audit")

            self.assertEqual(summary["status"], "ok")
            audit = summary["audit"]
            self.assertEqual(audit["baseline_num_columns"], 4)
            self.assertEqual(audit["baseline_top_k"], 2)
            self.assertEqual(audit["variant_count"], 8)
            variants = {row["variant"]: row for row in audit["variants"]}
            self.assertIn("learned_topk2_contextual", variants)
            self.assertIn("rank_matched_topk1_contextual", variants)
            self.assertIn("random_fixed_topk2", variants)
            self.assertIn("learned_topk2_scale_one_over_k", variants)
            self.assertIn("learned_topk2_scale_one_over_sqrt_k", variants)
            self.assertIn("dense_rank_flop_matched_residual", variants)
            self.assertIn("dense_rank_flop_matched_norm_matched", variants)
            self.assertIn("dense_stored_parameter_matched_residual", variants)
            self.assertEqual(variants["rank_matched_topk1_contextual"]["top_k"], 1)
            self.assertEqual(variants["rank_matched_topk1_contextual"]["num_columns"], 8)
            self.assertEqual(variants["dense_rank_flop_matched_residual"]["kind"], "dense")
            self.assertEqual(
                variants["dense_rank_flop_matched_norm_matched"]["kind"],
                "dense",
            )
            self.assertAlmostEqual(
                variants["dense_rank_flop_matched_norm_matched"][
                    "residual_norm_mean"
                ],
                variants["learned_topk2_contextual"]["residual_norm_mean"],
                places=5,
            )
            self.assertIn("oracle_support_regret", variants["learned_topk2_contextual"])
            self.assertIn("support_margin_mean", variants["learned_topk2_contextual"])
            self.assertIn("residual_norm_mean", variants["learned_topk2_contextual"])
            self.assertIn("norm_match_scale", variants["dense_rank_flop_matched_norm_matched"])
            self.assertIn(
                "stored_parameter_ratio_to_sparse",
                variants["dense_stored_parameter_matched_residual"],
            )

            saved = json.loads((tmp_path / "audit" / "summary.json").read_text())
            self.assertEqual(saved["audit"]["variant_count"], 8)
            self.assertTrue((tmp_path / "audit" / "variant_metrics.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "column_interventions.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "support_churn.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "notes.md").is_file())

            with (tmp_path / "audit" / "variant_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 8)
            self.assertIn("stored_parameters", metric_rows[0])
            self.assertIn("active_parameters_proxy", metric_rows[0])
            self.assertIn("raw_residual_norm_mean", metric_rows[0])
            self.assertIn("norm_match_scale", metric_rows[0])
            self.assertIn("stored_parameter_ratio_to_sparse", metric_rows[0])

            with (tmp_path / "audit" / "column_interventions.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                intervention_rows = list(csv.DictReader(handle))
            self.assertGreater(len(intervention_rows), 0)
            self.assertIn("force_loss_delta", intervention_rows[0])

            with (tmp_path / "audit" / "support_churn.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                churn_rows = list(csv.DictReader(handle))
            self.assertGreater(len(churn_rows), 0)
            self.assertIn("support_churn_jaccard", churn_rows[0])

    def test_support_deconfounding_requires_contextual_topk_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: bad_support_deconfounding
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
                run_support_deconfounding(config_path, tmp_path / "audit")


if __name__ == "__main__":
    unittest.main()
