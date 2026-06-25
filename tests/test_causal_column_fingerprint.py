from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_column_fingerprint import (
    run_causal_column_fingerprint,
)


class CausalColumnFingerprintTest(unittest.TestCase):
    def test_causal_column_fingerprint_writes_intervention_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_column_fingerprint
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

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_column_fingerprint(
                config_path,
                tmp_path / "fingerprint",
                load_balance_weights=[0.0, 0.01],
                max_pair_rows=2,
            )

            self.assertEqual(summary["status"], "ok")
            audit = summary["audit"]
            self.assertEqual(audit["num_columns"], 4)
            self.assertEqual(audit["top_k"], 2)
            self.assertEqual(audit["support_router"], "contextual_mlp")
            self.assertEqual(audit["load_balance_weights"], [0.0, 0.01])
            self.assertEqual(len(audit["variants"]), 2)
            self.assertEqual(audit["column_fingerprint_count"], 8)
            self.assertGreater(audit["pair_intervention_count"], 0)
            self.assertGreater(audit["per_token_pair_intervention_count"], 0)
            self.assertEqual(len(audit["heldout_stability"]), 2)
            self.assertEqual(len(audit["functional_churn"]), 2)
            self.assertIn(
                "position_ablate_delta_correlation",
                audit["heldout_stability"][0],
            )
            self.assertIn(
                "batch_ablate_delta_correlation",
                audit["heldout_stability"][0],
            )
            self.assertIn(
                "adjacent_support_identity_churn_fraction",
                audit["functional_churn"][0],
            )
            self.assertIn(
                "previous_support_changed_logit_mse_mean",
                audit["functional_churn"][0],
            )

            saved = json.loads(
                (tmp_path / "fingerprint" / "summary.json").read_text()
            )
            self.assertEqual(saved["audit"]["column_fingerprint_count"], 8)
            self.assertTrue(
                (tmp_path / "fingerprint" / "column_fingerprints.csv").is_file()
            )
            self.assertTrue(
                (tmp_path / "fingerprint" / "pair_interventions.csv").is_file()
            )
            self.assertTrue(
                (
                    tmp_path / "fingerprint" / "per_token_pair_interventions.csv"
                ).is_file()
            )
            self.assertTrue((tmp_path / "fingerprint" / "notes.md").is_file())

            with (tmp_path / "fingerprint" / "column_fingerprints.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                column_rows = list(csv.DictReader(handle))
            self.assertEqual(len(column_rows), 8)
            self.assertIn("ablate_loss_delta", column_rows[0])
            self.assertIn("force_loss_delta", column_rows[0])
            self.assertIn("force_residual_stream_l2_delta", column_rows[0])

            with (tmp_path / "fingerprint" / "pair_interventions.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertGreater(len(pair_rows), 0)
            self.assertIn("fixed_support_loss_delta", pair_rows[0])
            self.assertIn("empty_loss", pair_rows[0])
            self.assertIn("singleton_left_loss", pair_rows[0])
            self.assertIn("singleton_right_loss", pair_rows[0])
            self.assertIn("pair_loss", pair_rows[0])
            self.assertIn("pair_gain", pair_rows[0])
            self.assertIn("pair_synergy", pair_rows[0])
            self.assertIn("position_bin", pair_rows[0])
            self.assertIn("pair_value_cosine", pair_rows[0])
            self.assertTrue(
                {"all", "even", "odd"}.issubset(
                    {row["position_bin"] for row in pair_rows}
                )
            )
            self.assertTrue(
                {"common_target", "rare_target"}.issubset(
                    {row["token_class"] for row in pair_rows}
                )
            )

            with (
                tmp_path / "fingerprint" / "per_token_pair_interventions.csv"
            ).open(newline="", encoding="utf-8") as handle:
                per_token_rows = list(csv.DictReader(handle))
            self.assertGreater(len(per_token_rows), 0)
            self.assertIn("batch_index", per_token_rows[0])
            self.assertIn("position_index", per_token_rows[0])
            self.assertIn("token_index", per_token_rows[0])
            self.assertIn("singleton_left_loss", per_token_rows[0])
            self.assertIn("singleton_right_loss", per_token_rows[0])
            self.assertIn("singleton_left_gain", per_token_rows[0])
            self.assertIn("singleton_right_gain", per_token_rows[0])
            self.assertIn("pair_gain", per_token_rows[0])
            self.assertIn("pair_synergy", per_token_rows[0])
            self.assertIn("residual_norm_bin", per_token_rows[0])
            self.assertIn("residual_gain_bin", per_token_rows[0])
            self.assertIn("active_rank_proxy", per_token_rows[0])
            self.assertTrue(
                {"low", "mid", "high"} & {row["residual_norm_bin"] for row in per_token_rows}
            )

    def test_causal_column_fingerprint_can_include_rank_matched_topk1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_column_fingerprint_topk1
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
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_column_fingerprint(
                config_path,
                tmp_path / "fingerprint",
                load_balance_weights=[0.0],
                include_rank_matched_topk1=True,
                max_pair_rows=1,
            )

            variants = summary["audit"]["variants"]
            self.assertEqual(
                [row["variant"] for row in variants],
                ["baseline", "rank_matched_topk1_contextual"],
            )
            self.assertEqual(variants[1]["top_k"], 1)
            self.assertEqual(variants[1]["num_columns"], 8)
            self.assertEqual(summary["audit"]["column_fingerprint_count"], 12)
            self.assertEqual(len(summary["audit"]["heldout_stability"]), 2)
            self.assertEqual(len(summary["audit"]["functional_churn"]), 2)
            with (tmp_path / "fingerprint" / "pair_interventions.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertTrue(
                any(
                    row["variant"] == "rank_matched_topk1_contextual"
                    and row["intervention"] == "fixed_dominant_router_singleton"
                    for row in pair_rows
                )
            )

    def test_causal_column_fingerprint_requires_top_k_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_column_fingerprint_bad_topk
  seed: 1
  max_steps: 1

data:
  dataset: tiny_shakespeare_char
  seq_len: 16

model:
  base:
    layers: 1
    hidden_dim: 32
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 1
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "top_k: 2"):
                run_causal_column_fingerprint(config_path, tmp_path / "fingerprint")


if __name__ == "__main__":
    unittest.main()
