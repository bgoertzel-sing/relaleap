from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.column_redundancy import run_column_redundancy


class ColumnRedundancyTest(unittest.TestCase):
    def test_column_redundancy_writes_load_and_similarity_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_column_redundancy
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

            summary = run_column_redundancy(config_path, tmp_path / "audit")

            self.assertEqual(summary["status"], "ok")
            diagnostic = summary["diagnostic"]
            self.assertEqual(diagnostic["num_columns"], 4)
            self.assertEqual(diagnostic["top_k"], 2)
            self.assertEqual(diagnostic["support_router"], "contextual_mlp")
            self.assertIn("effective_num_columns", diagnostic)
            self.assertIn("normalized_load_entropy", diagnostic)
            self.assertIn("high_similarity_active_pair_count", diagnostic)
            self.assertIn("support_audit", diagnostic)
            self.assertEqual(diagnostic["support_audit"]["num_columns"], 4)

            saved = json.loads((tmp_path / "audit" / "summary.json").read_text())
            self.assertEqual(saved["diagnostic"]["num_columns"], 4)
            self.assertTrue((tmp_path / "audit" / "column_loads.csv").is_file())
            self.assertTrue(
                (tmp_path / "audit" / "column_pair_similarity.csv").is_file()
            )
            self.assertTrue((tmp_path / "audit" / "notes.md").is_file())

            with (tmp_path / "audit" / "column_loads.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                load_rows = list(csv.DictReader(handle))
            self.assertEqual(len(load_rows), 4)
            self.assertIn("support_fraction", load_rows[0])
            self.assertIn("value_norm", load_rows[0])
            self.assertIn("score_mean_rank", load_rows[0])

            with (tmp_path / "audit" / "column_pair_similarity.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 6)
            self.assertIn("cosine_similarity", pair_rows[0])
            self.assertIn("co_selected_count", pair_rows[0])


if __name__ == "__main__":
    unittest.main()
