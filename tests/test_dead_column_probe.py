from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dead_column_probe import run_dead_column_probe


class DeadColumnProbeTest(unittest.TestCase):
    def test_dead_column_probe_writes_variant_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_dead_column_probe
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

inference:
  pc_steps: 2
  hep_alpha: 0.0
  hep_alpha_sweep: "0.0,0.25"
  hep_update_clip_norm: 0.01
  hep_settling_objective: temporal_consistency_gradient

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_dead_column_probe(
                config_path,
                tmp_path / "probe",
                load_balance_weights=[0.0, 0.01],
            )

            self.assertEqual(summary["status"], "ok")
            probe = summary["probe"]
            self.assertEqual(probe["num_columns"], 4)
            self.assertEqual(probe["top_k"], 2)
            self.assertEqual(probe["support_router"], "contextual_mlp")
            self.assertEqual(probe["load_balance_weights"], [0.0, 0.01])
            self.assertEqual(len(probe["variants"]), 2)
            self.assertIn("decision", probe)
            self.assertIn(
                probe["decision"]["status"],
                {"recruited_without_ce_hurt", "no_safe_recruitment"},
            )
            self.assertTrue((tmp_path / "probe" / "summary.json").is_file())
            self.assertTrue((tmp_path / "probe" / "variant_metrics.csv").is_file())
            self.assertTrue((tmp_path / "probe" / "notes.md").is_file())

            saved = json.loads((tmp_path / "probe" / "summary.json").read_text())
            self.assertEqual(saved["probe"]["baseline_variant"], "baseline")
            with (tmp_path / "probe" / "variant_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("alpha0_ce_delta_from_baseline", rows[0])
            self.assertIn("used_column_delta_from_baseline", rows[0])


if __name__ == "__main__":
    unittest.main()
