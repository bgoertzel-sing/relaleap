from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.support_audit import run_support_audit


class SupportAuditTest(unittest.TestCase):
    def test_support_audit_writes_exhaustive_pair_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_support_audit
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

            summary = run_support_audit(config_path, tmp_path / "audit")

            self.assertEqual(summary["status"], "ok")
            audit = summary["audit"]
            self.assertEqual(audit["num_columns"], 4)
            self.assertEqual(audit["top_k"], 2)
            self.assertEqual(audit["support_router"], "contextual_mlp")
            self.assertEqual(audit["contextual_router_hidden_dim"], 16)
            self.assertEqual(audit["support_set_count"], 6)
            self.assertEqual(audit["singleton_count"], 4)
            self.assertIn("oracle_support_regret", audit)
            self.assertIn("best_one_swap_support", audit)
            self.assertIn("router_oracle_target_diagnostic", audit)
            self.assertIn(
                "holdout",
                audit["router_oracle_target_diagnostic"],
            )
            self.assertIn("router_oracle_target_nonlinear_diagnostic", audit)
            self.assertIn(
                "holdout",
                audit["router_oracle_target_nonlinear_diagnostic"],
            )
            self.assertIn("router_oracle_target_contextual_diagnostic", audit)
            self.assertIn(
                "holdout",
                audit["router_oracle_target_contextual_diagnostic"],
            )
            self.assertIn("contextual_router_support_intervention", audit)
            self.assertIn(
                "holdout",
                audit["contextual_router_support_intervention"],
            )
            self.assertIn("contextual_router_support_head", audit)
            self.assertIn(
                "holdout",
                audit["contextual_router_support_head"],
            )
            self.assertEqual(audit["support_audit"]["top_k"], 2)

            saved = json.loads((tmp_path / "audit" / "summary.json").read_text())
            self.assertEqual(saved["audit"]["support_set_count"], 6)
            self.assertTrue((tmp_path / "audit" / "support_losses.csv").is_file())
            self.assertTrue((tmp_path / "audit" / "pairwise_synergy.csv").is_file())
            self.assertTrue(
                (tmp_path / "audit" / "router_target_diagnostic.csv").is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "audit"
                    / "router_target_nonlinear_diagnostic.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "audit"
                    / "router_target_contextual_diagnostic.csv"
                ).is_file()
            )
            self.assertTrue(
                (tmp_path / "audit" / "router_support_intervention.csv").is_file()
            )
            self.assertTrue(
                (tmp_path / "audit" / "contextual_router_support_head.csv").is_file()
            )
            self.assertTrue((tmp_path / "audit" / "notes.md").is_file())

            with (tmp_path / "audit" / "pairwise_synergy.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertIn("pairwise_synergy", rows[0])

            with (tmp_path / "audit" / "router_target_diagnostic.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                diagnostic_rows = list(csv.DictReader(handle))
            self.assertEqual(len(diagnostic_rows), 3)
            self.assertIn("oracle_gap_recovery_fraction", diagnostic_rows[0])

            with (tmp_path / "audit" / "router_target_nonlinear_diagnostic.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                nonlinear_diagnostic_rows = list(csv.DictReader(handle))
            self.assertEqual(len(nonlinear_diagnostic_rows), 3)
            self.assertIn(
                "oracle_gap_recovery_fraction",
                nonlinear_diagnostic_rows[0],
            )

            with (tmp_path / "audit" / "router_target_contextual_diagnostic.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                contextual_diagnostic_rows = list(csv.DictReader(handle))
            self.assertEqual(len(contextual_diagnostic_rows), 3)
            self.assertIn(
                "oracle_gap_recovery_fraction",
                contextual_diagnostic_rows[0],
            )

            with (tmp_path / "audit" / "router_support_intervention.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                intervention_rows = list(csv.DictReader(handle))
            self.assertEqual(len(intervention_rows), 3)
            self.assertIn("intervention_loss", intervention_rows[0])
            self.assertIn("intervention_minus_router_loss", intervention_rows[0])

            with (tmp_path / "audit" / "contextual_router_support_head.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                support_head_rows = list(csv.DictReader(handle))
            self.assertEqual(len(support_head_rows), 3)
            self.assertIn("intervention_loss", support_head_rows[0])
            self.assertIn("intervention_minus_router_loss", support_head_rows[0])

    def test_support_audit_requires_top_k_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_support_audit_bad_topk
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
                run_support_audit(config_path, tmp_path / "audit")


if __name__ == "__main__":
    unittest.main()
