from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_shortcut_ablation import (
    run_contextual_router_shortcut_ablation,
)


class ContextualRouterShortcutAblationTest(unittest.TestCase):
    def test_ablation_writes_variant_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_contextual_router_shortcut_ablation
  seed: 1
  max_steps: 2

data:
  dataset: tiny_shakespeare_word
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

            summary = run_contextual_router_shortcut_ablation(
                config_path,
                root / "audit",
                probe_steps=3,
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(
                summary["decision"],
                "contextual_router_shortcut_ablation_completed",
            )
            ablation = summary["ablation"]
            self.assertEqual(ablation["support_router"], "contextual_mlp")
            self.assertEqual(ablation["top_k"], 2)
            self.assertEqual(ablation["support_set_count"], 6)
            self.assertEqual(
                set(ablation["variants"]),
                {"hidden_only", "position_only", "context_only", "full_context"},
            )
            self.assertIn(ablation["selected_variant"], ablation["variants"])
            self.assertIn("shortcut_interpretation", ablation)
            for variant in ablation["variants"].values():
                self.assertIn("holdout", variant)
                self.assertIn(
                    "intervention_oracle_gap_recovery_fraction",
                    variant["holdout"],
                )

            saved = json.loads((root / "audit" / "summary.json").read_text())
            self.assertEqual(saved["ablation"]["support_set_count"], 6)
            self.assertTrue((root / "audit" / "variant_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "notes.md").is_file())

            with (root / "audit" / "variant_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertIn("intervention_loss", rows[0])

    def test_ablation_requires_contextual_topk2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_contextual_router_shortcut_ablation_bad
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
    support_router: linear
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "top_k: 2"):
                run_contextual_router_shortcut_ablation(
                    config_path,
                    root / "audit",
                    probe_steps=1,
                )


if __name__ == "__main__":
    unittest.main()
