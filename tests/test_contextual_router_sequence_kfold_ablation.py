from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_sequence_kfold_ablation import (
    run_contextual_router_sequence_kfold_ablation,
)


class ContextualRouterSequenceKfoldAblationTest(unittest.TestCase):
    def test_sequence_kfold_ablation_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_contextual_router_sequence_kfold_ablation
  seed: 1
  max_steps: 1

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

            summary = run_contextual_router_sequence_kfold_ablation(
                config_path,
                root / "report",
                max_folds=2,
            )

            self.assertEqual(summary["status"], "ok")
            self.assertIn(
                summary["decision"],
                {
                    "causal_contextual_router_sequence_holdout_candidate",
                    "causal_contextual_router_sequence_holdout_underperforms_linear",
                    "sequence_kfold_causal_feature_ablation_completed",
                    "future_context_features_material_for_promoted_router",
                    "promoted_contextual_router_sequence_holdout_underperforms_linear",
                },
            )
            ablation = summary["ablation"]
            self.assertEqual(ablation["fold_count"], 2)
            self.assertEqual(ablation["promoted_support_router"], "contextual_mlp")
            self.assertEqual(ablation["promoted_top_k"], 2)
            self.assertIn(
                "promoted_contextual_topk2:actual_full_context",
                ablation["variants"],
            )
            self.assertIn(
                "promoted_contextual_topk2:causal_current_past_position",
                ablation["variants"],
            )
            self.assertIn(
                "causal_contextual_topk2:actual_causal_context",
                ablation["variants"],
            )
            self.assertIn("linear_topk2_control:linear_actual", ablation["variants"])
            self.assertIn(
                "contextual_topk1_control:actual_full_context",
                ablation["variants"],
            )
            self.assertIn(
                "causal_contextual_topk1_control:actual_causal_context",
                ablation["variants"],
            )
            self.assertFalse(
                ablation["variants"][
                    "causal_contextual_topk2:actual_causal_context"
                ]["uses_future_context"]
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "fold_metrics.csv").is_file())
            self.assertTrue((root / "report" / "variant_metrics.csv").is_file())
            self.assertTrue((root / "report" / "support_counts.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

            saved = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(saved["ablation"]["fold_count"], 2)
            with (root / "report" / "variant_metrics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                variant_rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(variant_rows), 6)
            self.assertIn("mean_router_loss", variant_rows[0])
            self.assertIn("causal_feature_safe", variant_rows[0])
            self.assertIn("causal_contextual_vs_linear_loss_delta", saved["ablation"])

    def test_sequence_kfold_ablation_requires_contextual_topk2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_contextual_router_sequence_kfold_ablation_bad
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
                run_contextual_router_sequence_kfold_ablation(
                    config_path,
                    root / "report",
                    max_folds=1,
                )


if __name__ == "__main__":
    unittest.main()
