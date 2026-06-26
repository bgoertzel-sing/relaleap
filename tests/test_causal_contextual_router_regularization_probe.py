from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_regularization_probe import (
    REGULARIZATION_CANDIDATE_FOUND,
    REGULARIZATION_NOT_ESTABLISHED,
    run_causal_contextual_router_regularization_probe,
)


class CausalContextualRouterRegularizationProbeTest(unittest.TestCase):
    def test_regularization_probe_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_regularization_probe
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
    support_router: contextual_mlp_causal
    contextual_router_hidden_dim: 16

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_regularization_probe(
                config_path,
                root / "probe",
                max_folds=2,
                smooth_weights=(0.01,),
                load_balance_weights=(0.01,),
                oracle_target_weights=(0.01,),
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {REGULARIZATION_CANDIDATE_FOUND, REGULARIZATION_NOT_ESTABLISHED},
            )
            probe = summary["probe"]
            self.assertEqual(probe["fold_count"], 2)
            self.assertEqual(probe["support_router"], "contextual_mlp_causal")
            self.assertIn("causal_contextual_topk2", probe["aggregate_metrics"])
            self.assertIn("linear_topk2", probe["aggregate_metrics"])
            self.assertIn(
                "causal_contextual_score_smooth_0.01",
                probe["aggregate_metrics"],
            )
            self.assertIn(
                "causal_contextual_load_balance_0.01",
                probe["aggregate_metrics"],
            )
            self.assertIn(
                "causal_contextual_oracle_target_0.01",
                probe["aggregate_metrics"],
            )
            self.assertEqual(len(probe["variant_gate_rows"]), 3)
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "fold_metrics.csv").is_file())
            self.assertTrue((root / "probe" / "aggregate_metrics.csv").is_file())
            self.assertTrue((root / "probe" / "variant_gate.csv").is_file())
            self.assertTrue((root / "probe" / "control_metrics.csv").is_file())
            self.assertTrue((root / "probe" / "support_counts.csv").is_file())
            self.assertTrue((root / "probe" / "notes.md").is_file())

            saved = json.loads((root / "probe" / "summary.json").read_text())
            variant = saved["probe"]["variant_gate_rows"][0]
            self.assertIn("mean_oracle_regret_delta_vs_causal", variant)
            self.assertIn("mean_functional_churn_delta_vs_causal", variant)
            self.assertTrue(
                all(
                    "load_balance_weight" in row and "oracle_target_weight" in row
                    for row in saved["probe"]["fold_metrics"]
                )
            )

    def test_requires_causal_contextual_topk2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_regularization_probe_bad
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
                run_causal_contextual_router_regularization_probe(
                    config_path,
                    root / "probe",
                    max_folds=1,
                )


if __name__ == "__main__":
    unittest.main()
