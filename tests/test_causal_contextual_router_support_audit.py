from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_support_audit import (
    CAUSAL_SUPPORT_AUDIT_BLOCKED,
    CAUSAL_SUPPORT_AUDIT_PASSED,
    run_causal_contextual_router_support_audit,
)


class CausalContextualRouterSupportAuditTest(unittest.TestCase):
    def test_support_audit_writes_artifacts_and_blocks_or_passes_by_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_support_audit
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

            summary = run_causal_contextual_router_support_audit(
                config_path,
                root / "audit",
                max_folds=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {CAUSAL_SUPPORT_AUDIT_PASSED, CAUSAL_SUPPORT_AUDIT_BLOCKED},
            )
            audit = summary["audit"]
            self.assertEqual(audit["fold_count"], 2)
            self.assertEqual(audit["support_router"], "contextual_mlp_causal")
            self.assertIn("causal_contextual_topk2", audit["aggregate_metrics"])
            self.assertIn("linear_topk2", audit["aggregate_metrics"])
            self.assertIn("full_context_oracle_topk2", audit["aggregate_metrics"])
            self.assertGreaterEqual(len(audit["gate_criteria"]), 4)
            self.assertTrue((root / "audit" / "summary.json").is_file())
            self.assertTrue((root / "audit" / "fold_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "aggregate_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "control_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "support_counts.csv").is_file())
            self.assertTrue((root / "audit" / "notes.md").is_file())

            saved = json.loads((root / "audit" / "summary.json").read_text())
            causal = saved["audit"]["aggregate_metrics"]["causal_contextual_topk2"]
            self.assertIn("mean_oracle_support_regret", causal)
            self.assertIn("mean_functional_churn_logit_l1", causal)
            self.assertIn("mean_support_load_entropy", causal)

    def test_requires_causal_contextual_topk2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_support_audit_bad
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
                run_causal_contextual_router_support_audit(
                    config_path,
                    root / "audit",
                    max_folds=1,
                )


if __name__ == "__main__":
    unittest.main()
