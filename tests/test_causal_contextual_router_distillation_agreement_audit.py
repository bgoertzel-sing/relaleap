from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_distillation_agreement_audit import (
    AGREEMENT_BLOCKED,
    AGREEMENT_SUPPORTED,
    run_causal_contextual_router_distillation_agreement_audit,
)


class CausalContextualRouterDistillationAgreementAuditTest(unittest.TestCase):
    def test_agreement_audit_writes_per_token_and_intervention_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_distillation_agreement_audit
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

            prior_audit = root / "prior"
            prior_audit.mkdir()
            (prior_audit / "summary.json").write_text("{}\n", encoding="utf-8")
            summary = run_causal_contextual_router_distillation_agreement_audit(
                config_path,
                root / "audit",
                audit_dir=prior_audit,
                runpod_audit_dir=None,
                max_folds=2,
                teacher_oracle_weight=0.01,
                student_distill_weight=0.01,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(summary["decision"], {AGREEMENT_SUPPORTED, AGREEMENT_BLOCKED})
            audit = summary["audit"]
            self.assertEqual(audit["fold_count"], 2)
            self.assertIn("student_vs_teacher", audit["agreement_aggregates"])
            self.assertIn(
                "teacher_student_disagreement_tokens::teacher_support_forced_into_student",
                audit["intervention_aggregates"],
            )
            self.assertIn(
                "causal_distilled_from_shuffled_teacher_0.01",
                audit["null_control_aggregates"],
            )
            self.assertIn(
                "causal_distilled_from_frequency_matched_teacher_0.01",
                audit["null_control_aggregates"],
            )
            self.assertIn(
                "causal_distilled_from_token_position_frequency_matched_teacher_0.01",
                audit["null_control_aggregates"],
            )
            self.assertEqual(
                audit["source_artifact_assessment"][0]["action"],
                "fail_closed_bounded_rerun",
            )
            self.assertTrue((root / "audit" / "summary.json").is_file())
            self.assertTrue((root / "audit" / "fold_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "aggregate_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "agreement_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "intervention_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "null_control_metrics.csv").is_file())
            self.assertTrue((root / "audit" / "null_sampling_diagnostics.csv").is_file())
            self.assertTrue((root / "audit" / "per_token_supports.csv").is_file())
            self.assertTrue((root / "audit" / "support_counts.csv").is_file())
            self.assertTrue((root / "audit" / "notes.md").is_file())

            saved = json.loads((root / "audit" / "summary.json").read_text())
            token_row = (
                (root / "audit" / "per_token_supports.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("teacher_support", token_row)
            self.assertIn("student_router_support_loss", token_row)
            self.assertIn("token_position_null_support", token_row)
            self.assertIn("token_position_null_support_forced_into_student_loss", token_row)
            self.assertGreater(len(saved["audit"]["agreement_rows"]), 0)
            self.assertGreater(len(saved["audit"]["null_control_rows"]), 0)
            self.assertIn(
                "teacher_student_disagreement_tokens::token_position_null_support_forced_into_student",
                saved["audit"]["intervention_aggregates"],
            )

    def test_requires_causal_contextual_topk2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_distillation_agreement_bad
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
                run_causal_contextual_router_distillation_agreement_audit(
                    config_path,
                    root / "audit",
                    runpod_audit_dir=None,
                    max_folds=1,
                )


if __name__ == "__main__":
    unittest.main()
