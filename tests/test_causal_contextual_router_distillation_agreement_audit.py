from __future__ import annotations

import csv
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
                capture_hidden_future=True,
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
            self.assertTrue((root / "audit" / "hidden_future_rows.csv").is_file())
            self.assertTrue((root / "audit" / "intervention_rows_exact.csv").is_file())
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
            capture = saved["audit"]["hidden_future_capture"]
            self.assertEqual(capture["status"], "captured")
            self.assertTrue(capture["hidden_future_schema_ok"])
            self.assertTrue(capture["exact_intervention_schema_ok"])
            self.assertFalse(capture["requires_gpu_now"])
            self.assertIn(
                "teacher_student_disagreement_tokens::token_position_null_support_forced_into_student",
                saved["audit"]["intervention_aggregates"],
            )
            with (root / "audit" / "hidden_future_rows.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                hidden_rows = list(csv.DictReader(handle))
            self.assertGreater(len(hidden_rows), 0)
            self.assertIn("current_hidden_json", hidden_rows[0])
            self.assertIn("future_hidden_json", hidden_rows[0])
            self.assertIn("teacher_support_logits_json", hidden_rows[0])
            self.assertIn("forbidden_predictor_fields", hidden_rows[0])
            self.assertIn("future_hidden_json", hidden_rows[0]["forbidden_predictor_fields"])

            with (root / "audit" / "intervention_rows_exact.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                exact_rows = list(csv.DictReader(handle))
            self.assertGreater(len(exact_rows), 0)
            self.assertIn("forced_support_pair", exact_rows[0])
            self.assertIn("forced_support_loss", exact_rows[0])
            self.assertIn("oracle_support_loss", exact_rows[0])

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

    def test_hidden_future_capture_can_include_train_split_for_single_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_train_capture
  seed: 1
  max_steps: 1

data:
  dataset: tiny_shakespeare_word
  seq_len: 12

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 24
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp_causal
    contextual_router_hidden_dim: 12
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_distillation_agreement_audit(
                config_path,
                root / "audit",
                runpod_audit_dir=None,
                max_folds=1,
                teacher_oracle_weight=0.01,
                student_distill_weight=0.01,
                capture_hidden_future=True,
                capture_train_hidden_future=True,
            )

            capture = summary["audit"]["hidden_future_capture"]
            self.assertEqual(capture["status"], "captured")
            self.assertTrue(capture["train_capture_enabled"])
            self.assertTrue(capture["split_coverage_available"])
            self.assertGreater(capture["train_sequence_count"], 0)
            self.assertGreater(capture["heldout_sequence_count"], 0)
            self.assertGreater(capture["hidden_future_split_counts"]["train"], 0)
            self.assertGreater(capture["hidden_future_split_counts"]["heldout"], 0)

            with (root / "audit" / "hidden_future_rows.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                hidden_rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in hidden_rows}, {"train", "heldout"})
            splits_by_sequence: dict[str, set[str]] = {}
            for row in hidden_rows:
                splits_by_sequence.setdefault(row["sequence_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_sequence.values()))

            with (root / "audit" / "intervention_rows_exact.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                exact_rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in exact_rows}, {"train", "heldout"})
            pair_counts: dict[tuple[str, str, str, str], set[str]] = {}
            for row in exact_rows:
                key = (
                    row["split"],
                    row["sequence_id"],
                    row["flat_position"],
                    row["position_index"],
                )
                pair_counts.setdefault(key, set()).add(row["forced_support_pair"])
            self.assertTrue(pair_counts)
            self.assertEqual({len(pairs) for pairs in pair_counts.values()}, {6})

    def test_train_hidden_future_capture_can_scale_batch_without_cross_fold_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_scaled_train_capture
  seed: 1
  max_steps: 1

data:
  dataset: tiny_shakespeare_word
  seq_len: 12

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 24
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp_causal
    contextual_router_hidden_dim: 12
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_distillation_agreement_audit(
                config_path,
                root / "audit",
                runpod_audit_dir=None,
                max_folds=1,
                teacher_oracle_weight=0.01,
                student_distill_weight=0.01,
                batch_size=5,
                capture_hidden_future=True,
                capture_train_hidden_future=True,
            )

            capture = summary["audit"]["hidden_future_capture"]
            self.assertEqual(capture["status"], "captured")
            self.assertEqual(summary["audit"]["batch_size"], 5)
            self.assertEqual(capture["train_sequence_count"], 4)
            self.assertEqual(capture["heldout_sequence_count"], 1)

            with (root / "audit" / "hidden_future_rows.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                hidden_rows = list(csv.DictReader(handle))
            splits_by_sequence: dict[str, set[str]] = {}
            for row in hidden_rows:
                splits_by_sequence.setdefault(row["sequence_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_sequence.values()))

            with (root / "audit" / "intervention_rows_exact.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                exact_rows = list(csv.DictReader(handle))
            pair_counts: dict[tuple[str, str, str, str], set[str]] = {}
            for row in exact_rows:
                key = (
                    row["split"],
                    row["sequence_id"],
                    row["flat_position"],
                    row["position_index"],
                )
                pair_counts.setdefault(key, set()).add(row["forced_support_pair"])
            self.assertTrue(pair_counts)
            self.assertEqual({len(pairs) for pairs in pair_counts.values()}, {6})

    def test_train_hidden_future_capture_requires_single_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_causal_contextual_router_train_capture_bad
  seed: 1
  max_steps: 1

data:
  dataset: tiny_shakespeare_word
  seq_len: 12

model:
  base:
    layers: 1
    hidden_dim: 24
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    support_router: contextual_mlp_causal
    contextual_router_hidden_dim: 12
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_folds=1"):
                run_causal_contextual_router_distillation_agreement_audit(
                    config_path,
                    root / "audit",
                    runpod_audit_dir=None,
                    max_folds=2,
                    capture_hidden_future=True,
                    capture_train_hidden_future=True,
                )


if __name__ == "__main__":
    unittest.main()
