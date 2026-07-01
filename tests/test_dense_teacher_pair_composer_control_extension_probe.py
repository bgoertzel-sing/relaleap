from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from relaleap.experiments.dense_teacher_pair_composer_control_extension_probe import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_dense_teacher_pair_composer_control_extension_probe,
)


class DenseTeacherPairComposerControlExtensionProbeTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_pair_composer_control_extension_probe(
                distillation_dir=root / "missing-distillation",
                design_path=root / "missing-design.json",
                truth_audit_path=root / "missing-truth.json",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_learned_router_and_sentinel_rows_from_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            distillation = root / "distillation"
            distillation.mkdir()
            design = root / "design.json"
            truth = root / "truth.json"
            _write_json(
                design,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_control_extension_design_recorded",
                    "selected_next_action": "implement_pair_composer_control_extension_probe_locally",
                },
            )
            _write_json(truth, {"status": "pass", "decision": "dense_teacher_pair_composer_truth_audit_gpu_blocked"})
            _write_tensor_bundle(distillation)

            summary = run_dense_teacher_pair_composer_control_extension_probe(
                distillation_dir=distillation,
                design_path=design,
                truth_audit_path=truth,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_teacher_pair_composer_control_extension_probe_gpu_blocked")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreater(summary["learned_router_holdout_true_decoder_ce_loss"], 0.0)
            self.assertIsNotNone(summary["learned_router_holdout_support_pair_accuracy"])
            router_arms = {row["arm"] for row in summary["router_rows"]}
            sentinel_arms = {row["arm"] for row in summary["sentinel_rows"]}
            self.assertIn("learned_causal_pair_router", router_arms)
            self.assertIn("delayed_pair_target_null", sentinel_arms)
            self.assertIn("misaligned_support_pair_null", sentinel_arms)
            self.assertIn("token_position_pair_router_null", sentinel_arms)
            control_arms = {row["arm"] for row in summary["control_rows"]}
            self.assertIn("same_parameter_independent_additive_control", control_arms)
            self.assertIn("rank_norm_matched_dense_ridge_control", control_arms)
            self.assertIn("matched_mlp_random_feature_residual_control", control_arms)
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertFalse(criteria["remaining_controls_complete_for_gpu"]["passed"])
            self.assertTrue(criteria["matched_dense_mlp_control_rows_measured"]["passed"])
            self.assertTrue(criteria["exact_finite_update_commutator_measured"]["passed"])
            self.assertTrue(criteria["retention_churn_measured"]["passed"])
            self.assertFalse(criteria["remaining_controls_complete_for_gpu"]["passed"])
            self.assertIn("support_pair_class_balance_sufficient", criteria)
            self.assertFalse(criteria["support_pair_class_balance_sufficient"]["passed"])
            self.assertEqual(summary["majority_pair_holdout_support_pair_accuracy"], 1.0)
            self.assertEqual(summary["learned_router_holdout_support_pair_accuracy"], 1.0)
            learned_holdout = next(
                row
                for row in summary["router_rows"]
                if row["arm"] == "learned_causal_pair_router" and row["split"] == "holdout"
            )
            self.assertTrue(learned_holdout["exact_finite_update_commutator_measured"])
            self.assertTrue(learned_holdout["retention_churn_measured"])
            self.assertEqual(learned_holdout["commutator_ce_abs_delta"], 0.0)
            self.assertGreaterEqual(learned_holdout["functional_churn_rate"], 0.0)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_tensor_bundle(path: Path) -> None:
    torch.manual_seed(7)
    batch, seq_len, hidden, vocab, columns, top_k = 2, 5, 4, 6, 4, 2
    inputs = torch.randint(0, vocab, (batch, seq_len))
    targets = torch.randint(0, vocab, (batch, seq_len))
    base_hidden = torch.randn(batch, seq_len, hidden) * 0.2
    decoder_weight = torch.randn(vocab, hidden) * 0.3
    decoder_bias = torch.randn(vocab) * 0.05
    base_logits = base_hidden.matmul(decoder_weight.t()) + decoder_bias
    per_column_hidden = torch.randn(batch, seq_len, columns, hidden) * 0.25
    per_column_logits = torch.einsum("bsch,vh->bscv", per_column_hidden, decoder_weight)
    teacher_hidden_residual = per_column_hidden[:, :, 0, :] + 0.7 * per_column_hidden[:, :, 1, :]
    teacher_logit_residual = teacher_hidden_residual.matmul(decoder_weight.t())
    teacher_logits = base_logits + teacher_logit_residual
    learned_support_indices = torch.tensor([0, 1], dtype=torch.long).repeat(batch, seq_len, 1)
    learned_support_scores = torch.randn(batch, seq_len, top_k)
    sparse_column_value_state = {
        "top_k": top_k,
        "num_columns": columns,
        "atoms_per_column": 1,
        "atom_logits": torch.randn(columns, vocab) * 0.1,
        "atom_values": torch.randn(columns, hidden) * 0.1,
    }
    payloads = {
        "inputs.pt": inputs,
        "targets.pt": targets,
        "base_hidden.pt": base_hidden,
        "frozen_decoder_state.pt": {"lm_head_weight": decoder_weight, "lm_head_bias": decoder_bias},
        "base_logits.pt": base_logits,
        "teacher_logits.pt": teacher_logits,
        "teacher_hidden_residual.pt": teacher_hidden_residual,
        "teacher_logit_residual.pt": teacher_logit_residual,
        "learned_support_indices.pt": learned_support_indices,
        "learned_support_scores.pt": learned_support_scores,
        "per_column_hidden_contributions.pt": per_column_hidden,
        "per_column_logit_contributions.pt": per_column_logits,
        "sparse_column_value_state.pt": sparse_column_value_state,
    }
    for filename, payload in payloads.items():
        torch.save(payload, path / filename)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
