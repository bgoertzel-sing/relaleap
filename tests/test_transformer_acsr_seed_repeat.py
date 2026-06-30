import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_seed_repeat import (
    run_transformer_acsr_seed_repeat,
)


class TransformerACSRSeedRepeatTests(unittest.TestCase):
    def test_seed_repeat_writes_fail_closed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_transformer_acsr_seed_repeat(
                out_dir=Path(tmp),
                seeds=(17,),
                training_steps=4,
            )
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["seed_count"], 1)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertIn("hidden_classifier_gate_pass_count", summary)
            self.assertIn("hidden_classifier_leakage_pass_count", summary)
            self.assertIn("mean_hidden_classifier_ce_gain_vs_token_position_null", summary)
            self.assertIn("mean_hidden_classifier_ce_gain_vs_learned_router", summary)
            self.assertIn(
                "mean_hidden_classifier_oracle_regret_recovery_vs_learned_router",
                summary,
            )
            self.assertIn("mean_hidden_classifier_support_overlap_with_oracle", summary)
            self.assertIn("robust_hidden_classifier_gate_passes", summary)
            self.assertIn("hidden_classifier_null_margin_gate_passes", summary)
            self.assertTrue(summary["hidden_classifier_learned_router_comparison_available"])
            self.assertFalse(summary["hidden_classifier_learned_router_gate_passes"])
            self.assertFalse(summary["hidden_classifier_sequence_heldout_gate_passes"])
            self.assertFalse(summary["hidden_classifier_rule_ood_evidence_available"])
            self.assertFalse(summary["hidden_classifier_rule_ood_gate_passes"])
            self.assertFalse(summary["hidden_classifier_churn_budget_evidence_available"])
            self.assertFalse(summary["hidden_classifier_churn_budget_gate_passes"])
            self.assertFalse(summary["hidden_classifier_commutator_budget_evidence_available"])
            self.assertFalse(summary["hidden_classifier_commutator_budget_gate_passes"])
            self.assertFalse(summary["hidden_classifier_sequence_ood_budget_audit_available"])
            self.assertFalse(summary["hidden_classifier_gpu_gate_passes"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["decision"], "transformer_acsr_seed_repeat_local_only_gpu_blocked")
            self.assertEqual(
                summary["selected_next_step"],
                "run_hidden_support_classifier_sequence_ood_budget_audit_before_gpu",
            )
            self.assertTrue((Path(tmp) / "seed_rows.csv").exists())
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "notes.md").exists())
            seed_rows = (Path(tmp) / "seed_rows.csv").read_text(encoding="utf-8")
            self.assertIn("direct_hidden_support_classifier_gate_passes", seed_rows)
            self.assertIn("direct_hidden_support_classifier_ce_gain_vs_frequency_null", seed_rows)
            self.assertIn(
                "direct_hidden_support_classifier_ce_gain_vs_learned_router",
                seed_rows,
            )
            self.assertIn(
                "direct_hidden_support_classifier_oracle_regret_recovery_vs_learned_router",
                seed_rows,
            )
            self.assertIn(
                "direct_hidden_support_classifier_rule_ood_evidence_available",
                seed_rows,
            )
            self.assertIn(
                "direct_hidden_support_classifier_churn_budget_evidence_available",
                seed_rows,
            )
            self.assertIn(
                "direct_hidden_support_classifier_commutator_budget_evidence_available",
                seed_rows,
            )
            notes = (Path(tmp) / "notes.md").read_text(encoding="utf-8")
            self.assertIn("learned-router comparison available: `True`", notes)
            self.assertIn("learned-router gate passes: `False`", notes)
            self.assertIn("sequence-heldout gate passes: `False`", notes)
            self.assertIn("rule-OOD evidence available: `False`", notes)
            self.assertIn("rule-OOD gate passes: `False`", notes)
            self.assertIn("churn budget evidence available: `False`", notes)
            self.assertIn("churn budget gate passes: `False`", notes)
            self.assertIn("commutator budget evidence available: `False`", notes)
            self.assertIn("commutator budget gate passes: `False`", notes)
            self.assertIn("Hidden support-classifier evidence is pre-GPU only", notes)


if __name__ == "__main__":
    unittest.main()
