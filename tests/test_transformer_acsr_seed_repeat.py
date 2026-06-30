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
            self.assertIn("mean_hidden_classifier_support_overlap_with_oracle", summary)
            self.assertIn("robust_hidden_classifier_gate_passes", summary)
            self.assertIn("hidden_classifier_null_margin_gate_passes", summary)
            self.assertFalse(summary["hidden_classifier_learned_router_comparison_available"])
            self.assertFalse(summary["hidden_classifier_sequence_ood_budget_audit_available"])
            self.assertFalse(summary["hidden_classifier_gpu_gate_passes"])
            self.assertIn(
                summary["decision"],
                {
                    "transformer_acsr_seed_repeat_passed_gpu_validation_ready",
                    "transformer_acsr_seed_repeat_local_only_gpu_blocked",
                },
            )
            self.assertTrue((Path(tmp) / "seed_rows.csv").exists())
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "notes.md").exists())
            self.assertIn("selected_next_step", summary)
            seed_rows = (Path(tmp) / "seed_rows.csv").read_text(encoding="utf-8")
            self.assertIn("direct_hidden_support_classifier_gate_passes", seed_rows)
            self.assertIn("direct_hidden_support_classifier_ce_gain_vs_frequency_null", seed_rows)


if __name__ == "__main__":
    unittest.main()
