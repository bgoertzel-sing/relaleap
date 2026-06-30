import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.hidden_support_classifier_sequence_ood_budget_audit import (
    run_hidden_support_classifier_sequence_ood_budget_audit,
)


class HiddenSupportClassifierSequenceOodBudgetAuditTests(unittest.TestCase):
    def test_audit_writes_fail_closed_sequence_ood_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = run_hidden_support_classifier_sequence_ood_budget_audit(
                out_dir=out_dir,
                seeds=(17,),
                training_steps=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["rule_combo_heldout_gate_passes"])
            self.assertFalse(summary["budget_gate_passes"])
            self.assertFalse(summary["residual_norm_budget_gate_passes"])
            self.assertFalse(summary["functional_churn_budget_gate_passes"])
            self.assertFalse(summary["commutator_budget_gate_passes"])
            self.assertIn("mean_hidden_classifier_ce_gain_vs_learned_router", summary)
            self.assertIn("mean_oracle_regret_recovery_vs_learned_router", summary)
            self.assertEqual(
                summary["selected_next_step"],
                "add_rule_combo_heldout_and_exact_budget_rows_or_close_hidden_classifier_branch",
            )

            for artifact in ("audit_rows.csv", "budget_rows.csv", "summary.json", "notes.md"):
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            with (out_dir / "audit_rows.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in rows}, {"sequence_heldout", "rule_combo_heldout"})
            rule_rows = [row for row in rows if row["split"] == "rule_combo_heldout"]
            self.assertEqual({row["evidence_measured"] for row in rule_rows}, {"False"})
            self.assertIn("hidden_classifier_ce_gain_vs_learned_router", rows[0])

            notes = (out_dir / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Rule-combo-heldout gate passes: `False`", notes)
            self.assertIn("GPU validation remains blocked", notes)


if __name__ == "__main__":
    unittest.main()
