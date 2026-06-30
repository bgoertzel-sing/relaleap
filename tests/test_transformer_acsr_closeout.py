import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_closeout import run_transformer_acsr_closeout


class TransformerACSRCloseoutTests(unittest.TestCase):
    def test_closeout_writes_redesign_required_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            out = Path(tmp) / "out"
            source.mkdir()
            (source / "summary.json").write_text(
                json.dumps(
                    {
                        "decision": "transformer_acsr_seed_repeat_local_only_gpu_blocked",
                        "seed_count": 3,
                        "completed_seed_count": 3,
                        "value_aware_gate_pass_count": 1,
                        "leakage_pass_count": 3,
                        "support_intervention_assay_valid_count": 3,
                        "mean_value_aware_ce_gain_vs_token_position_support": 0.0035,
                        "mean_value_aware_support_overlap_with_oracle": 0.1354,
                        "robust_value_gate_passes": False,
                        "oracle_overlap_gate_passes": False,
                        "advance_to_gpu_validation": False,
                    }
                ),
                encoding="utf-8",
            )

            summary = run_transformer_acsr_closeout(source_dir=source, out_dir=out)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"], "transformer_acsr_closeout_local_redesign_required"
            )
            self.assertEqual(
                summary["selected_next_step"],
                "design_oracle_overlap_aware_transformer_acsr_support_objective",
            )
            self.assertTrue(summary["branch_closed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertIn("value_aware_gate_not_robust_across_seeds", summary["failure_reasons"])
            self.assertTrue((out / "transformer_acsr_closeout.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "notes.md").exists())


if __name__ == "__main__":
    unittest.main()
