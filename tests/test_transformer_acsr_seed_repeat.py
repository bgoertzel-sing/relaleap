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


if __name__ == "__main__":
    unittest.main()
