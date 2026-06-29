from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.synthetic_mechanism_causal_modularity import (
    REQUIRED_ARTIFACTS,
    run_synthetic_mechanism_causal_modularity,
)


class SyntheticMechanismCausalModularityTest(unittest.TestCase):
    def test_generates_hidden_boundary_packet_and_fails_closed_without_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=out_dir,
                seed=3,
                vocab_size=12,
                seq_len=6,
                train_episodes_per_rule=2,
                holdout_episodes_per_rule=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_pregate_failed_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["notify_ben"])
            self.assertEqual(summary["strategic_change_level"], "major")
            self.assertFalse(summary["task_id_visible_to_model"])
            self.assertFalse(summary["mechanism_labels_enter_training"])
            self.assertGreater(summary["episode_row_count"], 0)
            self.assertGreater(summary["per_mechanism_intervention_row_count"], 0)
            self.assertGreater(summary["commutator_row_count"], 0)
            self.assertGreater(summary["forgetting_row_count"], 0)
            self.assertTrue(summary["missing_training_hooks"])
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("training_hooks_available", failed)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            with (out_dir / "episode_rows.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["task_id_visible_to_model"] for row in rows}, {"False"})
            self.assertEqual({row["mechanism_label_enters_training"] for row in rows}, {"False"})
            self.assertEqual({row["shared_vocab_id_space"] for row in rows}, {"True"})
            self.assertEqual(
                {row["latent_rule"] for row in rows},
                {"copy_shift", "reverse_window", "xor_prev", "affine_jump"},
            )
            self.assertIn("True", {row["mechanism_boundary_hidden"] for row in rows})

            with (out_dir / "comparator_controls.csv").open(newline="", encoding="utf-8") as handle:
                controls = list(csv.DictReader(handle))
            arms = {row["arm"] for row in controls}
            self.assertIn("promoted_contextual_topk2", arms)
            self.assertIn("intervention_trained_sparse_topk2", arms)
            self.assertIn("dense_rank_norm_matched", arms)
            self.assertIn("low_churn_mlp_control", arms)
            self.assertIn("token_position_router_topk2", arms)

    def test_schema_can_pass_when_training_hooks_are_declared_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=Path(tmpdir) / "out",
                training_hooks_available=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_pregate_ready")
            self.assertFalse(summary["missing_training_hooks"])
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(criteria["training_hooks_available"]["passed"])


if __name__ == "__main__":
    unittest.main()
