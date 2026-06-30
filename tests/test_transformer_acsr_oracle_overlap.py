import unittest
import csv
import tempfile
from pathlib import Path

from relaleap.experiments.transformer_acsr_oracle_overlap import (
    build_soft_oracle_support_targets,
    write_oracle_overlap_training_pregate,
)


class TransformerACSROracleOverlapTests(unittest.TestCase):
    def test_lower_loss_support_receives_higher_target_probability(self) -> None:
        rows = [
            {
                "batch_index": 0,
                "position_index": 1,
                "token_index": 7,
                "target_token": 3,
                "support": "[0, 2]",
                "support_loss": 1.40,
            },
            {
                "batch_index": 0,
                "position_index": 1,
                "token_index": 7,
                "target_token": 3,
                "support": "[1, 4]",
                "support_loss": 1.10,
            },
            {
                "batch_index": 0,
                "position_index": 1,
                "token_index": 7,
                "target_token": 3,
                "support": "[3, 5]",
                "support_loss": 1.25,
            },
        ]

        targets = build_soft_oracle_support_targets(rows, temperature=0.2)
        by_support = {row["support"]: row for row in targets}

        best = by_support["[1, 4]"]
        middle = by_support["[3, 5]"]
        worst = by_support["[0, 2]"]
        self.assertEqual(best["oracle_rank"], 1)
        self.assertEqual(best["oracle_regret"], 0.0)
        self.assertGreater(best["target_probability"], middle["target_probability"])
        self.assertGreater(middle["target_probability"], worst["target_probability"])
        self.assertAlmostEqual(
            sum(row["target_probability"] for row in targets),
            1.0,
            places=12,
        )

    def test_contexts_are_normalized_independently(self) -> None:
        rows = [
            {"batch_index": 0, "position_index": 0, "support": "0,1", "loss": 2.0},
            {"batch_index": 0, "position_index": 0, "support": "1,2", "loss": 1.0},
            {"batch_index": 0, "position_index": 1, "support": "0,1", "loss": 1.0},
            {"batch_index": 0, "position_index": 1, "support": "1,2", "loss": 2.0},
        ]

        targets = build_soft_oracle_support_targets(rows, temperature=0.5)
        context0 = [
            row
            for row in targets
            if row.get("batch_index") == 0 and row.get("position_index") == 0
        ]
        context1 = [
            row
            for row in targets
            if row.get("batch_index") == 0 and row.get("position_index") == 1
        ]

        self.assertAlmostEqual(sum(row["target_probability"] for row in context0), 1.0)
        self.assertAlmostEqual(sum(row["target_probability"] for row in context1), 1.0)
        self.assertEqual(context0[0]["support"], "[1, 2]")
        self.assertEqual(context1[0]["support"], "[0, 1]")

    def test_training_pregate_writes_fail_closed_prefix_safe_artifacts(self) -> None:
        rows = [
            {
                "arm": "promoted_contextual_topk2",
                "split": "holdout",
                "episode_index": episode,
                "position_index": position,
                "latent_rule": "copy_shift",
                "target_token": (episode + position) % 5,
                "learned_support": "2,5",
                "learned_ce_loss": 2.4,
                "best_singleton_support": "1",
                "best_singleton_ce_loss": 2.2,
                "best_pair_support": "1,3",
                "best_pair_ce_loss": 2.1,
                "oracle_support": "1,3" if position % 2 == 0 else "1",
                "oracle_support_size": 2 if position % 2 == 0 else 1,
                "oracle_ce_loss": 2.1 if position % 2 == 0 else 2.2,
                "oracle_regret": 0.3,
                "best_one_swap_support": "1,5",
                "best_one_swap_ce_loss": 2.3,
            }
            for episode in range(4)
            for position in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "oracle_rows.csv"
            keys = list(rows[0])
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            summary = write_oracle_overlap_training_pregate(
                source_csv=source,
                out_dir=Path(tmp) / "pregate",
                training_steps=2,
                seed=3,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreater(summary["heldout_context_count"], 0)
            primary = summary["primary_result"]
            self.assertFalse(primary["uses_target_token_as_predictor_feature"])
            self.assertFalse(primary["uses_oracle_loss_as_predictor_feature"])
            self.assertNotIn("target_token", primary["prefix_safe_feature_names"])
            self.assertTrue((Path(tmp) / "pregate" / "oracle_overlap_training_pregate.csv").exists())
            self.assertTrue((Path(tmp) / "pregate" / "summary.json").exists())
            self.assertTrue((Path(tmp) / "pregate" / "notes.md").exists())


if __name__ == "__main__":
    unittest.main()
