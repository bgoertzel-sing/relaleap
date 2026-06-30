import unittest

from relaleap.experiments.transformer_acsr_oracle_overlap import (
    build_soft_oracle_support_targets,
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


if __name__ == "__main__":
    unittest.main()
