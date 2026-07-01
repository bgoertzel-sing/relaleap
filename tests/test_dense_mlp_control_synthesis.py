from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_mlp_control_synthesis import (
    ORTHOGONALIZED_SPARSE_PREGATE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_dense_mlp_control_synthesis,
)


class DenseMlpControlSynthesisTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_mlp_control_synthesis(
                pair_closeout_path=root / "missing_pair.json",
                context_core_closeout_path=root / "missing_context.json",
                regret_closeout_path=root / "missing_regret.json",
                value_dictionary_closeout_path=root / "missing_value.json",
                norm_churn_pilot_path=root / "missing_norm.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_dense_mlp_dominance_selects_sparse_interference_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair = root / "pair.json"
            context = root / "context.json"
            regret = root / "regret.json"
            value = root / "value.json"
            norm = root / "norm.json"
            review = root / "latest-review.md"
            _write_json(
                pair,
                {
                    "status": "pass",
                    "decision": "dense_teacher_pair_composer_branch_closed",
                    "claim_status": "pair_composer_closed_dense_mlp_controls_dominate_no_gpu",
                    "selected_next_action": "redirect_from_pair_composer_to_dense_mlp_control_synthesis",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                    "decision_matrix": [
                        {
                            "signal": "dense_mlp_controls_dominate_pair_composer",
                            "required": True,
                            "passed": True,
                            "actual": {
                                "oracle_pair_holdout_ce": 0.99,
                                "best_matched_control": "matched_mlp_random_feature_residual_control",
                                "best_matched_control_ce": 0.39,
                            },
                        }
                    ],
                },
            )
            for path, selected in [
                (context, "design_low_churn_mlp_sparse_factorization_ceiling"),
                (regret, "design_low_churn_mlp_residual_control_pregate"),
                (value, "select_next_post_value_dictionary_local_branch_before_gpu"),
            ]:
                _write_json(
                    path,
                    {
                        "status": "pass",
                        "decision": f"{path.stem}_closed",
                        "claim_status": "closed_no_gpu",
                        "selected_next_action": selected,
                        "requires_gpu_now": False,
                        "promotion_allowed": False,
                        "advance_to_gpu_validation": False,
                    },
                )
            _write_json(
                norm,
                {
                    "status": "pass",
                    "decision": "norm_budgeted_churn_regularized_residual_pilot_completed",
                    "claim_status": "local_budgeted_pilot_no_challenger_clears_dense24_gate",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "scientific_gate": "blocked",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Keep GPU blocked and implement dense/MLP-dominance synthesis.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_mlp_control_synthesis(
                pair_closeout_path=pair,
                context_core_closeout_path=context,
                regret_closeout_path=regret,
                value_dictionary_closeout_path=value,
                norm_churn_pilot_path=norm,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "dense_mlp_dominance_synthesized_sparse_interference_pregate_selected",
            )
            self.assertEqual(summary["selected_next_action"], ORTHOGONALIZED_SPARSE_PREGATE_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertTrue(summary["direction_shift_recorded"])
            self.assertEqual(len(summary["pregate_spec"]), 5)
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
