from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_ceiling_closeout import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    VALUE_DICTIONARY_RESCUE_ACTION,
    run_low_churn_mlp_sparse_factorization_ceiling_closeout,
)


class LowChurnMlpSparseFactorizationCeilingCloseoutTests(unittest.TestCase):
    def test_missing_audit_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_low_churn_mlp_sparse_factorization_ceiling_closeout(
                decision_audit_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_weak_global_dictionary_closes_current_ceiling_and_selects_value_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit = root / "audit.json"
            review = root / "latest-review.md"
            _write_json(
                audit,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_sparse_factorization_decision_audit_recorded",
                    "claim_status": "sparse_factorization_proxy_artifact_and_deployable_gap_block_gpu",
                    "selected_next_action": "redesign_value_dictionary_or_close_sparse_factorization_ceiling",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                    "exact_oracle_nondeployable": True,
                    "learned_router_blocks_gpu": True,
                    "learned_router_heldout_r2": -0.16,
                    "oracle_learned_r2_gap": 1.16,
                    "global_dictionary_oracle_r2": 0.41,
                    "blocking_blame": [
                        {"blame_category": "proxy_artifact_failure", "disposition": "blocking"},
                        {
                            "blame_category": "value_dictionary_capacity_or_target_noncolumnability",
                            "disposition": "blocking",
                        },
                    ],
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Add a local sparse-factorization decision/blame audit before any GPU tuning.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_low_churn_mlp_sparse_factorization_ceiling_closeout(
                decision_audit_path=audit,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], VALUE_DICTIONARY_RESCUE_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], VALUE_DICTIONARY_RESCUE_ACTION)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("vector-centroid reusable-dictionary path", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
