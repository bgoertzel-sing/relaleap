from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.learned_router_sparse_value_closeout import (
    FLAT_VALUE_DIAGNOSTIC_ACTION,
    REPAIR_SOURCES_ACTION,
    REQUIRED_ARTIFACTS,
    run_learned_router_sparse_value_closeout,
)


class LearnedRouterSparseValueCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_learned_router_sparse_value_closeout(
                pregate_path=root / "missing_summary.json",
                pregate_rows_path=root / "missing_rows.csv",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_SOURCES_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_blocked_sparse_value_branch_redirects_to_flat_value_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pregate = root / "pregate.json"
            pregate_rows = root / "pregate_rows.csv"
            review = root / "latest-review.md"
            _write_json(
                pregate,
                {
                    "status": "pass",
                    "decision": "learned_router_sparse_value_pregate_local_gpu_blocked",
                    "claim_status": "learned_router_sparse_value_blocked_by_null_or_interference_controls",
                    "selected_next_action": "close_or_redirect_learned_router_sparse_value_branch_locally",
                    "pregate_primary_result": {
                        "primary_arm": "promoted_contextual_topk2",
                        "primary_holdout_ce": 2.70,
                        "token_position_ce_gain": 0.001,
                        "flat_control_ce_gain": -0.024,
                        "flat_control_ok": False,
                        "commutator_budget_ok": False,
                        "functional_churn_budget_ok": False,
                        "stored_upper_bound_blocks_promotion": True,
                        "pregate_passes": False,
                        "failure_reasons": "token_position_null_too_close_or_stronger;same_router_flat_value_control_stronger",
                    },
                },
            )
            pregate_rows.write_text(
                "\n".join(
                    [
                        "primary_arm,selected,pregate_passes,flat_control_ce_gain,failure_reasons",
                        "promoted_contextual_topk2,True,False,-0.024,same_router_flat_value_control_stronger",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "verdict: FIX",
                        "recommended_next_action: Do not launch RunPod yet",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_learned_router_sparse_value_closeout(
                pregate_path=pregate,
                pregate_rows_path=pregate_rows,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "learned_router_sparse_value_branch_closed")
            self.assertEqual(summary["selected_next_action"], FLAT_VALUE_DIAGNOSTIC_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "sparse_value_closed_flat_value_diagnostic_selected",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertIn("same_router_flat_value_control_stronger", summary["rationale"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("same-router flat-value-capacity diagnostic", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
