from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.hidden_support_classifier_branch_selector import (
    REQUIRED_ARTIFACTS,
    RETURN_LEARNED_ROUTER_ACTION,
    run_hidden_support_classifier_branch_selector,
)


class HiddenSupportClassifierBranchSelectorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_hidden_support_classifier_branch_selector(
                hidden_audit_path=root / "missing_hidden_audit.json",
                closeout_rows_path=root / "missing_closeout.csv",
                seed_repeat_path=root / "missing_seed_repeat.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["selected_next_action"],
                "repair_hidden_support_classifier_branch_selector_sources",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_closed_hidden_classifier_returns_to_learned_router_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden_audit = root / "hidden_audit.json"
            closeout = root / "closeout_rows.csv"
            seed_repeat = root / "seed_repeat.json"
            review = root / "latest-review.md"
            _write_json(
                hidden_audit,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "close_hidden_classifier_branch": True,
                    "closeout_status": "closed_hidden_support_classifier_branch_before_gpu",
                    "sequence_heldout_gate_passes": False,
                    "rule_combo_heldout_gate_passes": False,
                    "budget_gate_passes": False,
                    "mean_hidden_classifier_ce_gain_vs_learned_router": -0.03,
                    "mean_oracle_regret_recovery_vs_learned_router": -0.9,
                },
            )
            closeout.write_text(
                "\n".join(
                    [
                        "branch,status,next_step,deferred_exact_row_reason",
                        "direct_hidden_support_classifier,closed_hidden_support_classifier_branch_before_gpu,close_or_redesign_hidden_support_classifier_branch_before_gpu,sequence-heldout same-student intervention rows already lose to the learned router",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                seed_repeat,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_seed_repeat_local_only_gpu_blocked",
                    "hidden_classifier_gate_pass_count": 3,
                    "hidden_classifier_null_margin_gate_passes": True,
                    "value_aware_gate_pass_count": 1,
                    "advance_to_gpu_validation": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet; repair hidden-classifier gate/report inconsistency.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_hidden_support_classifier_branch_selector(
                hidden_audit_path=hidden_audit,
                closeout_rows_path=closeout,
                seed_repeat_path=seed_repeat,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "hidden_support_classifier_branch_selected")
            self.assertEqual(summary["selected_next_action"], RETURN_LEARNED_ROUTER_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertIn("learned-router/non-PC", summary["selected_next_step"])
            actions = {row["candidate_action"]: row for row in summary["candidate_actions"]}
            self.assertEqual(actions[RETURN_LEARNED_ROUTER_ACTION]["disposition"], "selected")
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("direct_hidden_classifier_closed", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
