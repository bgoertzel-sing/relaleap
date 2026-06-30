from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.regret_soft_utility_head_design import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_ACTION,
    run_regret_soft_utility_head_design,
)


class RegretSoftUtilityHeadDesignTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_regret_soft_utility_head_design(
                hidden_gate_path=root / "missing_hidden_gate.json",
                hidden_audit_path=root / "missing_hidden_audit.json",
                hidden_closeout_path=root / "missing_hidden_closeout.json",
                seed_repeat_path=root / "missing_seed_repeat.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_design_when_hidden_classifier_is_closed_but_prefix_signal_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden_gate = root / "hidden_gate.json"
            hidden_audit = root / "hidden_audit.json"
            hidden_closeout = root / "hidden_closeout.json"
            seed_repeat = root / "seed_repeat.json"
            review = root / "latest-review.md"
            _write_json(
                hidden_gate,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_feature_redesign_gate_gpu_blocked",
                    "claim_status": "hidden_feature_same_student_gate_loses_to_learned_router",
                    "selected_next_step": "design_regret_soft_utility_head_with_margin_conditioned_learned_router_fallback",
                    "hidden_feature_gate_passes": False,
                    "learned_router_gate_passes": False,
                    "null_gate_passes": True,
                    "future_perturbation_leakage_gate_passes": True,
                    "aggregates": {
                        "mean_ce_gain_vs_learned_router": -0.03,
                        "mean_oracle_regret_recovery_vs_learned_router": -0.9,
                    },
                },
            )
            _write_json(
                hidden_audit,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "close_hidden_classifier_branch": True,
                    "closeout_status": "closed_hidden_support_classifier_branch_before_gpu",
                },
            )
            _write_json(
                hidden_closeout,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_closed_redirect_selected",
                    "hidden_branch_closed": True,
                    "selected_next_action": "select_oracle_overlap_aware_transformer_acsr_support_objective_redesign",
                },
            )
            _write_json(
                seed_repeat,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_seed_repeat_local_only_gpu_blocked",
                    "advance_to_gpu_validation": False,
                    "hidden_classifier_gpu_gate_passes": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Do not launch RunPod yet",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_regret_soft_utility_head_design(
                hidden_gate_path=hidden_gate,
                hidden_audit_path=hidden_audit,
                hidden_closeout_path=hidden_closeout,
                seed_repeat_path=seed_repeat,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "regret_soft_utility_head_design_recorded")
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "design_only_regret_soft_utility_head_not_yet_evidence",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertAlmostEqual(summary["evidence"]["mean_ce_gain_vs_learned_router"], -0.03)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            components = {row["component"] for row in summary["utility_head_design"]}
            self.assertIn("regret_soft_support_utility_head", components)
            self.assertIn("margin_conditioned_learned_router_fallback", components)
            self.assertIn("budget_and_mechanism_audit", components)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
