from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.regret_soft_utility_head_probe import (
    IMPLEMENT_DIRECT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_regret_soft_utility_head_probe,
)


class RegretSoftUtilityHeadProbeTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_regret_soft_utility_head_probe(
                design_path=root / "missing_design.json",
                hidden_audit_dir=root / "missing_hidden",
                synthetic_dir=root / "missing_synthetic",
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

    def test_proxy_rows_do_not_authorize_gpu_without_direct_utility_head_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden"
            synthetic = root / "synthetic"
            hidden.mkdir()
            synthetic.mkdir()
            _write_json(
                root / "design.json",
                {
                    "status": "pass",
                    "decision": "regret_soft_utility_head_design_recorded",
                    "claim_status": "design_only_regret_soft_utility_head_not_yet_evidence",
                    "selected_next_action": "implement_regret_soft_utility_head_probe_locally",
                },
            )
            _write_json(
                hidden / "summary.json",
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "mean_hidden_classifier_ce_gain_vs_learned_router": -0.03,
                    "mean_oracle_regret_recovery_vs_learned_router": -0.9,
                },
            )
            (hidden / "audit_rows.csv").write_text(
                "split,diagnostic,gate_passes\nsequence_heldout,direct_hidden_support_classifier,False\n",
                encoding="utf-8",
            )
            (hidden / "budget_rows.csv").write_text(
                "budget,gate_passes\nresidual_norm,False\nfunctional_churn,False\nfinite_update_commutator,False\n",
                encoding="utf-8",
            )
            (synthetic / "support_head_sequence_heldout_diagnostic.csv").write_text(
                "\n".join(
                    [
                        "arm,diagnostic,split,target_source,uses_hidden_features,uses_token_position_features,uses_shuffled_targets,deployable_training_evidence,learned_router_ce,oracle_pair_ce_ceiling,predicted_support_ce,support_accuracy_vs_oracle_pair,support_change_fraction,residual_l2,beats_shuffled_target_null,beats_token_position_null",
                        "promoted_contextual_topk2,support_regret_trained_contextual_router_topk2,sequence_heldout,train_split_oracle_pair_supports,True,False,False,False,2.70,2.65,2.72,0.55,1.0,0.07,True,False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "arm_metrics.csv").write_text(
                "arm,holdout_ce,residual_l2\npromoted_contextual_topk2,2.70,0.07\n",
                encoding="utf-8",
            )
            (synthetic / "forgetting_rows.csv").write_text(
                "arm,functional_churn\npromoted_contextual_topk2,0.00003\n",
                encoding="utf-8",
            )
            (synthetic / "commutator_rows.csv").write_text(
                "arm,finite_update_commutator_l2\npromoted_contextual_topk2,0.00000003\n",
                encoding="utf-8",
            )
            (root / "latest-review.md").write_text(
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

            summary = run_regret_soft_utility_head_probe(
                design_path=root / "design.json",
                hidden_audit_dir=hidden,
                synthetic_dir=synthetic,
                strategy_review_path=root / "latest-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "regret_soft_utility_head_probe_direct_rows_missing")
            self.assertEqual(summary["selected_next_action"], IMPLEMENT_DIRECT_ACTION)
            self.assertEqual(summary["direct_regret_soft_row_count"], 0)
            self.assertEqual(summary["proxy_row_count"], 1)
            self.assertFalse(summary["proxy_candidate_passes"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            direct_gate = next(
                row for row in summary["gate_rows"] if row["gate"] == "direct_regret_soft_rows_present"
            )
            self.assertFalse(direct_gate["passes"])
            proxy = next(
                row
                for row in summary["candidate_rows"]
                if row["candidate"] == "support_regret_trained_contextual_router_proxy"
            )
            self.assertIn("proxy_not_direct", proxy["failure_reasons"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
