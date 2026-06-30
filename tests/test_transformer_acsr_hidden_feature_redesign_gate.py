from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_feature_redesign_gate import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_feature_redesign_gate,
)


class TransformerAcsrHiddenFeatureRedesignGateTests(unittest.TestCase):
    def test_hidden_feature_gate_replaces_proxy_and_blocks_when_learned_router_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closeout = root / "closeout.json"
            hidden_audit = root / "hidden_audit.json"
            oracle_pregate = root / "oracle_pregate.json"
            seed_root = root / "seeds"
            seed_dir = seed_root / "seed_17"
            seed_dir.mkdir(parents=True)
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_closed_redirect_selected",
                    "selected_next_action": "select_oracle_overlap_aware_transformer_acsr_support_objective_redesign",
                },
            )
            _write_json(
                hidden_audit,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "close_hidden_classifier_branch": True,
                    "mean_hidden_classifier_ce_gain_vs_learned_router": -0.03,
                },
            )
            _write_json(
                oracle_pregate,
                {
                    "status": "pass",
                    "decision": "oracle_overlap_transformer_acsr_training_pregate_gpu_blocked",
                    "source_format": "oracle_support_summary_rows",
                },
            )
            (seed_dir / "transformer_acsr_cpu_smoke_pilot.csv").write_text(
                "\n".join(
                    [
                        "row_role,direct_hidden_support_classifier_ce,direct_hidden_support_classifier_ce_gain_vs_token_position_null,direct_hidden_support_classifier_ce_gain_vs_shuffled_null,direct_hidden_support_classifier_ce_gain_vs_frequency_null,direct_hidden_support_classifier_overlap_with_oracle,direct_hidden_support_classifier_exact_match_with_oracle,direct_hidden_support_classifier_future_perturbation_max_prefix_delta",
                        "primary_transformer_acsr_cpu_smoke_pilot,1.20,0.1,0.2,0.3,0.5,0.4,0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (seed_dir / "support_head_sequence_heldout_diagnostic.csv").write_text(
                "\n".join(
                    [
                        "arm,diagnostic,split,learned_router_ce,oracle_pair_ce_ceiling",
                        "promoted_contextual_topk2,support_regret_trained_contextual_router_topk2,sequence_heldout,1.10,1.00",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_hidden_feature_redesign_gate(
                closeout_redirect_path=closeout,
                hidden_audit_path=hidden_audit,
                seed_root=seed_root,
                oracle_pregate_path=oracle_pregate,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_feature_redesign_gate_gpu_blocked",
            )
            self.assertEqual(
                summary["claim_status"],
                "hidden_feature_same_student_gate_loses_to_learned_router",
            )
            self.assertTrue(summary["oracle_overlap_proxy_pregate_replaced"])
            self.assertTrue(summary["closeout_redirect_selected"])
            self.assertFalse(summary["learned_router_gate_passes"])
            self.assertTrue(summary["null_gate_passes"])
            self.assertTrue(summary["future_perturbation_leakage_gate_passes"])
            self.assertFalse(summary["hidden_feature_gate_passes"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertAlmostEqual(
                summary["aggregates"]["mean_ce_gain_vs_learned_router"],
                -0.1,
            )
            self.assertEqual(
                summary["selected_next_step"],
                "design_regret_soft_utility_head_with_margin_conditioned_learned_router_fallback",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
