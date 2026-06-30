import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_null_gate_repeat import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_null_gate_repeat,
)


class TransformerAcsrHiddenNullGateRepeatTests(unittest.TestCase):
    def test_repeat_report_blocks_gpu_when_learned_router_and_ood_budgets_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_repeat = root / "seed_repeat.json"
            hidden_gate = root / "hidden_gate.json"
            sequence_audit = root / "sequence_audit.json"
            _write_json(
                seed_repeat,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_seed_repeat_local_only_gpu_blocked",
                    "seed_count": 3,
                    "hidden_classifier_null_margin_gate_passes": True,
                    "hidden_classifier_learned_router_gate_passes": False,
                    "hidden_classifier_churn_budget_gate_passes": False,
                    "hidden_classifier_commutator_budget_gate_passes": False,
                    "hidden_classifier_leakage_pass_count": 3,
                    "mean_hidden_classifier_ce_gain_vs_token_position_null": 0.01,
                    "mean_hidden_classifier_ce_gain_vs_shuffled_null": 0.02,
                    "mean_hidden_classifier_ce_gain_vs_frequency_null": 0.03,
                    "hidden_classifier_gpu_gate_required_fields": {
                        "weak_null_margins": "pass",
                        "learned_router_comparison": "fail",
                        "sequence_heldout": "fail",
                        "rule_ood": "missing",
                        "churn_budget": "missing",
                        "commutator_budget": "missing",
                    },
                },
            )
            _write_json(
                hidden_gate,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_feature_redesign_gate_gpu_blocked",
                    "null_gate_passes": True,
                    "learned_router_gate_passes": False,
                    "future_perturbation_leakage_gate_passes": True,
                    "aggregates": {
                        "mean_ce_gain_vs_learned_router": -0.04,
                        "mean_oracle_regret_recovery_vs_learned_router": -0.5,
                        "max_future_perturbation_prefix_delta": 0.0,
                    },
                },
            )
            _write_json(
                sequence_audit,
                {
                    "status": "pass",
                    "decision": "hidden_support_classifier_sequence_ood_budget_audit_gpu_blocked",
                    "sequence_heldout_gate_passes": False,
                    "rule_combo_heldout_gate_passes": False,
                    "rule_combo_evidence_available": False,
                    "budget_gate_passes": False,
                },
            )

            summary = run_transformer_acsr_hidden_null_gate_repeat(
                seed_repeat_path=seed_repeat,
                hidden_feature_gate_path=hidden_gate,
                sequence_audit_path=sequence_audit,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_null_gate_repeat_gpu_blocked",
            )
            self.assertTrue(summary["weak_null_gate_passes"])
            self.assertTrue(summary["leakage_gate_passes"])
            self.assertFalse(summary["learned_router_gate_passes"])
            self.assertFalse(summary["sequence_heldout_gate_passes"])
            self.assertFalse(summary["rule_ood_gate_passes"])
            self.assertFalse(summary["budget_gate_passes"])
            self.assertFalse(summary["repeat_gate_passes"])
            self.assertFalse(summary["advance_to_gpu_validation"])
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
