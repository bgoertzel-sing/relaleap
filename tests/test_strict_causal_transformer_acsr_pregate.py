from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.strict_causal_transformer_acsr_pregate import (
    FeatureContract,
    REQUIRED_ARTIFACTS,
    run_strict_causal_transformer_acsr_pregate,
)


class StrictCausalTransformerACSRPregateTests(unittest.TestCase):
    def test_feature_contract_rejects_future_oracle_and_task_fields(self) -> None:
        contract = FeatureContract()
        contract.assert_predictor_fields_allowed(
            ["current_hidden_json", "previous_hidden_json", "position_index"]
        )

        for field in (
            "next_hidden",
            "next_delta",
            "future_hidden_json_target_only",
            "teacher_support_logits_json_target_only",
            "oracle_support_eval_only",
            "target_token_eval_only",
            "task_id",
            "forced_support_loss",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    contract.assert_predictor_fields_allowed(["current_hidden_json", field])

    def test_records_existing_transformer_acsr_closeout_and_defers_duplicate_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_specs = []
            payloads = {
                "transformer_acsr_hidden_future_sequence_dataset": {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_future_sequence_dataset_trainable_locally",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
                "transformer_acsr_hidden_future_predictor_pregate": {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_future_predictor_pregate_gpu_blocked",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
                "transformer_acsr_hidden_future_control_audit": {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_future_control_audit_gpu_blocked",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
                "transformer_acsr_hidden_future_support_value_headroom": {
                    "status": "pass",
                    "decision": "support_value_headroom_negligible_close_teacher_imitation_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
                "transformer_acsr_hidden_future_support_value_closeout": {
                    "status": "pass",
                    "decision": "transformer_acsr_teacher_support_imitation_closed_before_gpu",
                    "selected_next_action": "close_transformer_acsr_teacher_support_imitation_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
            }
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                _write_json(path, payload)
                source_specs.append((name, path, True))
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Implement strict causal Transformer-ACSR.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_strict_causal_transformer_acsr_pregate(
                source_paths=tuple(source_specs),
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "strict_causal_transformer_acsr_existing_branch_closed_no_gpu",
            )
            self.assertTrue(summary["existing_transformer_acsr_branch_closed"])
            self.assertTrue(summary["new_training_deferred"])
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertTrue(summary["direction_shift_recorded"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(
                summary["deferred_or_rejected_review_recommendations"][0]["disposition"],
                "partially_accepted_contract_audit_only",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_required_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"

            summary = run_strict_causal_transformer_acsr_pregate(
                source_paths=(("transformer_acsr_hidden_future_sequence_dataset", missing, True),),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "strict_causal_transformer_acsr_pregate_failed_closed")
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
