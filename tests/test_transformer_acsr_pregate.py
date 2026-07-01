from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_pregate import (
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_pregate,
)


class TransformerACSRPregateTests(unittest.TestCase):
    def test_pregate_labels_sources_and_blocks_without_tensor_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_specs = []
            for name in (
                "anticipatory_contextual_support_routing_design",
                "causal_contextual_router_gate",
                "causal_contextual_router_distillation_synthesis",
                "transformer_acsr_seed_repeat",
                "transformer_acsr_hidden_feature_redesign_gate",
            ):
                path = root / f"{name}.json"
                _write_json(
                    path,
                    {
                        "status": "pass",
                        "decision": f"{name}_decision",
                        "claim_status": f"{name}_claim",
                        "requires_gpu_now": False,
                        "promotion_allowed": False,
                        "advance_to_gpu_validation": False,
                    },
                )
                source_specs.append((name, path, True))
            strategy = root / "latest-review.md"
            strategy.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Start a bounded local Transformer-ACSR pregate.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_pregate(
                source_paths=tuple(source_specs),
                strategy_review_path=strategy,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_pregate_inventory_recorded_gpu_blocked",
            )
            self.assertEqual(
                summary["claim_status"],
                "prefix_safe_transformer_acsr_training_data_incomplete_no_gpu",
            )
            self.assertFalse(summary["trainable_now"])
            self.assertFalse(summary["sequence_split_tensor_dataset_available"])
            self.assertTrue(summary["same_student_intervention_rows_available"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreaterEqual(summary["feature_provenance"]["prefix_safe_count"], 4)
            self.assertGreaterEqual(
                summary["feature_provenance"]["future_or_target_leaking_count"],
                3,
            )
            self.assertIn(
                "sequence_split_prefix_feature_tensor_dataset",
                summary["missing_requirements"],
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_pregate_fails_closed_when_required_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"

            summary = run_transformer_acsr_pregate(
                source_paths=(("anticipatory_contextual_support_routing_design", missing, True),),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "transformer_acsr_pregate_failed_closed")
            self.assertEqual(summary["claim_status"], "required_transformer_acsr_sources_missing")
            self.assertEqual(len(summary["failures"]), 1)
            self.assertFalse(summary["requires_gpu_now"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
