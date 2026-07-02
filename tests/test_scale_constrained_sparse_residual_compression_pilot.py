from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.scale_constrained_sparse_residual_compression_pilot import (
    ARMS,
    DECISION,
    OBJECTIVES,
    PRUNE_RULES,
    REQUIRED_ARTIFACTS,
    run_scale_constrained_sparse_residual_compression_pilot,
)


class ScaleConstrainedSparseResidualCompressionPilotTests(unittest.TestCase):
    def test_runs_local_pilot_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            review = root / "latest-review.md"
            _write_json(closeout, _closeout_payload())
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement a local executable scale-constrained sparse residual-compression pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_scale_constrained_sparse_residual_compression_pilot(
                closeout_path=closeout,
                strategy_review_path=review,
                out_dir=root / "out",
                seed=13,
                teacher_steps=2,
                value_steps=2,
                control_steps=2,
                atom_count=4,
                top_r=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DECISION)
            self.assertTrue(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(set(summary["objectives"]), set(OBJECTIVES))
            self.assertEqual(set(summary["arms"]), set(ARMS))
            self.assertEqual(set(summary["prune_rules"]), set(PRUNE_RULES))
            self.assertIn("same norm budget", summary["norm_controller_parity"])
            arm_keys = {(row["objective"], row["arm"]) for row in summary["arm_metrics"]}
            self.assertIn(("ce_only", "sparse_topr_norm_controller"), arm_keys)
            self.assertIn(("ce_only", "same_controller_flat_residual"), arm_keys)
            self.assertIn(("mse_only", "shuffled_target_sparse_null"), arm_keys)
            self.assertGreaterEqual(len(summary["arm_metrics"]), len(OBJECTIVES) * len(ARMS))
            self.assertGreaterEqual(len(summary["pruning_rows"]), len(OBJECTIVES) * len(PRUNE_RULES))
            gates = {row["criterion"]: row for row in summary["gate_rows"]}
            self.assertTrue(gates["closeout_selected_scale_constrained_path"]["passed"])
            self.assertTrue(gates["gpu_blocked"]["passed"])
            self.assertEqual(gates["sparse_matches_or_beats_flat_ce"]["gate_type"], "scientific")
            self.assertIn("Accepted the GPT-5.5-Pro recommendation", summary["strategy_review_handling"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_when_closeout_did_not_select_scale_constrained_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = _closeout_payload()
            payload["selected_next_action"] = "different_action"
            _write_json(root / "closeout.json", payload)

            summary = run_scale_constrained_sparse_residual_compression_pilot(
                closeout_path=root / "closeout.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
                seed=14,
                teacher_steps=2,
                value_steps=2,
                control_steps=2,
                atom_count=4,
                top_r=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["claim_status"], "scale_constrained_sparse_pilot_runtime_failed")
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("closeout_selected_scale_constrained_path", failed)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _closeout_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "prunable_soft_mixture_branch_closed_no_gpu",
        "claim_status": "prunable_soft_mixture_retired_before_gpu",
        "selected_next_action": "design_scale_constrained_sparse_residual_compression_pregate",
        "selected_next_step": "design a local scale-constrained sparse residual-compression pregate with flat/dense controls before GPU",
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "training_executed": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
