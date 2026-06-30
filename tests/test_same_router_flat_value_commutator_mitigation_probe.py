from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_commutator_mitigation_probe import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_same_router_flat_value_commutator_mitigation_probe,
)


class SameRouterFlatValueCommutatorMitigationProbeTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_commutator_mitigation_probe(
                design_path=root / "missing_design.json",
                diagnostic_dir=root / "missing_diagnostic",
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

    def test_existing_anchor_proxy_blocks_gpu_when_commutator_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostic = root / "diagnostic"
            synthetic = root / "synthetic"
            diagnostic.mkdir()
            synthetic.mkdir()
            _write_json(
                root / "design.json",
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_commutator_mitigation_design_recorded",
                    "claim_status": "design_only_flat_value_commutator_mitigation_not_yet_evidence",
                    "selected_next_action": "implement_flat_value_commutator_mitigation_probe_locally",
                },
            )
            (diagnostic / "budget_rows.csv").write_text(
                "\n".join(
                    [
                        "budget,reference_budget_value",
                        "residual_norm,0.09",
                        "functional_churn,0.00005",
                        "finite_update_commutator,0.00000003",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (diagnostic / "control_rows.csv").write_text(
                "\n".join(
                    [
                        "arm,flat_ce_gain_vs_control",
                        "promoted_contextual_topk2,0.02",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (diagnostic / "gate_rows.csv").write_text(
                "gate,passes\nflat_beats_promoted_sparse,True\n",
                encoding="utf-8",
            )
            (synthetic / "arm_metrics.csv").write_text(
                "\n".join(
                    [
                        "arm,holdout_ce,residual_l2",
                        "promoted_contextual_topk2,2.71,0.076",
                        "token_position_router_topk2,2.72,0.087",
                        "dense_rank_norm_matched,2.74,0.081",
                        "low_churn_mlp_active_matched,2.73,0.12",
                        "fixed_support_topk2,2.73,0.082",
                        "random_support_topk2,2.76,0.086",
                        "flat_column_value_mlp_topk2,2.686,0.074",
                        "flat_column_value_mlp_anchor_topk2,2.678,0.075",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "commutator_rows.csv").write_text(
                "\n".join(
                    [
                        "arm,finite_update_commutator_l2",
                        "flat_column_value_mlp_topk2,0.000000056",
                        "flat_column_value_mlp_anchor_topk2,0.000000048",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (synthetic / "forgetting_rows.csv").write_text(
                "\n".join(
                    [
                        "arm,functional_churn",
                        "flat_column_value_mlp_topk2,0.000028",
                        "flat_column_value_mlp_anchor_topk2,0.000026",
                    ]
                )
                + "\n",
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

            summary = run_same_router_flat_value_commutator_mitigation_probe(
                design_path=root / "design.json",
                diagnostic_dir=diagnostic,
                synthetic_dir=synthetic,
                strategy_review_path=root / "latest-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "same_router_flat_value_commutator_mitigation_probe_gpu_blocked",
            )
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "flat_value_commutator_mitigation_not_established",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["measured_passing_variant_count"], 0)
            self.assertEqual(summary["missing_required_variant_count"], 2)
            anchor = next(
                row
                for row in summary["variant_rows"]
                if row["variant"] == "flat_value_commutator_penalty_probe"
            )
            self.assertFalse(anchor["commutator_budget_ok"])
            self.assertIn("finite_update_commutator_budget_failed", anchor["failure_reasons"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
