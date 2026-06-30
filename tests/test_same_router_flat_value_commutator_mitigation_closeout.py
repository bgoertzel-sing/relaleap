from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.same_router_flat_value_commutator_mitigation_closeout import (
    CLOSE_GENERIC_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_same_router_flat_value_commutator_mitigation_closeout,
)


class SameRouterFlatValueCommutatorMitigationCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_same_router_flat_value_commutator_mitigation_closeout(
                probe_path=root / "missing_summary.json",
                variant_rows_path=root / "missing_variants.csv",
                gate_rows_path=root / "missing_gates.csv",
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

    def test_failed_mitigation_closes_flat_value_as_generic_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "summary.json"
            variants = root / "variant_rows.csv"
            gates = root / "gate_rows.csv"
            review = root / "latest-review.md"
            _write_json(
                probe,
                {
                    "status": "pass",
                    "decision": "same_router_flat_value_commutator_mitigation_probe_gpu_blocked",
                    "claim_status": "flat_value_commutator_mitigation_not_established",
                    "selected_next_action": "close_flat_value_commutator_mitigation_before_gpu",
                    "measured_passing_variant_count": 0,
                    "missing_required_variant_count": 2,
                },
            )
            variants.write_text(
                "\n".join(
                    [
                        "variant,measured,required_variant,variant_passes,commutator_ratio_to_budget,failure_reasons",
                        "flat_value_commutator_penalty_probe,True,True,False,1.62,finite_update_commutator_budget_failed",
                        "flat_value_order_averaged_updates,False,True,False,,missing direct row",
                        "flat_value_norm_clipped_updates,False,True,False,,missing direct row",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gates.write_text(
                "\n".join(
                    [
                        "gate,passes",
                        "at_least_one_measured_mitigation_variant,True",
                        "no_required_variants_missing,False",
                        "measured_variant_passes_all_gates,False",
                        "anchor_proxy_commutator_budget_passes,False",
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
                        "recommended_next_action: Do not launch RunPod yet",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_same_router_flat_value_commutator_mitigation_closeout(
                probe_path=probe,
                variant_rows_path=variants,
                gate_rows_path=gates,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "flat_value_commutator_mitigation_branch_closed_or_redirected",
            )
            self.assertEqual(summary["selected_next_action"], CLOSE_GENERIC_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "flat_value_capacity_closed_as_generic_capacity",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["evidence"]["measured_passing_variant_count"], 0)
            self.assertEqual(summary["evidence"]["missing_required_variant_count"], 2)
            self.assertTrue(summary["evidence"]["required_variants_missing"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            self.assertIn("generic capacity", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
