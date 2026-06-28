from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_negative_evidence_closeout import (
    DEMOTE_CORE_PERIPHERY_ACTION,
    LOCAL_MECHANISM_REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    REPAIR_SOURCES_ACTION,
    run_core_periphery_negative_evidence_closeout,
)


class CorePeripheryNegativeEvidenceCloseoutTest(unittest.TestCase):
    def test_demotes_current_mechanism_when_retention_and_pruning_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot.json"
            design = root / "design.json"
            synthesis = root / "synthesis.json"
            review = root / "latest-review.md"
            _write_json(
                pilot,
                {
                    "status": "pass",
                    "decision": "core_periphery_pc_column_nonsynthetic_pilot_recorded_but_blocked",
                    "claim_status": "local_nonsynthetic_signal_insufficient_for_gpu_or_promotion",
                    "scientific_gate": "blocked",
                    "selected_next_step": "inspect failed claim gates",
                    "primary_result": {
                        "primary_variant": "retention_constrained_gated_periphery",
                        "heldout_ce": 3.81,
                        "anchor_kl_drift": 2.39e-4,
                        "core_minus_dense_anchor_kl_drift": 1.93e-4,
                        "core_minus_mlp_anchor_kl_drift": 2.34e-4,
                        "periphery_deployment_fraction": 0.80,
                        "effective_periphery_residual_norm": 0.19,
                        "paired_train_periphery_utility_mean": 0.015,
                        "paired_heldout_periphery_utility_mean": 0.012,
                        "paired_heldout_periphery_utility_positive_fraction": 0.55,
                        "periphery_first_minus_core_first_prune_delta": -0.016,
                    },
                    "gate_criteria": [
                        {"criterion": "matched_dense_retention", "severity": "claim", "passed": False},
                        {"criterion": "matched_mlp_retention", "severity": "claim", "passed": False},
                        {"criterion": "periphery_first_pruning_signal", "severity": "claim", "passed": False},
                    ],
                },
            )
            _write_json(
                design,
                {
                    "status": "pass",
                    "scientific_gate": "ready_for_local_nonsynthetic_pilot_implementation",
                },
            )
            _write_json(
                synthesis,
                {
                    "status": "pass",
                    "decision": "core_periphery_synthesis_recorded",
                    "claim_status": "core_periphery_branch_requires_nonsynthetic_pilot",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Keep the pilot local",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_core_periphery_negative_evidence_closeout(
                pilot_path=pilot,
                design_path=design,
                synthesis_path=synthesis,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], DEMOTE_CORE_PERIPHERY_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(
                summary["claim_status"],
                "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
            )
            self.assertTrue(summary["evidence"]["useful_periphery_observed"])
            self.assertTrue(summary["evidence"]["retention_worse_than_dense_or_mlp"])
            self.assertTrue(summary["evidence"]["protected_core_factorization_failed"])
            self.assertEqual(summary["direction_shift"]["ben_should_be_notified"], False)
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], DEMOTE_CORE_PERIPHERY_ACTION)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_required_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot.json"
            design = root / "design.json"
            _write_json(pilot, {"status": "pass", "scientific_gate": "blocked"})
            _write_json(design, {"status": "pass"})

            summary = run_core_periphery_negative_evidence_closeout(
                pilot_path=pilot,
                design_path=design,
                synthesis_path=root / "missing-synthesis.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_SOURCES_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())

    def test_selects_local_repair_when_current_mechanism_clears_closeout_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot.json"
            design = root / "design.json"
            synthesis = root / "synthesis.json"
            _write_json(
                pilot,
                {
                    "status": "pass",
                    "decision": "local_candidate",
                    "claim_status": "local_nonsynthetic_candidate_not_gpu_or_promotion_evidence",
                    "scientific_gate": "ready_for_local_repeat_only",
                    "primary_result": {
                        "primary_variant": "retention_constrained_gated_periphery",
                        "core_minus_dense_anchor_kl_drift": -1.0e-5,
                        "core_minus_mlp_anchor_kl_drift": -1.0e-5,
                        "periphery_deployment_fraction": 0.75,
                        "effective_periphery_residual_norm": 0.1,
                        "paired_heldout_periphery_utility_mean": 0.02,
                        "periphery_first_minus_core_first_prune_delta": 0.01,
                    },
                },
            )
            _write_json(design, {"status": "pass", "scientific_gate": "ready"})
            _write_json(synthesis, {"status": "pass", "decision": "synthesis"})

            summary = run_core_periphery_negative_evidence_closeout(
                pilot_path=pilot,
                design_path=design,
                synthesis_path=synthesis,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], LOCAL_MECHANISM_REPAIR_ACTION)
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
