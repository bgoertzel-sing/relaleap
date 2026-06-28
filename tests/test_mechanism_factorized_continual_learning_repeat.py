from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mechanism_factorized_continual_learning_repeat import (
    REQUIRED_ARTIFACTS,
    run_mechanism_factorized_continual_learning_repeat,
)


class MechanismFactorizedContinualLearningRepeatTest(unittest.TestCase):
    def test_repeat_runs_missing_second_seed_and_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed3 = root / "seed3"
            seed5 = root / "seed5"

            summary = run_mechanism_factorized_continual_learning_repeat(
                seed_dirs=[seed3, seed5],
                seeds=[3, 5],
                out_dir=root / "repeat",
                steps_per_phase=1,
                refresh_missing=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "mechanism_factorized_cl_second_seed_repeat_recorded",
            )
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["repeat_row_count"], 2)
            self.assertIn("topk2_tradeoff_repeat_status", summary)
            self.assertIn("topk2_tradeoff_survives_all_seeds", _criteria(summary))
            self.assertIn("full_sparse_claim_survives_all_seeds", _criteria(summary))
            self.assertIn(
                "topk2_tradeoff_supporting_seed_count",
                summary["primary_result"],
            )
            self.assertTrue((seed3 / "summary.json").is_file())
            self.assertTrue((seed5 / "summary.json").is_file())
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "repeat" / artifact).is_file(), artifact)

    def test_repeat_fails_closed_when_sources_missing_and_refresh_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_mechanism_factorized_continual_learning_repeat(
                seed_dirs=[root / "missing_a", root / "missing_b"],
                seeds=[7, 11],
                out_dir=root / "repeat",
                refresh_missing=False,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "mechanism_factorized_cl_repeat_failed_closed")
            self.assertEqual(
                summary["selected_next_step"],
                "repair_missing_or_failed_mechanism_factorized_cl_seed_report",
            )
            self.assertFalse(
                next(
                    row
                    for row in summary["gate_criteria"]
                    if row["criterion"] == "required_seed_reports_present"
                )["passed"]
            )

    def test_repeat_consumes_existing_seed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_a = root / "seed_a"
            seed_b = root / "seed_b"
            _write_seed(seed_a, topk2_dense=True, topk2_random=True, full_claim=False)
            _write_seed(seed_b, topk2_dense=True, topk2_random=True, full_claim=False)

            summary = run_mechanism_factorized_continual_learning_repeat(
                seed_dirs=[seed_a, seed_b],
                seeds=[7, 11],
                out_dir=root / "repeat",
                refresh_missing=False,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["topk2_tradeoff_repeat_status"], "survives_second_seed")
            self.assertEqual(
                summary["claim_status"],
                "mechanism_factorized_sparse_retention_not_established",
            )
            self.assertEqual(
                summary["selected_next_step"],
                "use_repeated_topk2_tradeoff_signal_to_design_stricter_dense_null_controlled_interference_mitigation",
            )


def _criteria(summary: dict[str, object]) -> set[str]:
    return {str(row["criterion"]) for row in summary["gate_criteria"]}  # type: ignore[index]


def _write_seed(
    path: Path,
    *,
    topk2_dense: bool,
    topk2_random: bool,
    full_claim: bool,
) -> None:
    path.mkdir(parents=True)
    gates = [
        _gate("topk2_interference_per_gain_no_worse_than_dense", topk2_dense),
        _gate("topk2_beats_random_support_tradeoff_null", topk2_random),
        _gate("topk1_target_adaptation_no_worse_than_dense", full_claim),
        _gate("topk1_off_target_kl_no_worse_than_dense", full_claim),
        _gate("topk1_forgetting_no_worse_than_dense", full_claim),
        _gate("anchor_kl_sparse_no_worse_than_dense_anchor_kl", full_claim),
    ]
    packet = {
        "status": "pass",
        "decision": "mechanism_factorized_continual_learning_probe_recorded",
        "claim_status": "mechanism_factorized_sparse_retention_not_established",
        "selected_next_step": "run_second_seed_mechanism_factorized_cl_repeat_before_any_gpu_validation",
        "gate_criteria": gates,
        "primary_result": {
            "topk1_minus_dense_mean_target_ce_delta": 0.4,
            "topk1_minus_dense_mean_off_target_kl": -1.0,
            "topk2_minus_dense_forgetting_per_target_improvement": -0.2,
            "topk2_minus_dense_mean_final_forgetting": -0.1,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gate(name: str, passed: bool) -> dict[str, object]:
    return {"criterion": name, "passed": passed, "severity": "claim"}


if __name__ == "__main__":
    unittest.main()
