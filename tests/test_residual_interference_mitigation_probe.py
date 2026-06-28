from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.residual_interference_mitigation_probe import (
    REQUIRED_ARTIFACTS,
    run_residual_interference_mitigation_probe,
)


class ResidualInterferenceMitigationProbeTest(unittest.TestCase):
    def test_missing_source_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_residual_interference_mitigation_probe(
                source_summary_path=root / "missing.json",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "residual_interference_mitigation_probe_failed_closed",
            )
            self.assertEqual(
                summary["claim_status"],
                "residual_interference_mitigation_failed_closed",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_support_width_partial_candidate_passes_hard_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "summary.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "mechanism_factorized_continual_learning_probe_recorded",
                        "claim_status": "mechanism_factorized_sparse_retention_not_established",
                        "arm_metrics": [
                            _arm("dense_active_rank", -0.58, 0.08, 1.5, 0.22),
                            _arm("contextual_topk1", -0.10, -0.008, 0.03, 0.009),
                            _arm("contextual_topk2", -0.18, -0.002, 0.07, 0.002),
                            _arm("random_frequency_matched_topk2", -0.09, 0.0, 0.05, 0.012),
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_residual_interference_mitigation_probe(
                source_summary_path=source,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["claim_status"],
                "support_width_mitigation_partial_candidate_not_promoted",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "topk2_improves_sparse_target_adaptation_vs_topk1"
                    and row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            self.assertLess(
                summary["primary_result"]["topk2_minus_topk1_target_ce_delta"],
                0.0,
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _arm(
    arm: str,
    target: float,
    off_ce: float,
    off_kl: float,
    forgetting: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "mean_target_ce_delta": target,
        "mean_off_target_ce_drift": off_ce,
        "mean_off_target_kl": off_kl,
        "mean_final_forgetting": forgetting,
    }


if __name__ == "__main__":
    unittest.main()
