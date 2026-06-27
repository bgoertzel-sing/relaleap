from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_transfer_objective_validation_gate import (
    REQUIRED_OUTPUT_ARTIFACTS,
    run_acsr_transfer_objective_validation_gate,
)


class ACSRTransferObjectiveValidationGateTest(unittest.TestCase):
    def test_gate_passes_complete_local_runpod_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = (
                root / "acsr_transfer_objective_probe",
                root / "acsr_transfer_objective_probe_seed2",
                root / "runpod_acsr_transfer_objective_probe",
                root / "runpod_acsr_transfer_objective_probe_seed2",
            )
            _write_packet(sources[0], "configs/token.yaml", -0.03, -0.04, -0.05, 0.01)
            _write_packet(sources[1], "configs/token_seed2.yaml", -0.02, -0.01, -0.02, 0.015)
            _write_packet(sources[2], "configs/token.yaml", -0.031, -0.039, -0.049, 0.011)
            _write_packet(sources[3], "configs/token_seed2.yaml", -0.021, -0.011, -0.021, 0.014)

            summary = run_acsr_transfer_objective_validation_gate(
                source_dirs=sources,
                out_dir=root / "gate",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["claim_status"],
                "cross_backend_transfer_objective_supported_not_promoted",
            )
            self.assertEqual(len(summary["primary_metrics_by_packet"]), 4)
            self.assertTrue(summary["aggregate_metrics"]["all_partner_beats_direct"])
            self.assertTrue(summary["aggregate_metrics"]["all_own_ce_within_guardrail"])
            for artifact in REQUIRED_OUTPUT_ARTIFACTS:
                self.assertTrue((root / "gate" / artifact).is_file(), artifact)

    def test_missing_packet_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "acsr_transfer_objective_probe"
            _write_packet(source, "configs/token.yaml", -0.03, -0.04, -0.05, 0.01)

            summary = run_acsr_transfer_objective_validation_gate(
                source_dirs=(source, root / "missing_runpod"),
                out_dir=root / "gate",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_transfer_objective_validation_gate_failed_closed",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "required_source_artifacts_present"
                    for row in summary["gate_criteria"]
                    if not row["passed"]
                )
            )
            self.assertTrue((root / "gate" / "summary.json").is_file())


def _write_packet(
    path: Path,
    config_path: str,
    partner_vs_direct: float,
    partner_vs_token: float,
    partner_vs_random: float,
    own_vs_direct: float,
) -> None:
    path.mkdir(parents=True)
    direct_regret = 0.12
    transfer_regret = 0.09
    summary = {
        "status": "pass",
        "decision": "acsr_transfer_objective_probe_recorded",
        "claim_status": "local_transfer_objective_supported_not_promoted",
        "config_path": config_path,
        "primary_metrics": {
            "partner_transfer_minus_direct_ce": partner_vs_direct,
            "partner_transfer_minus_token_position_ce": partner_vs_token,
            "partner_transfer_minus_random_ce": partner_vs_random,
            "own_transfer_minus_direct_ce": own_vs_direct,
            "partner_transfer_oracle_regret": transfer_regret,
            "partner_direct_oracle_regret": direct_regret,
            "partner_transfer_residual_norm_normalized_delta_vs_direct": 12.0,
            "transfer_support_jaccard_with_direct": 0.6,
        },
        "git_commit": "test",
        "platform": "test",
    }
    (path / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    _write_csv(path / "gate_criteria.csv", [{"criterion": "own_ce_guardrail", "passed": True}])
    for artifact in ("metrics.csv", "arm_metrics.csv", "per_token_metrics.csv"):
        _write_csv(path / artifact, [{"metric": "present", "value": 1}])
    (path / "notes.md").write_text("# notes\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
