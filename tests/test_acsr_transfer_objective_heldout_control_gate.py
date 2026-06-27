from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_transfer_objective_heldout_control_gate import (
    REQUIRED_OUTPUT_ARTIFACTS,
    run_acsr_transfer_objective_heldout_control_gate,
)


class ACSRTransferObjectiveHeldoutControlGateTest(unittest.TestCase):
    def test_gate_passes_complete_heldout_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = (
                root / "acsr_transfer_objective_probe",
                root / "acsr_transfer_objective_probe_seed2",
                root / "runpod_acsr_transfer_objective_probe",
                root / "runpod_acsr_transfer_objective_probe_seed2",
            )
            for source in sources:
                _write_packet(source)

            summary = run_acsr_transfer_objective_heldout_control_gate(
                source_dirs=sources,
                out_dir=root / "gate",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["claim_status"],
                "heldout_transfer_controls_supported_not_promoted",
            )
            self.assertTrue(
                summary["aggregate_metrics"]["all_heldout_partner_beats_direct"]
            )
            self.assertTrue(
                summary["aggregate_metrics"]["all_low_norm_heldout_partner_beats_direct"]
            )
            self.assertTrue(
                summary["aggregate_metrics"]["all_high_norm_heldout_partner_beats_direct"]
            )
            for artifact in REQUIRED_OUTPUT_ARTIFACTS:
                self.assertTrue((root / "gate" / artifact).is_file(), artifact)

    def test_missing_packet_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "acsr_transfer_objective_probe"
            _write_packet(source)

            summary = run_acsr_transfer_objective_heldout_control_gate(
                source_dirs=(source, root / "missing_runpod"),
                out_dir=root / "gate",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_transfer_objective_heldout_control_gate_failed_closed",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "required_source_artifacts_present"
                    for row in summary["gate_criteria"]
                    if not row["passed"]
                )
            )
            self.assertTrue((root / "gate" / "summary.json").is_file())


def _write_packet(path: Path) -> None:
    path.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "acsr_transfer_objective_probe_recorded",
        "config_path": (
            "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_"
            "objective_gate_seed2.yaml"
            if "seed2" in str(path)
            else "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml"
        ),
        "git_commit": "test",
        "platform": "test",
    }
    (path / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    _write_csv(path / "gate_criteria.csv", [{"criterion": "own_ce_guardrail", "passed": True}])
    _write_csv(path / "arm_metrics.csv", [{"arm": "present", "ce_loss": 1.0}])
    (path / "notes.md").write_text("# notes\n", encoding="utf-8")

    rows = []
    for value_path in ("partner_values", "own_values"):
        for arm in (
            "direct_causal_mlp_baseline",
            "transfer_objective_router",
            "token_position_only_transfer_null",
            "random_frequency_support_null",
        ):
            rows.extend(_per_token_rows(value_path, arm))
    _write_csv(path / "per_token_metrics.csv", rows)


def _per_token_rows(value_path: str, arm: str) -> list[dict[str, object]]:
    rows = []
    # Four sequences, eight modeled positions each. The gate treats positions >=3
    # as held out because seq_len_minus_one is 8.
    for index in range(32):
        position = index % 8
        heldout = position >= 3
        base_loss = 5.0 + (0.01 * position)
        if value_path == "partner_values":
            adjustment = {
                "direct_causal_mlp_baseline": 0.0,
                "transfer_objective_router": -0.04 if heldout else -0.02,
                "token_position_only_transfer_null": 0.03,
                "random_frequency_support_null": 0.02,
            }[arm]
        else:
            adjustment = {
                "direct_causal_mlp_baseline": 0.0,
                "transfer_objective_router": 0.01,
                "token_position_only_transfer_null": 0.02,
                "random_frequency_support_null": 0.03,
            }[arm]
        rows.append(
            {
                "value_path": value_path,
                "arm": arm,
                "token_index": index,
                "ce_loss": base_loss + adjustment,
                "residual_update_l2": 0.2 if position < 6 else 0.8,
                "support": "1;2",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
