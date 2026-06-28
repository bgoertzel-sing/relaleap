from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_pc_column_nonsynthetic_pilot_design import (
    REQUIRED_ARTIFACTS,
    REQUIRED_HIDDEN_STATE_FIELDS,
    REQUIRED_PILOT_ARMS,
    run_core_periphery_pc_column_nonsynthetic_pilot_design,
)


class CorePeripheryPCColumnNonSyntheticPilotDesignTest(unittest.TestCase):
    def test_records_ready_local_design_with_controls_and_hidden_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis.json"
            synthesis.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "scientific_gate": "ready_for_non_synthetic_pilot_design",
                        "decision": "core_periphery_pc_column_local_repeats_supported",
                        "claim_status": "synthetic_local_repeat_only_not_gpu_or_promotion_evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "scientific_gate": "ready_for_tiny_pilot",
                        "decision": "core_periphery_pc_column_design_recorded",
                        "claim_status": "design_contract_only_not_training_evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Continue local fail-closed design",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            char_config = _write_config(root / "char.yaml", "char_validation", 64, 64, 12)
            larger_config = _write_config(root / "larger.yaml", "char_larger", 128, 96, 24)
            token_config = _write_config(root / "token.yaml", "token_larger", 96, 96, 24)

            summary = run_core_periphery_pc_column_nonsynthetic_pilot_design(
                synthesis_path=synthesis,
                design_contract_path=contract,
                char_config_path=char_config,
                larger_config_path=larger_config,
                token_config_path=token_config,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["scientific_gate"],
                "ready_for_local_nonsynthetic_pilot_implementation",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["failures"], [])
            self.assertTrue(
                set(REQUIRED_PILOT_ARMS).issubset(
                    {row["arm"] for row in summary["pilot_arms"]}
                )
            )
            self.assertTrue(
                set(REQUIRED_HIDDEN_STATE_FIELDS).issubset(
                    {row["field"] for row in summary["hidden_state_contract"]}
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_synthesis_blocks_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps({"status": "pass", "scientific_gate": "ready_for_tiny_pilot"}) + "\n",
                encoding="utf-8",
            )
            config = _write_config(root / "config.yaml", "char_validation", 64, 64, 12)

            summary = run_core_periphery_pc_column_nonsynthetic_pilot_design(
                synthesis_path=root / "missing.json",
                design_contract_path=contract,
                char_config_path=config,
                larger_config_path=config,
                token_config_path=config,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertTrue(
                any(
                    row["criterion"] == "synthesis_ready_for_nonsynthetic_design"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )


def _write_config(
    path: Path,
    experiment_id: str,
    seq_len: int,
    hidden_dim: int,
    num_columns: int,
) -> Path:
    path.write_text(
        f"""
run:
  experiment_id: {experiment_id}
  seed: 1
  max_steps: 25
data:
  dataset: tiny_shakespeare_char
  seq_len: {seq_len}
model:
  base:
    layers: 2
    hidden_dim: {hidden_dim}
  columns:
    num_columns: {num_columns}
    atoms_per_column: 4
    top_k: 2
    support_router: contextual_mlp
""".lstrip(),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
