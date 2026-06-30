from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.context_contrastive_core_periphery_probe import (
    CONTEXT_CANDIDATE,
    REQUIRED_ARTIFACTS,
    run_context_contrastive_core_periphery_probe,
)


class ContextContrastiveCorePeripheryProbeTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_context_contrastive_core_periphery_probe(
                design_dir=root / "missing_design",
                core_pilot_dir=root / "missing_core",
                low_churn_dir=root / "missing_low_churn",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "context_contrastive_core_periphery_probe_recorded_but_blocked",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_local_probe_and_blocks_on_claim_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design"
            core = root / "core"
            low_churn = root / "low_churn"
            _write_design(design)
            _write_core_pilot(core)
            _write_low_churn(low_churn)

            summary = run_context_contrastive_core_periphery_probe(
                design_dir=design,
                core_pilot_dir=core,
                low_churn_dir=low_churn,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "context_contrastive_core_periphery_probe_recorded_but_blocked",
            )
            self.assertEqual(summary["claim_status"], "context_contrastive_core_periphery_not_established")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(
                summary["selected_next_action"],
                "close_or_redesign_context_contrastive_core_periphery_before_gpu",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("retention_churn_budget_nonworse", failed)
            self.assertIn("periphery_first_pruning_signal_positive", failed)
            self.assertEqual(summary["candidate_observables"]["heldout_ce"], 3.79)

            with (root / "out" / "probe_rows.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(CONTEXT_CANDIDATE, {row["name"] for row in rows})
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _write_design(path: Path) -> None:
    path.mkdir(parents=True)
    payload = {
        "status": "pass",
        "decision": "context_contrastive_core_periphery_probe_design_recorded",
        "selected_next_action": "implement_context_contrastive_core_periphery_probe_locally",
    }
    (path / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_core_pilot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "core_periphery_pc_column_nonsynthetic_pilot_recorded_but_blocked",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "variant",
        "heldout_ce",
        "anchor_kl_drift",
        "functional_churn",
        "finite_update_commutator",
        "periphery_first_minus_core_first_prune_delta_heldout",
        "paired_heldout_periphery_utility_mean",
    ]
    rows = [
        [CONTEXT_CANDIDATE, 3.79, 0.0015, 0.17, 0.006, -0.03, 0.03],
        ["retention_constrained_gated_periphery", 3.81, 0.0002, 0.04, 0.001, -0.01, 0.01],
        ["token_position_only_router", 3.9, 0.001, 0.2, 0.006, 0.0, 0.0],
        ["permuted_periphery_target_null", 3.83, 0.0001, 0.0, 0.0001, 0.0, -0.001],
        ["frequency_support_router", 3.95, 0.001, 0.1, 0.005, 0.0, 0.0],
        ["dense_rank_norm_residual", 3.82, 0.0004, 0.05, 0.002, 0.0, 0.0],
        ["parameter_matched_causal_mlp", 3.7, 0.002, 0.4, 0.01, 0.0, 0.0],
        ["no_core_ablation", 3.9, 0.001, 0.2, 0.006, 0.0, 0.0],
        ["no_periphery_ablation", 3.88, 0.0005, 0.1, 0.003, 0.0, 0.0],
        ["equal_plasticity_core_periphery", 3.84, 0.0005, 0.1, 0.003, 0.0, 0.0],
    ]
    with (path / "variant_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    (path / "intervention_fingerprints.csv").write_text(
        "variant,unit,necessity_heldout_delta\n"
        f"{CONTEXT_CANDIDATE},core,0.001\n"
        f"{CONTEXT_CANDIDATE},periphery,0.03\n",
        encoding="utf-8",
    )


def _write_low_churn(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_residual_control_pilot_completed",
                "advance_to_gpu_validation": False,
                "advancement_row_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "arm_metrics.csv").write_text(
        "arm,heldout_ce_loss,heldout_anchor_kl_vs_base,heldout_prediction_flip_rate\n"
        "low_churn_mlp_residual_control,3.62,0.0001,0.02\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
