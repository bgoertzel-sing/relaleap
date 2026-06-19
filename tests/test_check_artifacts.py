from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.compare import write_comparison_baseline
from relaleap.experiments.check_artifacts import check_comparison_artifacts


class CheckArtifactsTest(unittest.TestCase):
    def test_complete_comparison_tree_passes_with_baseline_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_dir = Path(tmpdir) / "comparison"
            _write_comparison_tree(comparison_dir)

            report = check_comparison_artifacts(
                comparison_dir,
                require_baseline_comparison=True,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["summary_status"], "ok")
            self.assertEqual(report["verdict_status"], "pass")
            self.assertEqual(report["phase0_invariants"]["count"], 12)
            self.assertEqual(
                report["hep_alpha_acceptance"]["accepted_alpha"]["alpha"],
                0.25,
            )
            self.assertEqual(report["baseline_comparison"]["status"], "pass")
            self.assertEqual(report["failures"], [])

    def test_missing_required_baseline_comparison_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_dir = Path(tmpdir) / "comparison"
            _write_comparison_tree(comparison_dir, include_baseline=False)

            report = check_comparison_artifacts(
                comparison_dir,
                require_baseline_comparison=True,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["failures"],
                [
                    {
                        "field": "comparison.baseline_comparison.json",
                        "expected": "file exists",
                        "actual": "missing",
                        "path": str(comparison_dir / "baseline_comparison.json"),
                    }
                ],
            )

    def test_failed_baseline_gate_fails_artifact_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_dir = Path(tmpdir) / "comparison"
            _write_comparison_tree(comparison_dir, baseline_status="fail")

            report = check_comparison_artifacts(
                comparison_dir,
                require_baseline_comparison=True,
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn(
                {
                    "field": "baseline_comparison.status",
                    "expected": "pass",
                    "actual": "fail",
                },
                report["failures"],
            )

    def test_invalid_baseline_json_fails_artifact_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_dir = Path(tmpdir) / "comparison"
            _write_comparison_tree(comparison_dir)
            (comparison_dir / "baseline_comparison.json").write_text(
                "{not-json\n",
                encoding="utf-8",
            )

            report = check_comparison_artifacts(
                comparison_dir,
                require_baseline_comparison=True,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["failures"][0]["field"],
                "comparison.baseline_comparison",
            )
            self.assertEqual(
                report["failures"][0]["expected"],
                "valid JSON object",
            )

    def test_baseline_reference_passes_without_written_baseline_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_comparison_tree(comparison_dir, include_baseline=False)
            baseline_path = tmp_path / "baseline.json"
            summary = json.loads(
                (comparison_dir / "summary.json").read_text(encoding="utf-8")
            )
            write_comparison_baseline(baseline_path, summary)

            report = check_comparison_artifacts(
                comparison_dir,
                baseline_reference=baseline_path,
            )

            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["baseline_comparison"]["present"])
            self.assertEqual(
                report["baseline_reference_comparison"]["status"],
                "pass",
            )
            self.assertEqual(report["failures"], [])

    def test_baseline_reference_drift_fails_artifact_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_comparison_tree(comparison_dir, include_baseline=False)
            baseline_path = tmp_path / "baseline.json"
            summary = json.loads(
                (comparison_dir / "summary.json").read_text(encoding="utf-8")
            )
            baseline = write_comparison_baseline(baseline_path, summary)
            baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"] = 0.5
            baseline_path.write_text(
                json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = check_comparison_artifacts(
                comparison_dir,
                baseline_reference=baseline_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["baseline_reference_comparison"]["status"],
                "fail",
            )
            self.assertIn(
                {
                    "field": "baseline_reference.hep.acceptance.accepted_alpha.alpha",
                    "expected": 0.5,
                    "actual": 0.25,
                },
                report["failures"],
            )


def _write_comparison_tree(
    comparison_dir: Path,
    *,
    include_baseline: bool = True,
    baseline_status: str = "pass",
) -> None:
    comparison_dir.mkdir(parents=True)
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (comparison_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "verdict": {
                    "status": "pass",
                    "invariants_passed": True,
                    "invariant_count": 12,
                    "failed_invariants": [],
                    "best_hep_alpha_by_loss": {
                        "alpha": 1.0,
                        "experiment_id": "char_smoke_hep",
                        "loss": 3.52012658,
                        "max_logit_delta_from_ordinary": 0.20742974,
                    },
                    "hep_alpha_acceptance": {
                        "status": "accepted",
                        "max_logit_delta_from_ordinary": 0.1,
                        "min_loss_improvement_from_alpha0": 0.0,
                        "baseline_alpha0": {
                            "alpha": 0.0,
                            "experiment_id": "char_smoke_hep",
                            "loss": 3.56317067,
                            "max_logit_delta_from_ordinary": 0.0,
                        },
                        "accepted_alpha": {
                            "alpha": 0.25,
                            "experiment_id": "char_smoke_hep",
                            "loss": 3.55195642,
                            "loss_improvement_from_alpha0": 0.01,
                            "max_logit_delta_from_ordinary": 0.05,
                        },
                        "candidate_count": 2,
                        "rejected_count": 1,
                    },
                },
                "runs": [
                    {
                        "experiment_id": "char_smoke",
                        "config_path": "configs/char_smoke.yaml",
                        "residual_objective": "supervised_ce",
                        "status": "ok",
                        "training_steps": 10,
                        "final_residual_loss": 3.56,
                        "invariants": {
                            "frozen_base_unchanged": True,
                            "required_artifacts_written": True,
                            "zero_init_identity": True,
                        },
                    },
                    {
                        "experiment_id": "char_smoke_hep",
                        "config_path": "configs/char_smoke_hep.yaml",
                        "residual_objective": "supervised_ce",
                        "status": "ok",
                        "training_steps": 10,
                        "final_residual_loss": 3.55,
                        "invariants": {
                            "frozen_base_unchanged": True,
                            "hep_alpha_0_equivalence": True,
                            "required_artifacts_written": True,
                            "zero_init_identity": True,
                        },
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.56317067,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                            {
                                "alpha": 0.25,
                                "loss": 3.55195642,
                                "max_logit_delta_from_ordinary": 0.05185753,
                            },
                            {
                                "alpha": 1.0,
                                "loss": 3.52012658,
                                "max_logit_delta_from_ordinary": 0.20742974,
                            },
                        ],
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for stem in ["char_smoke", "char_smoke_hep"]:
        run_dir = comparison_dir / "runs" / stem
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    if include_baseline:
        (comparison_dir / "baseline_comparison.json").write_text(
            json.dumps(
                {
                    "status": baseline_status,
                    "mismatches": (
                        []
                        if baseline_status == "pass"
                        else [{"field": "hep.acceptance.accepted_alpha.alpha"}]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
