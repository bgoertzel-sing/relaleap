from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
                    "hep_alpha_acceptance": {
                        "status": "accepted",
                        "accepted_alpha": {
                            "alpha": 0.25,
                            "experiment_id": "char_smoke_hep",
                            "loss_improvement_from_alpha0": 0.01,
                            "max_logit_delta_from_ordinary": 0.05,
                        },
                    },
                },
                "runs": [
                    {
                        "experiment_id": "char_smoke",
                        "config_path": "configs/char_smoke.yaml",
                    },
                    {
                        "experiment_id": "char_smoke_hep",
                        "config_path": "configs/char_smoke_hep.yaml",
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
