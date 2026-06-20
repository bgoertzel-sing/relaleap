from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.decision_report import (
    GUIDED_ORACLE_CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    KEEP_OPT_IN,
    PROMOTE,
    PROMOTE_CLIPPED_HEP,
    write_clipped_hep_decision_report,
    write_guided_clipped_hep_decision_report,
    write_pinned_support_decision_report,
)


class PinnedSupportDecisionReportTest(unittest.TestCase):
    def test_support_stress_evidence_keeps_pinned_support_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_comparison(
                comparison_dir,
                pinned_nonzero_loss=4.5,
                pinned_nonzero_delta=0.5,
            )

            report = write_pinned_support_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], KEEP_OPT_IN)
            self.assertFalse(report["promote_to_default_phase0_baseline"])
            self.assertEqual(report["evidence"]["artifact_check_status"], "pass")
            self.assertGreater(report["evidence"]["max_support_change_fraction"], 0.0)
            self.assertGreater(
                report["evidence"]["max_pinned_vs_repicked_logit_delta"],
                0.0,
            )
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_improving_pinned_alpha_promotes_to_default_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_comparison(
                comparison_dir,
                pinned_nonzero_loss=3.9,
                pinned_nonzero_delta=0.05,
            )

            report = write_pinned_support_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], PROMOTE)
            self.assertTrue(report["promote_to_default_phase0_baseline"])

    def test_failed_artifact_check_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_comparison(
                comparison_dir,
                pinned_nonzero_loss=3.9,
                pinned_nonzero_delta=0.05,
            )
            artifact_check_path = tmp_path / "artifact_check.json"
            artifact_check_path.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_pinned_support_decision_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["promote_to_default_phase0_baseline"])


class ClippedHepDecisionReportTest(unittest.TestCase):
    def test_clipped_support_stress_evidence_keeps_clipping_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_clipped_comparison(
                comparison_dir,
                clipped_nonzero_loss=4.0,
                clipped_nonzero_delta=0.0,
                clipped_nonzero_pinned_vs_repicked=0.002,
            )

            report = write_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], KEEP_OPT_IN)
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])
            self.assertEqual(report["evidence"]["artifact_check_status"], "pass")
            self.assertGreater(
                report["evidence"]["max_pinned_vs_repicked_logit_delta_reduction"],
                0.0,
            )
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_improving_clipped_alpha_promotes_support_stress_mitigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_clipped_comparison(
                comparison_dir,
                clipped_nonzero_loss=3.9,
                clipped_nonzero_delta=0.05,
                clipped_nonzero_pinned_vs_repicked=0.02,
            )

            report = write_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], PROMOTE_CLIPPED_HEP)
            self.assertTrue(report["promote_to_default_support_stress_mitigation"])

    def test_failed_artifact_check_blocks_clipped_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_clipped_comparison(
                comparison_dir,
                clipped_nonzero_loss=3.9,
                clipped_nonzero_delta=0.05,
                clipped_nonzero_pinned_vs_repicked=0.02,
            )
            artifact_check_path = tmp_path / "artifact_check.json"
            artifact_check_path.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])


class GuidedClippedHepDecisionReportTest(unittest.TestCase):
    def test_improving_guided_alpha_records_oracle_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_guided_clipped_comparison(
                comparison_dir,
                guided_nonzero_loss=3.9,
                guided_nonzero_delta=0.05,
                guided_nonzero_pinned_vs_repicked=0.02,
            )

            report = write_guided_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], GUIDED_ORACLE_CONFIRMED)
            self.assertTrue(report["diagnostic_oracle_only"])
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])
            self.assertEqual(report["evidence"]["artifact_check_status"], "pass")
            self.assertEqual(report["evidence"]["guided_run_count"], 1)
            self.assertEqual(report["evidence"]["clipped_baseline_run_count"], 1)
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_failed_artifact_check_blocks_guided_oracle_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_guided_clipped_comparison(
                comparison_dir,
                guided_nonzero_loss=3.9,
                guided_nonzero_delta=0.05,
                guided_nonzero_pinned_vs_repicked=0.02,
            )
            artifact_check_path = tmp_path / "artifact_check.json"
            artifact_check_path.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_guided_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertTrue(report["diagnostic_oracle_only"])
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])


def _write_comparison(
    comparison_dir: Path,
    *,
    pinned_nonzero_loss: float,
    pinned_nonzero_delta: float,
) -> None:
    comparison_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "invariant_count": 8,
            "failed_invariants": [],
            "artifact_invariants_passed": True,
            "artifact_invariant_count": 6,
            "failed_artifact_invariants": [],
            "hep_alpha_acceptance": {"status": "accepted"},
        },
        "runs": [
            _run_entry("repicked", pinned=False, nonzero_loss=4.0, nonzero_delta=0.0),
            _run_entry(
                "pinned",
                pinned=True,
                nonzero_loss=pinned_nonzero_loss,
                nonzero_delta=pinned_nonzero_delta,
            ),
        ],
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for run_id in ("repicked", "pinned"):
        run_dir = comparison_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": run_id,
                    "status": "ok",
                    "artifact_invariants": {
                        "summary_json": True,
                        "metrics_csv": True,
                        "notes_md": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")


def _write_clipped_comparison(
    comparison_dir: Path,
    *,
    clipped_nonzero_loss: float,
    clipped_nonzero_delta: float,
    clipped_nonzero_pinned_vs_repicked: float,
) -> None:
    comparison_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "invariant_count": 8,
            "failed_invariants": [],
            "artifact_invariants_passed": True,
            "artifact_invariant_count": 6,
            "failed_artifact_invariants": [],
            "hep_alpha_acceptance": {"status": "no_accepted_alpha"},
        },
        "runs": [
            _clipped_run_entry(
                "unclipped",
                clip_norm=None,
                nonzero_loss=4.0,
                nonzero_delta=0.0,
                nonzero_pinned_vs_repicked=1.0,
            ),
            _clipped_run_entry(
                "clipped",
                clip_norm=0.01,
                nonzero_loss=clipped_nonzero_loss,
                nonzero_delta=clipped_nonzero_delta,
                nonzero_pinned_vs_repicked=clipped_nonzero_pinned_vs_repicked,
            ),
        ],
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for run_id in ("unclipped", "clipped"):
        run_dir = comparison_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": run_id,
                    "status": "ok",
                    "artifact_invariants": {
                        "summary_json": True,
                        "metrics_csv": True,
                        "notes_md": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")


def _write_guided_clipped_comparison(
    comparison_dir: Path,
    *,
    guided_nonzero_loss: float,
    guided_nonzero_delta: float,
    guided_nonzero_pinned_vs_repicked: float,
) -> None:
    comparison_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "invariant_count": 8,
            "failed_invariants": [],
            "artifact_invariants_passed": True,
            "artifact_invariant_count": 6,
            "failed_artifact_invariants": [],
            "hep_alpha_acceptance": {"status": "accepted"},
        },
        "runs": [
            _guided_clipped_run_entry(
                "clipped",
                settling_objective="residual_adapter",
                nonzero_loss=4.0,
                nonzero_delta=0.0,
                nonzero_pinned_vs_repicked=0.002,
            ),
            _guided_clipped_run_entry(
                "guided",
                settling_objective="supervised_ce_gradient",
                nonzero_loss=guided_nonzero_loss,
                nonzero_delta=guided_nonzero_delta,
                nonzero_pinned_vs_repicked=guided_nonzero_pinned_vs_repicked,
            ),
        ],
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for run_id in ("clipped", "guided"):
        run_dir = comparison_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": run_id,
                    "status": "ok",
                    "artifact_invariants": {
                        "summary_json": True,
                        "metrics_csv": True,
                        "notes_md": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")


def _run_entry(
    experiment_id: str,
    *,
    pinned: bool,
    nonzero_loss: float,
    nonzero_delta: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "config_path": f"configs/{experiment_id}.yaml",
        "status": "ok",
        "pinned_support": pinned,
        "support_stress": True,
        "support_instability": {
            "support_change_fraction": 0.5,
            "pinned_vs_repicked_logit_delta": 1.0,
        },
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": 4.0,
                "max_logit_delta_from_ordinary": 0.0,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": 0.0,
            },
            {
                "alpha": 0.25,
                "loss": nonzero_loss,
                "max_logit_delta_from_ordinary": nonzero_delta,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": 1.0,
            },
        ],
    }


def _clipped_run_entry(
    experiment_id: str,
    *,
    clip_norm: float | None,
    nonzero_loss: float,
    nonzero_delta: float,
    nonzero_pinned_vs_repicked: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "config_path": f"configs/{experiment_id}.yaml",
        "status": "ok",
        "pinned_support": False,
        "support_stress": True,
        "hep_update_clip_norm": clip_norm,
        "support_instability": {
            "support_change_fraction": 0.5,
            "pinned_vs_repicked_logit_delta": nonzero_pinned_vs_repicked,
        },
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": 4.0,
                "max_logit_delta_from_ordinary": 0.0,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": 0.0,
            },
            {
                "alpha": 0.25,
                "loss": nonzero_loss,
                "max_logit_delta_from_ordinary": nonzero_delta,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": nonzero_pinned_vs_repicked,
            },
        ],
    }


def _guided_clipped_run_entry(
    experiment_id: str,
    *,
    settling_objective: str,
    nonzero_loss: float,
    nonzero_delta: float,
    nonzero_pinned_vs_repicked: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "config_path": f"configs/{experiment_id}.yaml",
        "status": "ok",
        "pinned_support": False,
        "support_stress": True,
        "hep_update_clip_norm": 0.01,
        "hep_settling_objective": settling_objective,
        "support_instability": {
            "support_change_fraction": 0.5,
            "pinned_vs_repicked_logit_delta": nonzero_pinned_vs_repicked,
        },
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": 4.0,
                "max_logit_delta_from_ordinary": 0.0,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": 0.0,
            },
            {
                "alpha": 0.25,
                "loss": nonzero_loss,
                "max_logit_delta_from_ordinary": nonzero_delta,
                "support_change_fraction": 0.5,
                "pinned_vs_repicked_logit_delta": nonzero_pinned_vs_repicked,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
