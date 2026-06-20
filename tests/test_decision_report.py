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
    SELECT_TEMPORAL_CLIPPED_HEP,
    SELECT_TEMPORAL_CLIPPED_HEP_AGGREGATE,
    SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE,
    DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
    write_clipped_hep_decision_report,
    write_guided_clipped_hep_decision_report,
    write_pinned_support_decision_report,
    write_temporal_clipped_hep_aggregate_report,
    write_temporal_clipped_hep_cross_scale_aggregate_report,
    write_temporal_clipped_hep_decision_report,
    write_temporal_clipped_hep_promotion_gate_report,
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


class TemporalClippedHepDecisionReportTest(unittest.TestCase):
    def test_improving_temporal_alpha_selects_label_free_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_temporal_clipped_comparison(
                comparison_dir,
                temporal_nonzero_loss=3.99,
                entropy_nonzero_loss=4.01,
                guided_nonzero_loss=3.9,
                temporal_nonzero_delta=0.01,
                temporal_nonzero_pinned_vs_repicked=0.02,
            )

            report = write_temporal_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], SELECT_TEMPORAL_CLIPPED_HEP)
            self.assertTrue(report["selected_label_free_support_stress_candidate"])
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])
            self.assertTrue(report["deployable_label_free_signal"])
            self.assertEqual(report["evidence"]["artifact_check_status"], "pass")
            self.assertEqual(report["evidence"]["temporal_run_count"], 1)
            self.assertEqual(report["evidence"]["entropy_run_count"], 1)
            self.assertEqual(report["evidence"]["guided_run_count"], 1)
            self.assertEqual(report["evidence"]["clipped_baseline_run_count"], 1)
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_failed_artifact_check_blocks_temporal_candidate_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_temporal_clipped_comparison(
                comparison_dir,
                temporal_nonzero_loss=3.99,
                entropy_nonzero_loss=4.01,
                guided_nonzero_loss=3.9,
                temporal_nonzero_delta=0.01,
                temporal_nonzero_pinned_vs_repicked=0.02,
            )
            artifact_check_path = tmp_path / "artifact_check.json"
            artifact_check_path.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_temporal_clipped_hep_decision_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["selected_label_free_support_stress_candidate"])
            self.assertFalse(report["promote_to_default_support_stress_mitigation"])


class TemporalClippedHepAggregateReportTest(unittest.TestCase):
    def test_all_selected_temporal_reports_pass_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_paths = []
            for seed in (1, 2):
                for backend in ("local", "colab"):
                    comparison_dir = tmp_path / f"{backend}_comparison_seed{seed}"
                    _write_temporal_clipped_comparison(
                        comparison_dir,
                        temporal_nonzero_loss=3.99,
                        entropy_nonzero_loss=4.01,
                        guided_nonzero_loss=3.9,
                        temporal_nonzero_delta=0.01,
                        temporal_nonzero_pinned_vs_repicked=0.02,
                    )
                    out_dir = tmp_path / f"temporal_seed{seed}_{backend}_decision"
                    decision = write_temporal_clipped_hep_decision_report(
                        comparison_dir,
                        out_dir,
                    )
                    self.assertEqual(decision["status"], "pass")
                    report_paths.append(out_dir / "decision_report.json")

            aggregate = write_temporal_clipped_hep_aggregate_report(
                report_paths,
                tmp_path / "aggregate",
            )

            self.assertEqual(aggregate["status"], "pass")
            self.assertEqual(
                aggregate["decision"],
                SELECT_TEMPORAL_CLIPPED_HEP_AGGREGATE,
            )
            self.assertTrue(aggregate["selected_label_free_support_stress_candidate"])
            self.assertFalse(aggregate["promote_to_default_support_stress_mitigation"])
            self.assertEqual(aggregate["evidence"]["report_count"], 4)
            self.assertEqual(aggregate["evidence"]["selected_report_count"], 4)
            self.assertEqual(
                aggregate["evidence"]["accepted_temporal_report_count"],
                4,
            )
            self.assertTrue(
                (tmp_path / "aggregate" / "decision_report.json").is_file()
            )
            self.assertTrue((tmp_path / "aggregate" / "decision_report.md").is_file())

    def test_missing_temporal_report_fails_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            aggregate = write_temporal_clipped_hep_aggregate_report(
                [tmp_path / "missing.json"],
                tmp_path / "aggregate",
            )

            self.assertEqual(aggregate["status"], "fail")
            self.assertEqual(aggregate["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(aggregate["selected_label_free_support_stress_candidate"])
            self.assertEqual(
                aggregate["evidence"]["failures"],
                [
                    {
                        "field": "decision_report",
                        "expected": "file exists",
                        "actual": "missing",
                        "path": str(tmp_path / "missing.json"),
                    }
                ],
            )

    def test_cross_scale_temporal_reports_pass_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_paths = []
            for scale in ("seed_smoke", "validation", "extended"):
                for backend in ("local", "colab"):
                    comparison_dir = tmp_path / f"{backend}_{scale}_comparison"
                    _write_temporal_clipped_comparison(
                        comparison_dir,
                        temporal_nonzero_loss=3.99,
                        entropy_nonzero_loss=4.01,
                        guided_nonzero_loss=3.9,
                        temporal_nonzero_delta=0.01,
                        temporal_nonzero_pinned_vs_repicked=0.02,
                    )
                    out_dir = (
                        tmp_path / f"temporal_clipped_hep_{scale}_{backend}_decision"
                    )
                    decision = write_temporal_clipped_hep_decision_report(
                        comparison_dir,
                        out_dir,
                    )
                    self.assertEqual(decision["status"], "pass")
                    report_paths.append(out_dir / "decision_report.json")

            aggregate = write_temporal_clipped_hep_cross_scale_aggregate_report(
                report_paths,
                tmp_path / "cross_scale_aggregate",
            )

            self.assertEqual(aggregate["status"], "pass")
            self.assertEqual(
                aggregate["decision"],
                SELECT_TEMPORAL_CLIPPED_HEP_CROSS_SCALE_AGGREGATE,
            )
            self.assertTrue(aggregate["selected_label_free_support_stress_candidate"])
            self.assertFalse(aggregate["promote_to_default_support_stress_mitigation"])
            self.assertEqual(aggregate["evidence"]["report_count"], 6)
            self.assertEqual(aggregate["evidence"]["scale_count"], 3)
            self.assertEqual(
                aggregate["evidence"]["accepted_temporal_report_count"],
                6,
            )
            self.assertTrue(
                (tmp_path / "cross_scale_aggregate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "cross_scale_aggregate" / "decision_report.md").is_file()
            )

    def test_cross_scale_aggregate_requires_each_scale_backend_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_seed_smoke_comparison"
            _write_temporal_clipped_comparison(
                comparison_dir,
                temporal_nonzero_loss=3.99,
                entropy_nonzero_loss=4.01,
                guided_nonzero_loss=3.9,
                temporal_nonzero_delta=0.01,
                temporal_nonzero_pinned_vs_repicked=0.02,
            )
            out_dir = tmp_path / "temporal_seed_smoke_local_decision"
            write_temporal_clipped_hep_decision_report(comparison_dir, out_dir)

            aggregate = write_temporal_clipped_hep_cross_scale_aggregate_report(
                [out_dir / "decision_report.json"],
                tmp_path / "cross_scale_aggregate",
            )

            self.assertEqual(aggregate["status"], "fail")
            self.assertEqual(aggregate["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                {
                    "field": "decision_report.scale_backend_pair",
                    "expected": "extended/colab",
                    "actual": "missing",
                },
                aggregate["evidence"]["failures"],
            )


class TemporalClippedHepPromotionGateReportTest(unittest.TestCase):
    def test_passing_cross_scale_aggregate_defines_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_paths = []
            for scale in ("seed_smoke", "validation", "extended"):
                for backend in ("local", "colab"):
                    comparison_dir = tmp_path / f"{backend}_{scale}_comparison"
                    _write_temporal_clipped_comparison(
                        comparison_dir,
                        temporal_nonzero_loss=3.99,
                        entropy_nonzero_loss=4.01,
                        guided_nonzero_loss=3.9,
                        temporal_nonzero_delta=0.01,
                        temporal_nonzero_pinned_vs_repicked=0.02,
                    )
                    out_dir = (
                        tmp_path / f"temporal_clipped_hep_{scale}_{backend}_decision"
                    )
                    write_temporal_clipped_hep_decision_report(comparison_dir, out_dir)
                    report_paths.append(out_dir / "decision_report.json")

            cross_scale = write_temporal_clipped_hep_cross_scale_aggregate_report(
                report_paths,
                tmp_path / "cross_scale_aggregate",
            )
            self.assertEqual(cross_scale["status"], "pass")

            gate = write_temporal_clipped_hep_promotion_gate_report(
                tmp_path / "cross_scale_aggregate" / "decision_report.json",
                tmp_path / "promotion_gate",
            )

            self.assertEqual(gate["status"], "pass")
            self.assertEqual(
                gate["decision"],
                DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
            )
            self.assertFalse(gate["promote_to_default_support_stress_mitigation"])
            self.assertEqual(len(gate["evidence"]["required_evidence"]), 2)
            self.assertEqual(
                gate["evidence"]["required_evidence"][0]["gate"],
                "larger_char_local_colab",
            )
            self.assertEqual(
                gate["evidence"]["required_evidence"][1]["gate"],
                "non_char_tokenized_local_colab",
            )
            self.assertTrue(
                (tmp_path / "promotion_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "promotion_gate" / "decision_report.md").is_file()
            )

    def test_missing_cross_scale_aggregate_blocks_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_report = tmp_path / "missing_cross_scale.json"

            gate = write_temporal_clipped_hep_promotion_gate_report(
                missing_report,
                tmp_path / "promotion_gate",
            )

            self.assertEqual(gate["status"], "fail")
            self.assertEqual(gate["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(gate["selected_label_free_support_stress_candidate"])
            self.assertEqual(
                gate["evidence"]["failures"],
                [
                    {
                        "field": "cross_scale_report",
                        "expected": "file exists",
                        "actual": "missing",
                        "path": str(missing_report),
                    }
                ],
            )


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


def _write_temporal_clipped_comparison(
    comparison_dir: Path,
    *,
    temporal_nonzero_loss: float,
    entropy_nonzero_loss: float,
    guided_nonzero_loss: float,
    temporal_nonzero_delta: float,
    temporal_nonzero_pinned_vs_repicked: float,
) -> None:
    comparison_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "invariant_count": 16,
            "failed_invariants": [],
            "artifact_invariants_passed": True,
            "artifact_invariant_count": 12,
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
                "entropy",
                settling_objective="prediction_entropy_gradient",
                nonzero_loss=entropy_nonzero_loss,
                nonzero_delta=0.01,
                nonzero_pinned_vs_repicked=0.002,
            ),
            _guided_clipped_run_entry(
                "temporal",
                settling_objective="temporal_consistency_gradient",
                nonzero_loss=temporal_nonzero_loss,
                nonzero_delta=temporal_nonzero_delta,
                nonzero_pinned_vs_repicked=temporal_nonzero_pinned_vs_repicked,
            ),
            _guided_clipped_run_entry(
                "guided",
                settling_objective="supervised_ce_gradient",
                nonzero_loss=guided_nonzero_loss,
                nonzero_delta=0.01,
                nonzero_pinned_vs_repicked=0.002,
            ),
        ],
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for run_id in ("clipped", "entropy", "temporal", "guided"):
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
