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
    SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
    DEFINE_POST_PROMOTION_RESIDUAL_LEARNING_GATE,
    KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT,
    CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION,
    CONTINUE_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    CONTINUE_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    CONTINUE_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
    CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
    DIAGNOSE_PC_RESIDUAL_OBJECTIVE,
    STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
    write_anchored_pc_residual_objective_decision_report,
    write_clipped_hep_decision_report,
    write_confidence_penalty_residual_objective_decision_report,
    write_label_smoothing_residual_objective_decision_report,
    write_focal_residual_objective_decision_report,
    write_margin_penalty_residual_objective_decision_report,
    write_guided_clipped_hep_decision_report,
    write_pc_residual_objective_diagnostics_report,
    write_pinned_support_decision_report,
    write_post_promotion_residual_learning_gate_report,
    write_residual_objective_gate_decision_report,
    write_temporal_clipped_hep_aggregate_report,
    write_temporal_clipped_hep_cross_scale_aggregate_report,
    write_temporal_clipped_hep_decision_report,
    write_temporal_clipped_hep_promotion_gate_report,
    write_temporal_clipped_hep_promotion_gate_satisfaction_report,
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


class TemporalClippedHepPromotionGateSatisfactionReportTest(unittest.TestCase):
    def test_larger_and_tokenized_local_colab_reports_satisfy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report = tmp_path / "promotion_gate" / "decision_report.json"
            gate_report.parent.mkdir(parents=True)
            gate_report.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report_paths = []
            for gate in ("larger", "token_larger"):
                for backend in ("local", "colab"):
                    comparison_dir = tmp_path / f"{backend}_{gate}_comparison"
                    _write_temporal_clipped_comparison(
                        comparison_dir,
                        temporal_nonzero_loss=3.99,
                        entropy_nonzero_loss=4.01,
                        guided_nonzero_loss=3.9,
                        temporal_nonzero_delta=0.01,
                        temporal_nonzero_pinned_vs_repicked=0.02,
                    )
                    out_dir = tmp_path / f"temporal_clipped_hep_{gate}_{backend}_decision"
                    decision = write_temporal_clipped_hep_decision_report(
                        comparison_dir,
                        out_dir,
                    )
                    self.assertEqual(decision["status"], "pass")
                    report_paths.append(out_dir / "decision_report.json")

            satisfaction = write_temporal_clipped_hep_promotion_gate_satisfaction_report(
                gate_report,
                report_paths,
                tmp_path / "promotion_gate_satisfaction",
            )

            self.assertEqual(satisfaction["status"], "pass")
            self.assertEqual(
                satisfaction["decision"],
                SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
            )
            self.assertTrue(satisfaction["promotion_gate_satisfied"])
            self.assertTrue(
                satisfaction["promote_to_default_support_stress_mitigation"]
            )
            self.assertEqual(satisfaction["evidence"]["report_count"], 4)
            self.assertEqual(
                satisfaction["evidence"]["accepted_temporal_report_count"],
                4,
            )
            self.assertTrue(
                (
                    tmp_path
                    / "promotion_gate_satisfaction"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "promotion_gate_satisfaction"
                    / "decision_report.md"
                ).is_file()
            )

    def test_missing_gate_pair_blocks_satisfaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report = tmp_path / "promotion_gate" / "decision_report.json"
            gate_report.parent.mkdir(parents=True)
            gate_report.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": DEFINE_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            comparison_dir = tmp_path / "local_larger_comparison"
            _write_temporal_clipped_comparison(
                comparison_dir,
                temporal_nonzero_loss=3.99,
                entropy_nonzero_loss=4.01,
                guided_nonzero_loss=3.9,
                temporal_nonzero_delta=0.01,
                temporal_nonzero_pinned_vs_repicked=0.02,
            )
            out_dir = tmp_path / "temporal_clipped_hep_larger_local_decision"
            write_temporal_clipped_hep_decision_report(comparison_dir, out_dir)

            satisfaction = write_temporal_clipped_hep_promotion_gate_satisfaction_report(
                gate_report,
                [out_dir / "decision_report.json"],
                tmp_path / "promotion_gate_satisfaction",
            )

            self.assertEqual(satisfaction["status"], "fail")
            self.assertEqual(satisfaction["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(satisfaction["promotion_gate_satisfied"])
            self.assertFalse(
                satisfaction["promote_to_default_support_stress_mitigation"]
            )
            self.assertIn(
                {
                    "field": "decision_report.gate_backend_pair",
                    "expected": "non_char_tokenized_local_colab/colab",
                    "actual": "missing",
                },
                satisfaction["evidence"]["failures"],
            )


class PostPromotionResidualLearningGateReportTest(unittest.TestCase):
    def test_passing_promotion_satisfaction_defines_residual_learning_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            satisfaction_report = tmp_path / "satisfaction" / "decision_report.json"
            satisfaction_report.parent.mkdir(parents=True)
            satisfaction_report.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                        "promotion_gate_satisfied": True,
                        "promote_to_default_support_stress_mitigation": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "char_smoke_hep_support_stress.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  experiment_id: char_smoke_hep_support_stress",
                        "inference:",
                        "  hep_update_clip_norm: 0.01",
                        "  hep_settling_objective: temporal_consistency_gradient",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            gate = write_post_promotion_residual_learning_gate_report(
                satisfaction_report,
                config_path,
                tmp_path / "post_promotion_gate",
            )

            self.assertEqual(gate["status"], "pass")
            self.assertEqual(
                gate["decision"],
                DEFINE_POST_PROMOTION_RESIDUAL_LEARNING_GATE,
            )
            self.assertTrue(
                gate["promoted_temporal_support_stress_default_confirmed"]
            )
            self.assertFalse(gate["promote_residual_learning_method"])
            self.assertEqual(len(gate["evidence"]["required_evidence"]), 3)
            self.assertEqual(
                gate["evidence"]["required_evidence"][0]["gate"],
                "pc_residual_objective_under_promoted_temporal_default",
            )
            self.assertTrue(
                (tmp_path / "post_promotion_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "post_promotion_gate" / "decision_report.md").is_file()
            )

    def test_unpromoted_default_config_blocks_residual_learning_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            satisfaction_report = tmp_path / "satisfaction" / "decision_report.json"
            satisfaction_report.parent.mkdir(parents=True)
            satisfaction_report.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": SATISFY_TEMPORAL_CLIPPED_HEP_PROMOTION_GATE,
                        "promotion_gate_satisfied": True,
                        "promote_to_default_support_stress_mitigation": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "char_smoke_hep_support_stress.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  experiment_id: char_smoke_hep_support_stress",
                        "inference:",
                        "  hep_update_clip_norm: 0.01",
                        "  hep_settling_objective: residual_adapter",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            gate = write_post_promotion_residual_learning_gate_report(
                satisfaction_report,
                config_path,
                tmp_path / "post_promotion_gate",
            )

            self.assertEqual(gate["status"], "fail")
            self.assertEqual(gate["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(
                gate["promoted_temporal_support_stress_default_confirmed"]
            )
            self.assertIn(
                {
                    "field": "default_support_stress_config.hep_settling_objective",
                    "expected": "temporal_consistency_gradient",
                    "actual": "residual_adapter",
                    "path": str(config_path),
                },
                gate["evidence"]["failures"],
            )


class ResidualObjectiveGateDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_gate_keeps_supervised_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_objective_gate_comparison"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_residual_objective_gate_decision_report(
                comparison_dirs,
                tmp_path / "objective_gate_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT,
            )
            self.assertFalse(report["continue_pc_residual_objective_validation"])
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["backend_count"], 2)
            self.assertEqual(report["evidence"]["pc_ce_win_count"], 0)
            self.assertTrue(
                (
                    tmp_path
                    / "objective_gate_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "objective_gate_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_pc_ce_win_continues_pc_validation_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_objective_gate_comparison"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.50,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_residual_objective_gate_decision_report(
                comparison_dirs,
                tmp_path / "objective_gate_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(report["continue_pc_residual_objective_validation"])
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["pc_ce_win_count"], 2)

    def test_enabled_support_stress_preset_blocks_objective_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_objective_gate_comparison"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.50,
                support_stress_preset=True,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_residual_objective_gate_decision_report(
                [comparison_dir],
                tmp_path / "objective_gate_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["continue_pc_residual_objective_validation"])
            self.assertIn(
                {
                    "field": "run.supervised.support_stress_preset",
                    "expected": False,
                    "actual": True,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )
            self.assertIn(
                {
                    "field": "comparison.backend",
                    "expected": "colab",
                    "actual": "missing",
                },
                report["evidence"]["failures"],
            )


class PcResidualObjectiveDiagnosticsReportTest(unittest.TestCase):
    def test_valid_local_and_colab_diagnostics_quantify_pc_ce_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_objective_gate_comparison"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_pc_residual_objective_diagnostics_report(
                comparison_dirs,
                tmp_path / "pc_objective_diagnostics",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], DIAGNOSE_PC_RESIDUAL_OBJECTIVE)
            self.assertEqual(report["evidence"]["pc_worse_ce_backend_count"], 2)
            self.assertAlmostEqual(
                report["evidence"]["mean_pc_minus_supervised_best_hep_loss"],
                0.02,
            )
            self.assertAlmostEqual(
                report["evidence"]["mean_pc_best_hep_loss_improvement_from_alpha0"],
                0.001,
            )
            self.assertTrue(
                (
                    tmp_path
                    / "pc_objective_diagnostics"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "pc_objective_diagnostics"
                    / "decision_report.md"
                ).is_file()
            )

    def test_diagnostics_fail_without_matching_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_objective_gate_comparison"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_pc_residual_objective_diagnostics_report(
                [comparison_dir],
                tmp_path / "pc_objective_diagnostics",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                {
                    "field": "comparison.backend",
                    "expected": "colab",
                    "actual": "missing",
                },
                report["evidence"]["failures"],
            )


class AnchoredPcResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_anchor_stops_pc_validation_without_ce_win(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_anchor_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    anchored_pc_best_hep_loss=3.5801,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_anchored_pc_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "anchored_pc_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertFalse(report["continue_pc_residual_objective_validation"])
            self.assertIsNone(report["selected_pc_residual_objective_variant"])
            self.assertEqual(report["evidence"]["anchored_pc_ce_win_count"], 0)
            self.assertGreater(
                report["evidence"]["mean_pc_to_anchored_gap_reduction"],
                0.0,
            )
            self.assertTrue(
                (
                    tmp_path
                    / "anchored_pc_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "anchored_pc_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_anchor_ce_win_continues_pc_validation_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_anchor_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    anchored_pc_best_hep_loss=3.57,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_anchored_pc_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "anchored_pc_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_PC_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(report["continue_pc_residual_objective_validation"])
            self.assertEqual(
                report["selected_pc_residual_objective_variant"],
                "pc_logit_mse_ce_anchor",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["anchored_pc_ce_win_count"], 2)

    def test_missing_anchored_run_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_anchor_objective_gate"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_anchored_pc_residual_objective_decision_report(
                [comparison_dir],
                tmp_path / "anchored_pc_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["continue_pc_residual_objective_validation"])
            self.assertIn(
                {
                    "field": "comparison.runs.pc_logit_mse_ce_anchor",
                    "expected": "one run",
                    "actual": 0,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )


class ConfidencePenaltyResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_confidence_penalty_stops_variant_without_ce_win(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_confidence_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    confidence_penalty_best_hep_loss=3.5801,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_confidence_penalty_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "confidence_penalty_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertFalse(
                report["continue_confidence_penalty_residual_objective_validation"]
            )
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(report["evidence"]["confidence_penalty_ce_win_count"], 0)
            self.assertLess(
                report[
                    "evidence"
                ]["mean_confidence_penalty_minus_supervised_final_residual_loss"],
                0.0,
            )
            self.assertTrue(
                (
                    tmp_path
                    / "confidence_penalty_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "confidence_penalty_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_confidence_penalty_ce_win_continues_variant_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_confidence_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    confidence_penalty_best_hep_loss=3.57,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_confidence_penalty_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "confidence_penalty_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(
                report["continue_confidence_penalty_residual_objective_validation"]
            )
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_confidence_penalty",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["confidence_penalty_ce_win_count"], 2)

    def test_missing_confidence_penalty_run_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_confidence_objective_gate"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_confidence_penalty_residual_objective_decision_report(
                [comparison_dir],
                tmp_path / "confidence_penalty_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(
                report["continue_confidence_penalty_residual_objective_validation"]
            )
            self.assertIn(
                {
                    "field": "comparison.runs.supervised_ce_confidence_penalty",
                    "expected": "one run",
                    "actual": 0,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )


class MarginPenaltyResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_margin_penalty_stops_variant_without_ce_win(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_margin_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    margin_penalty_best_hep_loss=3.5801,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_margin_penalty_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "margin_penalty_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertFalse(
                report["continue_margin_penalty_residual_objective_validation"]
            )
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(report["evidence"]["margin_penalty_ce_win_count"], 0)
            self.assertTrue(
                (tmp_path / "margin_penalty_decision" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "margin_penalty_decision" / "decision_report.md").is_file()
            )

    def test_margin_penalty_ce_win_continues_variant_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_margin_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    margin_penalty_best_hep_loss=3.57,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_margin_penalty_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "margin_penalty_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(
                report["continue_margin_penalty_residual_objective_validation"]
            )
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_margin_penalty",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["margin_penalty_ce_win_count"], 2)

    def test_missing_margin_penalty_run_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_margin_objective_gate"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_margin_penalty_residual_objective_decision_report(
                [comparison_dir],
                tmp_path / "margin_penalty_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(
                report["continue_margin_penalty_residual_objective_validation"]
            )
            self.assertIn(
                {
                    "field": "comparison.runs.supervised_ce_margin_penalty",
                    "expected": "one run",
                    "actual": 0,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )


class LabelSmoothingResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_label_smoothing_stops_variant_without_ce_win(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_label_smoothing_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    label_smoothing_best_hep_loss=3.5801,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_label_smoothing_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "label_smoothing_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertFalse(
                report["continue_label_smoothing_residual_objective_validation"]
            )
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(report["evidence"]["label_smoothing_ce_win_count"], 0)
            self.assertTrue(
                (
                    tmp_path
                    / "label_smoothing_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "label_smoothing_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_label_smoothing_ce_win_continues_variant_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_label_smoothing_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    label_smoothing_best_hep_loss=3.57,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_label_smoothing_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "label_smoothing_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(
                report["continue_label_smoothing_residual_objective_validation"]
            )
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_label_smoothing",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["label_smoothing_ce_win_count"], 2)

    def test_missing_label_smoothing_run_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_label_smoothing_objective_gate"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_label_smoothing_residual_objective_decision_report(
                [comparison_dir],
                tmp_path / "label_smoothing_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(
                report["continue_label_smoothing_residual_objective_validation"]
            )
            self.assertIn(
                {
                    "field": "comparison.runs.supervised_ce_label_smoothing",
                    "expected": "one run",
                    "actual": 0,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )


class FocalResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_valid_local_and_colab_focal_stops_variant_without_ce_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_focal_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    focal_best_hep_loss=3.5801,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_focal_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "focal_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION)
            self.assertFalse(report["continue_focal_residual_objective_validation"])
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(report["evidence"]["focal_ce_win_count"], 0)
            self.assertTrue((tmp_path / "focal_decision" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "focal_decision" / "decision_report.md").is_file())

    def test_focal_ce_win_continues_variant_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend in ("local", "colab"):
                comparison_dir = tmp_path / f"{backend}_focal_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    focal_best_hep_loss=3.57,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_focal_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "focal_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(report["continue_focal_residual_objective_validation"])
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_focal",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["focal_ce_win_count"], 2)

    def test_mixed_focal_evidence_stops_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for backend, focal_loss in (
                ("local", 3.57),
                ("colab", 3.5801),
            ):
                comparison_dir = tmp_path / f"{backend}_focal_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    focal_best_hep_loss=focal_loss,
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_focal_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "focal_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION)
            self.assertFalse(report["continue_focal_residual_objective_validation"])
            self.assertEqual(report["evidence"]["focal_ce_win_count"], 1)

    def test_missing_focal_run_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "local_focal_objective_gate"
            _write_residual_objective_gate_comparison(
                comparison_dir,
                pc_best_hep_loss=3.60,
                support_stress_preset=False,
            )
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_focal_residual_objective_decision_report(
                [comparison_dir],
                tmp_path / "focal_decision",
                artifact_check_paths=[artifact_check],
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["continue_focal_residual_objective_validation"])
            self.assertIn(
                {
                    "field": "comparison.runs.supervised_ce_focal",
                    "expected": "one run",
                    "actual": 0,
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
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


def _write_residual_objective_gate_comparison(
    comparison_dir: Path,
    *,
    pc_best_hep_loss: float,
    support_stress_preset: bool,
    anchored_pc_best_hep_loss: float | None = None,
    confidence_penalty_best_hep_loss: float | None = None,
    margin_penalty_best_hep_loss: float | None = None,
    label_smoothing_best_hep_loss: float | None = None,
    focal_best_hep_loss: float | None = None,
) -> None:
    comparison_dir.mkdir(parents=True)
    runs = [
        _residual_objective_run_entry(
            "supervised",
            residual_objective="supervised_ce",
            initial_loss=3.7,
            final_loss=3.6,
            best_hep_loss=3.58,
            support_stress_preset=support_stress_preset,
        ),
        _residual_objective_run_entry(
            "pc",
            residual_objective="pc_logit_mse",
            initial_loss=0.03,
            final_loss=0.02,
            best_hep_loss=pc_best_hep_loss,
            support_stress_preset=support_stress_preset,
        ),
    ]
    if anchored_pc_best_hep_loss is not None:
        runs.append(
            _residual_objective_run_entry(
                "anchored_pc",
                residual_objective="pc_logit_mse_ce_anchor",
                initial_loss=0.40,
                final_loss=0.38,
                best_hep_loss=anchored_pc_best_hep_loss,
                support_stress_preset=support_stress_preset,
            )
        )
    if confidence_penalty_best_hep_loss is not None:
        runs.append(
            _residual_objective_run_entry(
                "confidence_penalty",
                residual_objective="supervised_ce_confidence_penalty",
                initial_loss=3.69,
                final_loss=3.55,
                best_hep_loss=confidence_penalty_best_hep_loss,
                support_stress_preset=support_stress_preset,
            )
        )
    if margin_penalty_best_hep_loss is not None:
        runs.append(
            _residual_objective_run_entry(
                "margin_penalty",
                residual_objective="supervised_ce_margin_penalty",
                initial_loss=3.70,
                final_loss=3.56,
                best_hep_loss=margin_penalty_best_hep_loss,
                support_stress_preset=support_stress_preset,
            )
        )
    if label_smoothing_best_hep_loss is not None:
        runs.append(
            _residual_objective_run_entry(
                "label_smoothing",
                residual_objective="supervised_ce_label_smoothing",
                initial_loss=3.70,
                final_loss=3.56,
                best_hep_loss=label_smoothing_best_hep_loss,
                support_stress_preset=support_stress_preset,
            )
        )
    if focal_best_hep_loss is not None:
        runs.append(
            _residual_objective_run_entry(
                "focal",
                residual_objective="supervised_ce_focal",
                initial_loss=3.40,
                final_loss=3.30,
                best_hep_loss=focal_best_hep_loss,
                support_stress_preset=support_stress_preset,
            )
        )
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
        "runs": runs,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for run in runs:
        run_id = str(run["experiment_id"])
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


def _residual_objective_run_entry(
    experiment_id: str,
    *,
    residual_objective: str,
    initial_loss: float,
    final_loss: float,
    best_hep_loss: float,
    support_stress_preset: bool,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "config_path": f"configs/{experiment_id}.yaml",
        "status": "ok",
        "dataset": "tiny_shakespeare_char",
        "residual_objective": residual_objective,
        "initial_residual_loss": initial_loss,
        "final_residual_loss": final_loss,
        "residual_loss_delta": final_loss - initial_loss,
        "residual_loss_ratio": final_loss / initial_loss,
        "training_steps": 25,
        "support_stress": True,
        "support_stress_preset": support_stress_preset,
        "hep_update_clip_norm": 0.01,
        "hep_settling_objective": "temporal_consistency_gradient",
        "invariants": {
            "zero_init_identity": True,
            "frozen_base_unchanged": True,
            "hep_alpha_0_equivalence": True,
            "residual_parameters_updated": True,
        },
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": best_hep_loss + 0.001,
                "max_logit_delta_from_ordinary": 0.0,
                "support_change_fraction": 0.0,
                "pinned_vs_repicked_logit_delta": 0.0,
            },
            {
                "alpha": 1.0,
                "loss": best_hep_loss,
                "max_logit_delta_from_ordinary": 0.001,
                "support_change_fraction": 0.0,
                "pinned_vs_repicked_logit_delta": 0.0,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
