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
    DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE,
    SATISFY_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE,
    DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE,
    RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC,
    CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION,
    DEFINE_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE,
    CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION,
    DEFINE_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE,
    SATISFY_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE,
    DEFINE_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
    SATISFY_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
    DEFINE_POST_SUPPORT_WIDTH_RESIDUAL_LEARNING_GATE,
    DEFINE_RESIDUAL_CAPACITY_REPEAT_GATE,
    STOP_RESIDUAL_CAPACITY_VALIDATION,
    RUN_COLAB_SUPPORT_WIDTH_DECONFOUNDING_AUDIT,
    DIAGNOSE_EXHAUSTIVE_SUPPORT_AUDIT,
    DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
    SATISFY_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
    DIAGNOSE_PC_RESIDUAL_OBJECTIVE,
    STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
    STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
    CONTINUE_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
    write_anchored_pc_residual_objective_decision_report,
    write_clipped_hep_decision_report,
    write_confidence_penalty_residual_objective_decision_report,
    write_label_smoothing_residual_objective_decision_report,
    write_focal_residual_objective_decision_report,
    write_focal_residual_objective_promotion_gate_report,
    write_focal_residual_objective_promotion_gate_satisfaction_report,
    write_temporal_consistency_residual_objective_decision_report,
    write_residual_learning_next_direction_report,
    write_residual_capacity_support_diagnostic_gate_report,
    write_residual_capacity_support_diagnostic_decision_report,
    write_residual_capacity_support_diagnostic_colab_decision_report,
    write_residual_support_width_validation_gate_report,
    write_residual_support_width_validation_decision_report,
    write_residual_support_width_repeat_gate_report,
    write_residual_support_width_repeat_decision_report,
    write_residual_support_width_promotion_gate_report,
    write_residual_support_width_promotion_gate_satisfaction_report,
    write_post_support_width_residual_learning_gate_report,
    write_post_support_width_residual_capacity_decision_report,
    write_support_width_deconfounding_audit_report,
    write_exhaustive_support_audit_report,
    write_contextual_support_router_decision_report,
    write_contextual_support_router_promotion_gate_report,
    write_contextual_support_router_promotion_gate_satisfaction_report,
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

    def test_passing_focal_decision_defines_promotion_gate(self) -> None:
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

            decision = write_focal_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "focal_decision",
                artifact_check_paths=artifact_checks,
            )
            self.assertEqual(
                decision["decision"],
                CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
            )

            gate = write_focal_residual_objective_promotion_gate_report(
                tmp_path / "focal_decision" / "decision_report.json",
                tmp_path / "focal_promotion_gate",
            )

            self.assertEqual(gate["status"], "pass")
            self.assertEqual(
                gate["decision"],
                DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE,
            )
            self.assertFalse(gate["promote_residual_learning_method"])
            self.assertEqual(
                gate["evidence"]["required_evidence"][0]["gate"],
                "char_xxlarge_seed2_local_colab",
            )
            self.assertEqual(
                gate["evidence"]["required_evidence"][1]["gate"],
                "token_larger_seed2_local_colab",
            )
            self.assertTrue(
                (tmp_path / "focal_promotion_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "focal_promotion_gate" / "decision_report.md").is_file()
            )

    def test_missing_focal_decision_blocks_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_report = tmp_path / "missing_focal_decision.json"

            gate = write_focal_residual_objective_promotion_gate_report(
                missing_report,
                tmp_path / "focal_promotion_gate",
            )

            self.assertEqual(gate["status"], "fail")
            self.assertEqual(gate["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(gate["selected_residual_objective_variant"])
            self.assertIn(
                {
                    "field": "focal_decision_report",
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(missing_report),
                },
                gate["evidence"]["failures"],
            )

    def test_seed2_gate_satisfaction_stops_focal_when_token_repeat_loses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report = _write_passing_focal_promotion_gate(tmp_path)
            comparison_dirs = []
            artifact_checks = []
            for name, focal_loss in (
                ("char_xxlarge_focal_temporal_clipped_objective_gate_seed2_local", 3.57),
                (
                    "colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2",
                    3.57,
                ),
                ("token_larger_focal_temporal_clipped_objective_gate_seed2_local", 3.59),
                (
                    "colab_token_larger_focal_temporal_clipped_objective_gate_seed2",
                    3.59,
                ),
            ):
                comparison_dir = tmp_path / name
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

            report = write_focal_residual_objective_promotion_gate_satisfaction_report(
                gate_report,
                comparison_dirs,
                tmp_path / "focal_gate_satisfaction",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION)
            self.assertFalse(report["promotion_gate_satisfied"])
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(report["evidence"]["comparison_count"], 4)
            self.assertEqual(report["evidence"]["focal_ce_win_count"], 2)
            self.assertTrue(
                (
                    tmp_path
                    / "focal_gate_satisfaction"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "focal_gate_satisfaction"
                    / "decision_report.md"
                ).is_file()
            )

    def test_seed2_gate_satisfaction_promotes_focal_when_all_repeats_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report = _write_passing_focal_promotion_gate(tmp_path)
            comparison_dirs = []
            artifact_checks = []
            for name in (
                "char_xxlarge_focal_temporal_clipped_objective_gate_seed2_local",
                "colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2",
                "token_larger_focal_temporal_clipped_objective_gate_seed2_local",
                "colab_token_larger_focal_temporal_clipped_objective_gate_seed2",
            ):
                comparison_dir = tmp_path / name
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

            report = write_focal_residual_objective_promotion_gate_satisfaction_report(
                gate_report,
                comparison_dirs,
                tmp_path / "focal_gate_satisfaction",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                SATISFY_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE,
            )
            self.assertTrue(report["promotion_gate_satisfied"])
            self.assertTrue(report["promote_residual_learning_method"])
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_focal",
            )
            self.assertEqual(report["evidence"]["focal_ce_win_count"], 4)


class TemporalConsistencyResidualObjectiveDecisionReportTest(unittest.TestCase):
    def test_tiny_temporal_consistency_margin_stops_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for scale in ("validation", "extended"):
                comparison_dir = tmp_path / f"{scale}_temporal_consistency_weight_sweep_temporal_clipped_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    temporal_consistency_best_hep_losses=[3.57999],
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_temporal_consistency_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "temporal_consistency_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertFalse(
                report["continue_temporal_consistency_residual_objective_validation"]
            )
            self.assertIsNone(report["selected_residual_objective_variant"])
            self.assertEqual(
                report["evidence"]["temporal_consistency_clear_margin_count"],
                0,
            )

    def test_material_temporal_consistency_margin_continues_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dirs = []
            artifact_checks = []
            for scale in ("validation", "extended"):
                comparison_dir = tmp_path / f"{scale}_temporal_consistency_weight_sweep_temporal_clipped_objective_gate"
                _write_residual_objective_gate_comparison(
                    comparison_dir,
                    pc_best_hep_loss=3.60,
                    temporal_consistency_best_hep_losses=[3.57],
                    support_stress_preset=False,
                )
                artifact_check = comparison_dir / "artifact_check_local.json"
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                comparison_dirs.append(comparison_dir)
                artifact_checks.append(artifact_check)

            report = write_temporal_consistency_residual_objective_decision_report(
                comparison_dirs,
                tmp_path / "temporal_consistency_decision",
                artifact_check_paths=artifact_checks,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
            )
            self.assertTrue(
                report["continue_temporal_consistency_residual_objective_validation"]
            )
            self.assertEqual(
                report["selected_residual_objective_variant"],
                "supervised_ce_temporal_consistency",
            )


class ResidualLearningNextDirectionReportTest(unittest.TestCase):
    def test_stopped_objective_reports_select_capacity_support_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports = _write_stopped_residual_objective_reports(tmp_path)

            report = write_residual_learning_next_direction_report(
                reports,
                tmp_path / "next_direction",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "residual_capacity_support_diagnostic",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(report["evidence"]["report_count"], 7)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertTrue((tmp_path / "next_direction" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "next_direction" / "decision_report.md").is_file())

    def test_missing_stop_report_blocks_direction_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports = _write_stopped_residual_objective_reports(tmp_path)
            reports = reports[:-1]

            report = write_residual_learning_next_direction_report(
                reports,
                tmp_path / "next_direction",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "decision_report.kind",
                    "expected": "temporal_consistency_residual_objective_decision",
                    "actual": "missing",
                },
                report["evidence"]["failures"],
            )


class ResidualCapacitySupportDiagnosticGateReportTest(unittest.TestCase):
    def test_valid_config_matrix_defines_local_diagnostic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            next_direction = _write_next_direction_report(tmp_path, status="pass")
            configs = _write_capacity_support_configs(tmp_path)

            report = write_residual_capacity_support_diagnostic_gate_report(
                next_direction,
                configs,
                tmp_path / "capacity_support_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "residual_capacity_support_diagnostic",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(report["evidence"]["config_count"], 4)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertIn("relaleap.experiments.compare", report["commands"]["compare"])
            self.assertIn(
                "relaleap.experiments.check_artifacts",
                report["commands"]["check_artifacts"],
            )
            self.assertTrue(
                (tmp_path / "capacity_support_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "capacity_support_gate" / "decision_report.md").is_file()
            )

    def test_failed_next_direction_report_blocks_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            next_direction = _write_next_direction_report(tmp_path, status="fail")
            configs = _write_capacity_support_configs(tmp_path)

            report = write_residual_capacity_support_diagnostic_gate_report(
                next_direction,
                configs,
                tmp_path / "capacity_support_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "next_direction_report.status",
                    "expected": "pass",
                    "actual": "fail",
                    "path": str(next_direction),
                },
                report["evidence"]["failures"],
            )


class ResidualCapacitySupportDiagnosticDecisionReportTest(unittest.TestCase):
    def test_support_width_winner_selects_matching_colab_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_capacity_support_comparison(comparison_dir)
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_residual_capacity_support_diagnostic_decision_report(
                comparison_dir,
                tmp_path / "capacity_support_decision",
                artifact_check_path=artifact_check,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                RUN_COLAB_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "colab_residual_capacity_support_diagnostic",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertLess(
                report["evidence"]["support_minus_baseline_best_hep_loss"],
                0.0,
            )
            self.assertEqual(report["evidence"]["best_variant"]["variant"], "support_width")
            self.assertEqual(report["evidence"]["accepted_support_width_alpha"]["alpha"], 1.0)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertTrue(
                (tmp_path / "capacity_support_decision" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "capacity_support_decision" / "decision_report.md").is_file()
            )

    def test_failed_artifact_check_blocks_colab_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            _write_capacity_support_comparison(comparison_dir)
            artifact_check = comparison_dir / "artifact_check_local.json"
            artifact_check.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_residual_capacity_support_diagnostic_decision_report(
                comparison_dir,
                tmp_path / "capacity_support_decision",
                artifact_check_path=artifact_check,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "artifact_check.status",
                    "expected": "pass",
                    "actual": "fail",
                    "path": str(comparison_dir),
                },
                report["evidence"]["failures"],
            )

    def test_matching_colab_evidence_continues_support_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_capacity_support_comparison(local_dir)
            _write_capacity_support_comparison(colab_dir)
            local_check = local_dir / "artifact_check_local.json"
            colab_check = colab_dir / "artifact_check_local.json"
            for artifact_check in (local_check, colab_check):
                artifact_check.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )

            report = write_residual_capacity_support_diagnostic_colab_decision_report(
                (local_dir, colab_dir),
                tmp_path / "capacity_support_colab_decision",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "residual_capacity_support_validation_gate",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["evidence"]["backend_count"], 2)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertTrue(
                (
                    tmp_path
                    / "capacity_support_colab_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "capacity_support_colab_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_failed_colab_artifact_check_blocks_support_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_capacity_support_comparison(local_dir)
            _write_capacity_support_comparison(colab_dir)
            local_check = local_dir / "artifact_check_local.json"
            colab_check = colab_dir / "artifact_check_local.json"
            local_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )
            colab_check.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_residual_capacity_support_diagnostic_colab_decision_report(
                (local_dir, colab_dir),
                tmp_path / "capacity_support_colab_decision",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "artifact_check.status",
                    "expected": "pass",
                    "actual": "fail",
                    "path": str(colab_dir),
                },
                report["evidence"]["failures"],
            )


class ResidualSupportWidthValidationGateReportTest(unittest.TestCase):
    def test_valid_larger_char_token_matrix_defines_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            colab_decision = _write_capacity_support_colab_decision(
                tmp_path,
                status="pass",
                decision=CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION,
            )
            configs = _write_support_width_validation_configs(tmp_path)

            report = write_residual_support_width_validation_gate_report(
                colab_decision,
                configs,
                tmp_path / "support_width_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_RESIDUAL_SUPPORT_WIDTH_VALIDATION_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "support_width_larger_char_token_validation",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(report["evidence"]["config_count"], 4)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertIn("relaleap.experiments.compare", report["commands"]["compare"])
            self.assertIn(
                "relaleap.experiments.check_artifacts",
                report["commands"]["check_artifacts"],
            )
            self.assertTrue(
                (tmp_path / "support_width_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "support_width_gate" / "decision_report.md").is_file()
            )

    def test_failed_colab_decision_blocks_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            colab_decision = _write_capacity_support_colab_decision(
                tmp_path,
                status="fail",
                decision=INSUFFICIENT_EVIDENCE,
            )
            configs = _write_support_width_validation_configs(tmp_path)

            report = write_residual_support_width_validation_gate_report(
                colab_decision,
                configs,
                tmp_path / "support_width_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "colab_decision_report.status",
                    "expected": "pass",
                    "actual": "fail",
                    "path": str(colab_decision),
                },
                report["evidence"]["failures"],
            )


class ResidualSupportWidthValidationDecisionReportTest(unittest.TestCase):
    def test_matching_local_colab_width_improvements_continue_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir)
            _write_support_width_validation_comparison(colab_dir)

            report = write_residual_support_width_validation_decision_report(
                (local_dir, colab_dir),
                tmp_path / "support_width_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "support_width_repeat_or_capacity_interaction_validation",
            )
            self.assertFalse(report["promote_residual_learning_method"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(report["evidence"]["backend_count"], 2)
            self.assertEqual(report["evidence"]["failures"], [])
            for backend in report["evidence"]["backends"]:
                for scale in ("larger_char", "tokenized"):
                    self.assertTrue(
                        backend["scales"][scale][
                            "support_beats_baseline_alpha0_loss"
                        ]
                    )
            self.assertTrue(
                (tmp_path / "support_width_decision" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "support_width_decision" / "decision_report.md").is_file()
            )

    def test_failed_colab_artifact_check_blocks_width_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir)
            _write_support_width_validation_comparison(colab_dir)
            (colab_dir / "artifact_check_local.json").write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_residual_support_width_validation_decision_report(
                (local_dir, colab_dir),
                tmp_path / "support_width_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "artifact_check.status",
                    "expected": "pass",
                    "actual": "fail",
                    "path": str(colab_dir),
                },
                report["evidence"]["failures"],
            )


class ResidualSupportWidthRepeatGateReportTest(unittest.TestCase):
    def test_seed2_matrix_defines_repeat_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            validation_decision = _write_support_width_validation_decision(
                tmp_path,
                status="pass",
                decision=CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION,
            )
            configs = _write_support_width_validation_configs(tmp_path, seed=2)

            report = write_residual_support_width_repeat_gate_report(
                validation_decision,
                configs,
                tmp_path / "support_width_repeat_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], DEFINE_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE)
            self.assertEqual(
                report["selected_next_direction"],
                "support_width_larger_char_token_seed2_repeat",
            )
            self.assertFalse(report["promote_support_width_default"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(
                report["default_support_stress_mitigation"],
                "temporal_clipped_hep",
            )
            self.assertEqual(report["evidence"]["config_count"], 4)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertIn("seed2", report["commands"]["compare"])
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_repeat_gate"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_repeat_gate"
                    / "decision_report.md"
                ).is_file()
            )

    def test_seed1_matrix_blocks_repeat_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            validation_decision = _write_support_width_validation_decision(
                tmp_path,
                status="pass",
                decision=CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION,
            )
            configs = _write_support_width_validation_configs(tmp_path, seed=1)

            report = write_residual_support_width_repeat_gate_report(
                validation_decision,
                configs,
                tmp_path / "support_width_repeat_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "config.seed",
                    "expected": 2,
                    "actual": 1,
                    "path": str(configs[0]),
                },
                report["evidence"]["failures"],
            )


class ResidualSupportWidthRepeatDecisionReportTest(unittest.TestCase):
    def test_matching_seed2_local_colab_repeats_satisfy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir, seed=2)
            _write_support_width_validation_comparison(colab_dir, seed=2)

            report = write_residual_support_width_repeat_decision_report(
                (local_dir, colab_dir),
                tmp_path / "support_width_repeat_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                SATISFY_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "residual_support_width_promotion_gate",
            )
            self.assertFalse(report["promote_support_width_default"])
            self.assertEqual(report["evidence"]["backend_count"], 2)
            self.assertEqual(report["evidence"]["failures"], [])
            for backend in report["evidence"]["backends"]:
                for scale in ("larger_char", "tokenized"):
                    self.assertTrue(
                        backend["scales"][scale][
                            "support_beats_baseline_final_loss"
                        ]
                    )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_repeat_decision"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_repeat_decision"
                    / "decision_report.md"
                ).is_file()
            )

    def test_non_seed2_repeat_identity_blocks_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir)
            _write_support_width_validation_comparison(colab_dir, seed=2)

            report = write_residual_support_width_repeat_decision_report(
                (local_dir, colab_dir),
                tmp_path / "support_width_repeat_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertTrue(
                any(
                    failure["field"].startswith("local.")
                    and failure["field"].endswith(".seed")
                    for failure in report["evidence"]["failures"]
                )
            )


class ResidualSupportWidthPromotionGateReportTest(unittest.TestCase):
    def test_passing_repeat_decision_defines_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repeat_decision = _write_support_width_repeat_decision(
                tmp_path,
                status="pass",
                decision=SATISFY_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE,
                promote_support_width_default=False,
            )

            report = write_residual_support_width_promotion_gate_report(
                repeat_decision,
                tmp_path / "support_width_promotion_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "support_width_seed3_promotion_gate_evidence",
            )
            self.assertFalse(report["promote_support_width_default"])
            self.assertEqual(report["default_residual_objective"], "supervised_ce")
            self.assertEqual(
                report["default_support_stress_mitigation"],
                "temporal_clipped_hep",
            )
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertEqual(len(report["evidence"]["required_evidence"]), 2)
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_promotion_gate"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_promotion_gate"
                    / "decision_report.md"
                ).is_file()
            )

    def test_repeat_decision_that_promotes_default_blocks_gate_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repeat_decision = _write_support_width_repeat_decision(
                tmp_path,
                status="pass",
                decision=SATISFY_RESIDUAL_SUPPORT_WIDTH_REPEAT_GATE,
                promote_support_width_default=True,
            )

            report = write_residual_support_width_promotion_gate_report(
                repeat_decision,
                tmp_path / "support_width_promotion_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "repeat_decision_report.promote_support_width_default",
                    "expected": False,
                    "actual": True,
                    "path": str(repeat_decision),
                },
                report["evidence"]["failures"],
            )


class ResidualSupportWidthPromotionGateSatisfactionReportTest(unittest.TestCase):
    def test_matching_seed3_local_colab_evidence_satisfies_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            promotion_gate = _write_support_width_promotion_gate(
                tmp_path,
                status="pass",
                decision=DEFINE_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
            )
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir, seed=3)
            _write_support_width_validation_comparison(colab_dir, seed=3)

            report = write_residual_support_width_promotion_gate_satisfaction_report(
                promotion_gate,
                (local_dir, colab_dir),
                tmp_path / "support_width_promotion_gate_satisfaction",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                SATISFY_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
            )
            self.assertTrue(report["promotion_gate_satisfied"])
            self.assertTrue(report["promote_support_width_default"])
            self.assertEqual(report["selected_support_width_top_k"], 2)
            self.assertEqual(
                report["selected_next_direction"],
                "promote_default_residual_support_width_top_k_2",
            )
            self.assertEqual(report["evidence"]["backend_count"], 2)
            self.assertEqual(report["evidence"]["failures"], [])
            for backend in report["evidence"]["backends"]:
                for scale in ("larger_char", "tokenized"):
                    self.assertTrue(
                        backend["scales"][scale][
                            "support_beats_baseline_alpha0_loss"
                        ]
                    )
                    self.assertTrue(
                        backend["scales"][scale][
                            "support_beats_baseline_final_loss"
                        ]
                    )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_promotion_gate_satisfaction"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "support_width_promotion_gate_satisfaction"
                    / "decision_report.md"
                ).is_file()
            )

    def test_non_seed3_identity_blocks_promotion_gate_satisfaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            promotion_gate = _write_support_width_promotion_gate(
                tmp_path,
                status="pass",
                decision=DEFINE_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
            )
            local_dir = tmp_path / "local"
            colab_dir = tmp_path / "colab"
            _write_support_width_validation_comparison(local_dir, seed=2)
            _write_support_width_validation_comparison(colab_dir, seed=3)

            report = write_residual_support_width_promotion_gate_satisfaction_report(
                promotion_gate,
                (local_dir, colab_dir),
                tmp_path / "support_width_promotion_gate_satisfaction",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["promotion_gate_satisfied"])
            self.assertFalse(report["promote_support_width_default"])
            self.assertIsNone(report["selected_next_direction"])
            self.assertTrue(
                any(
                    failure["field"].startswith("local.")
                    and failure["field"].endswith(".seed")
                    for failure in report["evidence"]["failures"]
                )
            )


class PostSupportWidthResidualLearningGateReportTest(unittest.TestCase):
    def test_promoted_top_k2_capacity_matrix_defines_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            promotion_report = _write_support_width_promotion_satisfaction(
                tmp_path,
                status="pass",
                decision=SATISFY_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
                selected_top_k=2,
            )
            configs = _write_post_support_width_capacity_configs(tmp_path)

            report = write_post_support_width_residual_learning_gate_report(
                promotion_report,
                configs,
                tmp_path / "post_support_width_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_POST_SUPPORT_WIDTH_RESIDUAL_LEARNING_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "residual_capacity_under_top_k2_validation",
            )
            self.assertEqual(report["default_support_width_top_k"], 2)
            self.assertEqual(report["evidence"]["config_count"], 4)
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertIn("relaleap.experiments.compare", report["commands"]["compare"])
            self.assertTrue(
                (tmp_path / "post_support_width_gate" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "post_support_width_gate" / "decision_report.md").is_file()
            )

    def test_top_k_drift_blocks_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            promotion_report = _write_support_width_promotion_satisfaction(
                tmp_path,
                status="pass",
                decision=SATISFY_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE,
                selected_top_k=2,
            )
            configs = _write_post_support_width_capacity_configs(
                tmp_path,
                baseline_top_k=1,
            )

            report = write_post_support_width_residual_learning_gate_report(
                promotion_report,
                configs,
                tmp_path / "post_support_width_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": "config.top_k",
                    "expected": 2,
                    "actual": 1,
                    "path": str(configs[0]),
                },
                report["evidence"]["failures"],
            )


class PostSupportWidthResidualCapacityDecisionReportTest(unittest.TestCase):
    def test_mixed_capacity_evidence_stops_capacity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local_capacity"
            colab_dir = tmp_path / "colab_capacity"
            _write_post_support_width_capacity_comparison(
                local_dir,
                char_capacity_delta=-0.03,
                token_capacity_delta=0.01,
            )
            _write_post_support_width_capacity_comparison(
                colab_dir,
                char_capacity_delta=0.01,
                token_capacity_delta=-0.02,
            )

            report = write_post_support_width_residual_capacity_decision_report(
                (local_dir, colab_dir),
                tmp_path / "capacity_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], STOP_RESIDUAL_CAPACITY_VALIDATION)
            self.assertEqual(
                report["selected_next_direction"],
                "support_width_deconfounding_matrix_and_exhaustive_support_audit",
            )
            self.assertFalse(report["promote_residual_capacity_default"])
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertEqual(len(report["evidence"]["capacity_failures"]), 2)
            self.assertTrue(
                (tmp_path / "capacity_decision" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "capacity_decision" / "decision_report.md").is_file()
            )

    def test_consistent_capacity_wins_define_one_repeat_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local_capacity"
            colab_dir = tmp_path / "colab_capacity"
            _write_post_support_width_capacity_comparison(
                local_dir,
                char_capacity_delta=-0.03,
                token_capacity_delta=-0.01,
            )
            _write_post_support_width_capacity_comparison(
                colab_dir,
                char_capacity_delta=-0.02,
                token_capacity_delta=-0.02,
            )

            report = write_post_support_width_residual_capacity_decision_report(
                (local_dir, colab_dir),
                tmp_path / "capacity_decision",
                artifact_check_paths=(
                    local_dir / "artifact_check_local.json",
                    colab_dir / "artifact_check_local.json",
                ),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], DEFINE_RESIDUAL_CAPACITY_REPEAT_GATE)
            self.assertEqual(
                report["selected_next_direction"],
                "residual_capacity_seed2_repeat",
            )
            self.assertEqual(report["evidence"]["capacity_failures"], [])


def _write_support_width_validation_comparison(
    comparison_dir: Path,
    *,
    seed: int | None = None,
) -> None:
    comparison_dir.mkdir(parents=True)
    suffix = "" if seed is None else f"_seed{seed}"
    runs = [
        _support_width_validation_run(
            f"char_larger_hep_temporal_clipped_objective_gate{suffix}",
            f"configs/char_larger_hep_temporal_clipped_objective_gate{suffix}.yaml",
            dataset="tiny_shakespeare_char",
            top_k=1,
            alpha0_loss=3.40,
            final_loss=3.40,
            best_loss=3.399,
        ),
        _support_width_validation_run(
            f"char_larger_support_wide_hep_temporal_clipped_objective_gate{suffix}",
            f"configs/char_larger_support_wide_hep_temporal_clipped_objective_gate{suffix}.yaml",
            dataset="tiny_shakespeare_char",
            top_k=2,
            alpha0_loss=3.15,
            final_loss=3.15,
            best_loss=3.149,
        ),
        _support_width_validation_run(
            f"token_larger_hep_temporal_clipped_objective_gate{suffix}",
            f"configs/token_larger_hep_temporal_clipped_objective_gate{suffix}.yaml",
            dataset="tiny_shakespeare_word",
            top_k=1,
            alpha0_loss=4.06,
            final_loss=4.06,
            best_loss=4.059,
        ),
        _support_width_validation_run(
            f"token_larger_support_wide_hep_temporal_clipped_objective_gate{suffix}",
            f"configs/token_larger_support_wide_hep_temporal_clipped_objective_gate{suffix}.yaml",
            dataset="tiny_shakespeare_word",
            top_k=2,
            alpha0_loss=3.53,
            final_loss=3.53,
            best_loss=3.531,
        ),
    ]
    (comparison_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "verdict": {
                    "status": "pass",
                    "phase0_invariants": {"passed": True},
                    "artifact_invariants": {"passed": True},
                },
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "artifact_check_local.json").write_text(
        json.dumps({"status": "pass"}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_post_support_width_capacity_comparison(
    comparison_dir: Path,
    *,
    char_capacity_delta: float,
    token_capacity_delta: float,
) -> None:
    comparison_dir.mkdir(parents=True)
    char_baseline_loss = 3.10
    token_baseline_loss = 3.55
    runs = [
        _support_width_validation_run(
            "char_larger_hep_temporal_clipped_objective_gate",
            "configs/char_larger_hep_temporal_clipped_objective_gate.yaml",
            dataset="tiny_shakespeare_char",
            top_k=2,
            alpha0_loss=char_baseline_loss,
            final_loss=char_baseline_loss,
            best_loss=char_baseline_loss,
        ),
        _support_width_validation_run(
            "char_larger_capacity_hep_temporal_clipped_objective_gate",
            "configs/char_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            dataset="tiny_shakespeare_char",
            top_k=2,
            alpha0_loss=char_baseline_loss + char_capacity_delta,
            final_loss=char_baseline_loss + char_capacity_delta,
            best_loss=char_baseline_loss + char_capacity_delta,
        ),
        _support_width_validation_run(
            "token_larger_hep_temporal_clipped_objective_gate",
            "configs/token_larger_hep_temporal_clipped_objective_gate.yaml",
            dataset="tiny_shakespeare_word",
            top_k=2,
            alpha0_loss=token_baseline_loss,
            final_loss=token_baseline_loss,
            best_loss=token_baseline_loss,
        ),
        _support_width_validation_run(
            "token_larger_capacity_hep_temporal_clipped_objective_gate",
            "configs/token_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            dataset="tiny_shakespeare_word",
            top_k=2,
            alpha0_loss=token_baseline_loss + token_capacity_delta,
            final_loss=token_baseline_loss + token_capacity_delta,
            best_loss=token_baseline_loss + token_capacity_delta,
        ),
    ]
    (comparison_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "verdict": {
                    "status": "pass",
                    "phase0_invariants": {"passed": True},
                    "artifact_invariants": {"passed": True},
                },
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "artifact_check_local.json").write_text(
        json.dumps({"status": "pass"}, indent=2) + "\n",
        encoding="utf-8",
    )


def _support_width_validation_run(
    experiment_id: str,
    config_path: str,
    *,
    dataset: str,
    top_k: int,
    alpha0_loss: float,
    final_loss: float,
    best_loss: float,
) -> dict[str, object]:
    return {
        "artifact_invariants": {
            "metrics_csv": True,
            "notes_md": True,
            "summary_json": True,
        },
        "config_path": config_path,
        "dataset": dataset,
        "experiment_id": experiment_id,
        "final_residual_loss": final_loss,
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": alpha0_loss,
                "max_logit_delta_from_ordinary": 0.0,
                "pinned_vs_repicked_logit_delta": 0.0,
                "support_change_fraction": 0.0,
            },
            {
                "alpha": 1.0,
                "loss": best_loss,
                "max_logit_delta_from_ordinary": 0.001,
                "pinned_vs_repicked_logit_delta": 0.001,
                "support_change_fraction": 0.5 if top_k == 2 else 0.0,
            },
        ],
        "hep_settling_objective": "temporal_consistency_gradient",
        "hep_update_clip_norm": 0.01,
        "invariants": {
            "frozen_base_unchanged": True,
            "hep_alpha_0_equivalence": True,
            "residual_parameters_updated": True,
            "zero_init_identity": True,
        },
        "residual_objective": "supervised_ce",
        "status": "ok",
        "support_stress": True,
        "support_stress_preset": False,
        "top_k": top_k,
        "training_steps": 50,
    }


def _write_capacity_support_comparison(comparison_dir: Path) -> None:
    comparison_dir.mkdir(parents=True)
    runs = [
        _capacity_support_run(
            "char_validation_hep_temporal_clipped_objective_gate",
            "configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            best_loss=3.58,
            final_loss=3.59,
            support_change=0.0,
            pinned_delta=0.0,
        ),
        _capacity_support_run(
            "char_validation_capacity_hep_temporal_clipped_objective_gate",
            "configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml",
            best_loss=3.57,
            final_loss=3.58,
            support_change=0.0,
            pinned_delta=0.0,
        ),
        _capacity_support_run(
            "char_validation_support_wide_hep_temporal_clipped_objective_gate",
            "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml",
            best_loss=3.50,
            final_loss=3.51,
            support_change=0.37,
            pinned_delta=0.005,
        ),
        _capacity_support_run(
            "char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate",
            "configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml",
            best_loss=3.52,
            final_loss=3.53,
            support_change=0.29,
            pinned_delta=0.002,
        ),
    ]
    (comparison_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "verdict": {
                    "status": "pass",
                    "phase0_invariants": {"passed": True},
                    "artifact_invariants": {"passed": True},
                },
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _capacity_support_run(
    experiment_id: str,
    config_path: str,
    *,
    best_loss: float,
    final_loss: float,
    support_change: float,
    pinned_delta: float,
) -> dict[str, object]:
    return {
        "artifact_invariants": {
            "metrics_csv": True,
            "notes_md": True,
            "summary_json": True,
        },
        "config_path": config_path,
        "experiment_id": experiment_id,
        "final_residual_loss": final_loss,
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": best_loss + 0.01,
                "max_logit_delta_from_ordinary": 0.0,
                "pinned_vs_repicked_logit_delta": 0.0,
                "support_change_fraction": support_change,
            },
            {
                "alpha": 1.0,
                "loss": best_loss,
                "max_logit_delta_from_ordinary": 0.001,
                "pinned_vs_repicked_logit_delta": pinned_delta,
                "support_change_fraction": support_change,
            },
        ],
        "hep_settling_objective": "temporal_consistency_gradient",
        "hep_update_clip_norm": 0.01,
        "invariants": {
            "frozen_base_unchanged": True,
            "hep_alpha_0_equivalence": True,
            "residual_parameters_updated": True,
            "zero_init_identity": True,
        },
        "residual_objective": "supervised_ce",
        "status": "ok",
        "support_stress": True,
        "support_stress_preset": False,
        "training_steps": 25,
    }


def _write_next_direction_report(tmp_path: Path, *, status: str) -> Path:
    report_dir = tmp_path / "residual_learning_next_direction"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": DEFINE_RESIDUAL_CAPACITY_SUPPORT_DIAGNOSTIC_GATE,
                "selected_next_direction": "residual_capacity_support_diagnostic",
                "promote_residual_learning_method": False,
                "default_residual_objective": "supervised_ce",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_capacity_support_configs(tmp_path: Path) -> list[Path]:
    specs = [
        ("baseline.yaml", "baseline", 12, 1),
        ("capacity.yaml", "capacity", 24, 1),
        ("support.yaml", "support", 12, 2),
        ("capacity_support.yaml", "capacity_support", 24, 2),
    ]
    paths = []
    for filename, suffix, num_columns, top_k in specs:
        path = tmp_path / filename
        path.write_text(
            f"""run:
  experiment_id: char_validation_{suffix}_hep_temporal_clipped_objective_gate
  seed: 1
  max_steps: 25

data:
  dataset: tiny_shakespeare_char
  seq_len: 64

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 2
    hidden_dim: 64
  columns:
    num_columns: {num_columns}
    atoms_per_column: 4
    top_k: {top_k}
    insertion_sites: 1
    support_stress: true
    support_stress_preset: false

inference:
  pc_steps: 3
  hep_alpha: 0.0
  hep_alpha_sweep: "0.0,0.25,0.5,1.0"
  hep_update_clip_norm: 0.01
  hep_settling_objective: temporal_consistency_gradient

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _write_capacity_support_colab_decision(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
) -> Path:
    report_dir = tmp_path / "residual_capacity_support_diagnostic_colab_decision"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "selected_next_direction": (
                    "residual_capacity_support_validation_gate"
                    if decision == CONTINUE_RESIDUAL_CAPACITY_SUPPORT_VALIDATION
                    else None
                ),
                "promote_residual_learning_method": False,
                "default_residual_objective": "supervised_ce",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_support_width_validation_decision(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
) -> Path:
    report_dir = tmp_path / "residual_support_width_validation_decision"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "selected_next_direction": (
                    "support_width_repeat_or_capacity_interaction_validation"
                    if decision == CONTINUE_RESIDUAL_SUPPORT_WIDTH_VALIDATION
                    else None
                ),
                "promote_residual_learning_method": False,
                "default_residual_objective": "supervised_ce",
                "default_support_stress_mitigation": "temporal_clipped_hep",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_support_width_repeat_decision(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
    promote_support_width_default: bool,
) -> Path:
    report_dir = tmp_path / "residual_support_width_repeat_decision"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "promote_support_width_default": promote_support_width_default,
                "evidence": {
                    "backend_count": 2,
                    "failures": [],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_support_width_promotion_gate(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
) -> Path:
    report_dir = tmp_path / "residual_support_width_promotion_gate"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "selected_next_direction": (
                    "support_width_seed3_promotion_gate_evidence"
                    if decision == DEFINE_RESIDUAL_SUPPORT_WIDTH_PROMOTION_GATE
                    else None
                ),
                "promote_support_width_default": False,
                "evidence": {
                    "failures": [],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_support_width_promotion_satisfaction(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
    selected_top_k: int | None,
) -> Path:
    report_dir = tmp_path / "residual_support_width_promotion_gate_satisfaction"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "decision_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "selected_support_width_top_k": selected_top_k,
                "promotion_gate_satisfied": status == "pass",
                "promote_support_width_default": status == "pass",
                "evidence": {
                    "failures": [],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_post_support_width_capacity_configs(
    tmp_path: Path,
    *,
    baseline_top_k: int = 2,
) -> list[Path]:
    specs = [
        (
            "char_larger.yaml",
            "char_larger_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_char",
            128,
            24,
            baseline_top_k,
        ),
        (
            "char_larger_capacity.yaml",
            "char_larger_capacity_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_char",
            128,
            48,
            2,
        ),
        (
            "token_larger.yaml",
            "token_larger_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_word",
            64,
            24,
            baseline_top_k,
        ),
        (
            "token_larger_capacity.yaml",
            "token_larger_capacity_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_word",
            64,
            48,
            2,
        ),
    ]
    paths = []
    for filename, experiment_id, dataset, seq_len, num_columns, top_k in specs:
        path = tmp_path / filename
        path.write_text(
            f"""run:
  experiment_id: {experiment_id}
  seed: 1
  max_steps: 50

data:
  dataset: {dataset}
  seq_len: {seq_len}

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 2
    hidden_dim: 96
  columns:
    num_columns: {num_columns}
    atoms_per_column: 4
    top_k: {top_k}
    insertion_sites: 1
    support_stress: true
    support_stress_preset: false

inference:
  pc_steps: 4
  hep_alpha: 0.0
  hep_alpha_sweep: "0.0,0.25,0.5,1.0"
  hep_update_clip_norm: 0.01
  hep_settling_objective: temporal_consistency_gradient

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _write_support_width_validation_configs_with_seed(
    tmp_path: Path,
    *,
    seed: int,
) -> list[Path]:
    specs = [
        (
            "char_larger_baseline.yaml",
            "char_larger_hep_temporal_clipped_objective_gate"
            + (f"_seed{seed}" if seed != 1 else ""),
            "tiny_shakespeare_char",
            128,
            1,
        ),
        (
            "char_larger_support.yaml",
            "char_larger_support_wide_hep_temporal_clipped_objective_gate"
            + (f"_seed{seed}" if seed != 1 else ""),
            "tiny_shakespeare_char",
            128,
            2,
        ),
        (
            "token_larger_baseline.yaml",
            "token_larger_hep_temporal_clipped_objective_gate"
            + (f"_seed{seed}" if seed != 1 else ""),
            "tiny_shakespeare_word",
            64,
            1,
        ),
        (
            "token_larger_support.yaml",
            "token_larger_support_wide_hep_temporal_clipped_objective_gate"
            + (f"_seed{seed}" if seed != 1 else ""),
            "tiny_shakespeare_word",
            64,
            2,
        ),
    ]
    paths = []
    for filename, experiment_id, dataset, seq_len, top_k in specs:
        path = tmp_path / filename
        path.write_text(
            f"""run:
  experiment_id: {experiment_id}
  seed: {seed}
  max_steps: 50

data:
  dataset: {dataset}
  seq_len: {seq_len}

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 2
    hidden_dim: 96
  columns:
    num_columns: 24
    atoms_per_column: 4
    top_k: {top_k}
    insertion_sites: 1
    support_stress: true
    support_stress_preset: false

inference:
  pc_steps: 4
  hep_alpha: 0.0
  hep_alpha_sweep: "0.0,0.25,0.5,1.0"
  hep_update_clip_norm: 0.01
  hep_settling_objective: temporal_consistency_gradient

outputs:
  require_summary_json: true
  require_metrics_csv: true
  require_notes_md: true
""",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _write_support_width_validation_configs(
    tmp_path: Path,
    *,
    seed: int = 1,
) -> list[Path]:
    return _write_support_width_validation_configs_with_seed(tmp_path, seed=seed)


def _write_stopped_residual_objective_reports(tmp_path: Path) -> list[Path]:
    specs = [
        (
            "residual_objective_gate_decision",
            KEEP_SUPERVISED_CE_RESIDUAL_OBJECTIVE_DEFAULT,
        ),
        (
            "anchored_pc_residual_objective_decision",
            STOP_PC_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
        (
            "confidence_penalty_residual_objective_decision",
            STOP_CONFIDENCE_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
        (
            "margin_penalty_residual_objective_decision",
            STOP_MARGIN_PENALTY_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
        (
            "label_smoothing_residual_objective_decision",
            STOP_LABEL_SMOOTHING_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
        (
            "focal_residual_objective_promotion_gate_satisfaction",
            STOP_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
        (
            "temporal_consistency_residual_objective_decision",
            STOP_TEMPORAL_CONSISTENCY_RESIDUAL_OBJECTIVE_VALIDATION,
        ),
    ]
    paths = []
    for dirname, decision in specs:
        report_dir = tmp_path / dirname
        report_dir.mkdir(parents=True)
        report_path = report_dir / "decision_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "decision": decision,
                    "promote_residual_learning_method": False,
                    "default_residual_objective": "supervised_ce",
                    "next_step": "placeholder",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(report_path)
    return paths


def _write_passing_focal_promotion_gate(tmp_path: Path) -> Path:
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

    decision = write_focal_residual_objective_decision_report(
        comparison_dirs,
        tmp_path / "focal_decision",
        artifact_check_paths=artifact_checks,
    )
    if decision["decision"] != CONTINUE_FOCAL_RESIDUAL_OBJECTIVE_VALIDATION:
        raise AssertionError(decision)
    gate = write_focal_residual_objective_promotion_gate_report(
        tmp_path / "focal_decision" / "decision_report.json",
        tmp_path / "focal_promotion_gate",
    )
    if gate["decision"] != DEFINE_FOCAL_RESIDUAL_OBJECTIVE_PROMOTION_GATE:
        raise AssertionError(gate)
    return tmp_path / "focal_promotion_gate" / "decision_report.json"


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
    temporal_consistency_best_hep_losses: list[float] | None = None,
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
    if temporal_consistency_best_hep_losses is not None:
        for index, best_loss in enumerate(temporal_consistency_best_hep_losses):
            suffix = "" if index == 0 else f"_w{index}"
            runs.append(
                _residual_objective_run_entry(
                    f"temporal_consistency{suffix}",
                    residual_objective="supervised_ce_temporal_consistency",
                    initial_loss=3.75,
                    final_loss=3.65,
                    best_hep_loss=best_loss,
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


class SupportWidthDeconfoundingAuditReportTest(unittest.TestCase):
    def test_valid_local_matrix_records_support_width_utilization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            artifact_check_path = tmp_path / "artifact_check.json"
            _write_support_width_deconfounding_comparison(comparison_dir)
            artifact_check_path.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_support_width_deconfounding_audit_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                RUN_COLAB_SUPPORT_WIDTH_DECONFOUNDING_AUDIT,
            )
            self.assertFalse(report["promote_support_width_default"])
            self.assertTrue(report["evidence"]["support_width_improves_loss"])
            self.assertFalse(report["evidence"]["capacity_improves_loss"])
            self.assertTrue(report["evidence"]["support_width_improves_utilization"])
            self.assertEqual(
                report["evidence"]["comparisons"][
                    "support_width_minus_baseline_used_columns"
                ],
                10.0,
            )
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_failed_artifact_check_blocks_audit_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            artifact_check_path = tmp_path / "artifact_check.json"
            _write_support_width_deconfounding_comparison(comparison_dir)
            artifact_check_path.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_support_width_deconfounding_audit_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertEqual(report["evidence"]["failures"][0]["field"], "artifact_check.status")

    def test_stale_support_width_audit_artifacts_fail_without_type_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comparison_dir = tmp_path / "comparison"
            artifact_check_path = tmp_path / "artifact_check.json"
            _write_support_width_deconfounding_comparison(comparison_dir)
            summary_path = comparison_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for run in summary["runs"]:
                run.pop("num_columns")
                run.pop("top_k")
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifact_check_path.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_support_width_deconfounding_audit_report(
                comparison_dir,
                tmp_path / "report",
                artifact_check_path=artifact_check_path,
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertTrue(
                any(
                    failure["field"].endswith(".matrix_cell")
                    for failure in report["evidence"]["failures"]
                )
            )


class ContextualSupportRouterDecisionReportTest(unittest.TestCase):
    def test_matching_local_colab_contextual_router_evidence_defines_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local_contextual"
            colab_dir = tmp_path / "colab_contextual"
            local_check = tmp_path / "local_artifact_check.json"
            colab_check = tmp_path / "colab_artifact_check.json"
            _write_contextual_support_router_comparison(local_dir)
            _write_contextual_support_router_comparison(colab_dir)
            for path in (local_check, colab_check):
                path.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )

            report = write_contextual_support_router_decision_report(
                (local_dir, colab_dir),
                tmp_path / "contextual_router_decision",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
            )
            self.assertFalse(report["promote_contextual_support_router_default"])
            self.assertEqual(report["evidence"]["contextual_loss_win_count"], 2)
            self.assertEqual(
                report["evidence"]["contextual_utilization_win_count"],
                2,
            )
            self.assertEqual(
                report["evidence"]["contextual_churn_reduction_count"],
                2,
            )
            self.assertEqual(report["evidence"]["contextual_nonzero_hep_win_count"], 0)
            self.assertTrue(
                (tmp_path / "contextual_router_decision" / "decision_report.json").is_file()
            )
            self.assertTrue(
                (tmp_path / "contextual_router_decision" / "decision_report.md").is_file()
            )

    def test_failed_artifact_check_blocks_contextual_router_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            local_dir = tmp_path / "local_contextual"
            colab_dir = tmp_path / "colab_contextual"
            local_check = tmp_path / "local_artifact_check.json"
            colab_check = tmp_path / "colab_artifact_check.json"
            _write_contextual_support_router_comparison(local_dir)
            _write_contextual_support_router_comparison(colab_dir)
            local_check.write_text(
                json.dumps({"status": "pass"}, indent=2) + "\n",
                encoding="utf-8",
            )
            colab_check.write_text(
                json.dumps({"status": "fail"}, indent=2) + "\n",
                encoding="utf-8",
            )

            report = write_contextual_support_router_decision_report(
                (local_dir, colab_dir),
                tmp_path / "contextual_router_decision",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertTrue(
                any(
                    failure["field"] == "artifact_check.status"
                    for failure in report["evidence"]["failures"]
                )
            )


class ContextualSupportRouterPromotionGateReportTest(unittest.TestCase):
    def test_passing_contextual_decision_defines_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            decision_path = _write_contextual_support_router_decision(
                tmp_path,
                status="pass",
                decision=DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
                promote_contextual_support_router_default=False,
            )
            config_paths = _write_contextual_support_router_gate_configs(tmp_path)

            report = write_contextual_support_router_promotion_gate_report(
                decision_path,
                config_paths,
                tmp_path / "contextual_router_promotion_gate",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
            )
            self.assertEqual(
                report["selected_next_direction"],
                "contextual_support_router_promotion_gate_larger_char_token",
            )
            self.assertFalse(report["promote_contextual_support_router_default"])
            self.assertEqual(report["evidence"]["failures"], [])
            self.assertEqual(len(report["evidence"]["required_evidence"]), 2)
            self.assertIn("colab_compare", report["commands"])
            self.assertTrue(
                (
                    tmp_path
                    / "contextual_router_promotion_gate"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "contextual_router_promotion_gate"
                    / "decision_report.md"
                ).is_file()
            )

    def test_promoting_contextual_decision_blocks_gate_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            decision_path = _write_contextual_support_router_decision(
                tmp_path,
                status="pass",
                decision=DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
                promote_contextual_support_router_default=True,
            )
            config_paths = _write_contextual_support_router_gate_configs(tmp_path)

            report = write_contextual_support_router_promotion_gate_report(
                decision_path,
                config_paths,
                tmp_path / "contextual_router_promotion_gate",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(report["selected_next_direction"])
            self.assertIn(
                {
                    "field": (
                        "contextual_decision_report."
                        "promote_contextual_support_router_default"
                    ),
                    "expected": False,
                    "actual": True,
                    "path": str(decision_path),
                },
                report["evidence"]["failures"],
            )


class ContextualSupportRouterPromotionGateSatisfactionReportTest(unittest.TestCase):
    def test_matching_local_colab_gate_evidence_promotes_contextual_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report_path = _write_contextual_support_router_decision(
                tmp_path,
                status="pass",
                decision=DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
                promote_contextual_support_router_default=False,
            )
            local_dir = tmp_path / "contextual_support_router_promotion_gate"
            colab_dir = tmp_path / "colab_contextual_support_router_promotion_gate"
            local_check = tmp_path / "local_artifact_check.json"
            colab_check = tmp_path / "colab_artifact_check.json"
            _write_contextual_support_router_promotion_gate_comparison(local_dir)
            _write_contextual_support_router_promotion_gate_comparison(colab_dir)
            for path in (local_check, colab_check):
                path.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )

            report = write_contextual_support_router_promotion_gate_satisfaction_report(
                gate_report_path,
                (local_dir, colab_dir),
                tmp_path / "contextual_router_promotion_gate_satisfaction",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["decision"],
                SATISFY_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
            )
            self.assertTrue(report["promote_contextual_support_router_default"])
            self.assertEqual(report["evidence"]["gate_cell_count"], 4)
            self.assertEqual(report["evidence"]["contextual_loss_win_count"], 4)
            self.assertEqual(report["evidence"]["contextual_utilization_win_count"], 4)
            self.assertEqual(report["evidence"]["contextual_nonzero_hep_win_count"], 0)
            self.assertTrue(
                (
                    tmp_path
                    / "contextual_router_promotion_gate_satisfaction"
                    / "decision_report.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "contextual_router_promotion_gate_satisfaction"
                    / "decision_report.md"
                ).is_file()
            )

    def test_missing_tokenized_win_blocks_contextual_router_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_report_path = _write_contextual_support_router_decision(
                tmp_path,
                status="pass",
                decision=DEFINE_CONTEXTUAL_SUPPORT_ROUTER_PROMOTION_GATE,
                promote_contextual_support_router_default=False,
            )
            local_dir = tmp_path / "contextual_support_router_promotion_gate"
            colab_dir = tmp_path / "colab_contextual_support_router_promotion_gate"
            local_check = tmp_path / "local_artifact_check.json"
            colab_check = tmp_path / "colab_artifact_check.json"
            _write_contextual_support_router_promotion_gate_comparison(
                local_dir,
                token_contextual_alpha0_loss=3.7,
            )
            _write_contextual_support_router_promotion_gate_comparison(colab_dir)
            for path in (local_check, colab_check):
                path.write_text(
                    json.dumps({"status": "pass"}, indent=2) + "\n",
                    encoding="utf-8",
                )

            report = write_contextual_support_router_promotion_gate_satisfaction_report(
                gate_report_path,
                (local_dir, colab_dir),
                tmp_path / "contextual_router_promotion_gate_satisfaction",
                artifact_check_paths=(local_check, colab_check),
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertFalse(report["promote_contextual_support_router_default"])
            self.assertTrue(
                any(
                    failure["field"].endswith(".alpha0_loss")
                    for failure in report["evidence"]["failures"]
                )
            )


class ExhaustiveSupportAuditReportTest(unittest.TestCase):
    def test_valid_audit_selects_router_support_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audit_dir = tmp_path / "audit"
            _write_exhaustive_support_audit(audit_dir)

            report = write_exhaustive_support_audit_report(
                audit_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], DIAGNOSE_EXHAUSTIVE_SUPPORT_AUDIT)
            self.assertEqual(
                report["selected_next_direction"],
                "router_support_selection",
            )
            self.assertTrue(report["evidence"]["router_improvement_signal"])
            self.assertTrue(report["evidence"]["column_redundancy_signal"])
            self.assertTrue(report["evidence"]["pairwise_composition_signal"])
            self.assertEqual(
                report["evidence"][
                    "best_router_target_holdout_oracle_gap_recovery_fraction"
                ],
                0.45,
            )
            self.assertEqual(
                report["evidence"][
                    "contextual_support_head_holdout_oracle_gap_recovery_fraction"
                ],
                0.7,
            )
            self.assertIn("contextual support-head", report["next_step"])
            self.assertTrue((tmp_path / "report" / "decision_report.json").is_file())
            self.assertTrue((tmp_path / "report" / "decision_report.md").is_file())

    def test_missing_audit_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audit_dir = tmp_path / "audit"
            _write_exhaustive_support_audit(audit_dir)
            (audit_dir / "pairwise_synergy.csv").unlink()

            report = write_exhaustive_support_audit_report(
                audit_dir,
                tmp_path / "report",
            )

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["decision"], INSUFFICIENT_EVIDENCE)
            self.assertTrue(
                any(
                    failure["field"] == "artifacts.pairwise_synergy_csv"
                    for failure in report["evidence"]["failures"]
                )
            )


def _write_exhaustive_support_audit(audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "experiment_id": "test_exhaustive_support_audit",
        "config_path": "configs/test.yaml",
        "audit": {
            "num_columns": 4,
            "top_k": 2,
            "support_set_count": 6,
            "router_loss": 3.5,
            "oracle_loss": 3.4,
            "oracle_support_regret": 0.1,
            "oracle_support_regret_positive_fraction": 0.75,
            "best_global_fixed_support": "0,1",
            "best_global_fixed_support_loss": 3.55,
            "router_minus_best_global_fixed_support_loss": -0.05,
            "dominant_router_support": "0,2",
            "dominant_router_support_count": 10,
            "best_one_swap_support": "0,1",
            "best_one_swap_recovers_oracle_gap_fraction": 1.0,
            "support_audit": {
                "used_columns": 3,
                "dead_columns": 1,
                "unique_support_sets": 4,
            },
            "router_oracle_target_diagnostic": {
                "selector": "linear_hidden_to_oracle_pair",
                "holdout": {
                    "oracle_target_accuracy": 0.5,
                    "oracle_gap_recovery_fraction": 0.25,
                    "selector_minus_router_loss": -0.025,
                },
            },
            "router_oracle_target_nonlinear_diagnostic": {
                "selector": "mlp_hidden_to_oracle_pair",
                "holdout": {
                    "oracle_target_accuracy": 0.55,
                    "oracle_gap_recovery_fraction": 0.35,
                    "selector_minus_router_loss": -0.035,
                },
            },
            "router_oracle_target_contextual_diagnostic": {
                "selector": "mlp_contextual_hidden_to_oracle_pair",
                "holdout": {
                    "oracle_target_accuracy": 0.6,
                    "oracle_gap_recovery_fraction": 0.45,
                    "selector_minus_router_loss": -0.045,
                },
            },
            "contextual_router_support_intervention": {
                "selector": "mlp_contextual_hidden_to_oracle_pair",
                "intervention": "per_token_predicted_support_indices",
                "holdout": {
                    "intervention_loss": 3.52,
                    "intervention_minus_router_loss": -0.08,
                    "oracle_gap_recovery_fraction": 0.8,
                },
            },
            "contextual_router_support_head": {
                "selector": "mlp_contextual_support_head_ce_minimizer",
                "intervention": "per_token_predicted_support_indices",
                "holdout": {
                    "intervention_loss": 3.53,
                    "intervention_minus_router_loss": -0.07,
                    "oracle_gap_recovery_fraction": 0.7,
                },
            },
            "top_supports_by_synergy": [
                {
                    "support": "0,1",
                    "pairwise_synergy": 0.02,
                }
            ],
        },
        "artifacts": {
            "summary_json": str(audit_dir / "summary.json"),
            "support_losses_csv": str(audit_dir / "support_losses.csv"),
            "pairwise_synergy_csv": str(audit_dir / "pairwise_synergy.csv"),
            "router_target_diagnostic_csv": str(
                audit_dir / "router_target_diagnostic.csv"
            ),
            "router_target_nonlinear_diagnostic_csv": str(
                audit_dir / "router_target_nonlinear_diagnostic.csv"
            ),
            "router_target_contextual_diagnostic_csv": str(
                audit_dir / "router_target_contextual_diagnostic.csv"
            ),
            "router_support_intervention_csv": str(
                audit_dir / "router_support_intervention.csv"
            ),
            "contextual_router_support_head_csv": str(
                audit_dir / "contextual_router_support_head.csv"
            ),
            "notes_md": str(audit_dir / "notes.md"),
        },
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "support_losses.csv").write_text("support_key,loss\n0,3.5\n")
    (audit_dir / "pairwise_synergy.csv").write_text(
        "support_key,pairwise_synergy\n0,1,0.02\n"
    )
    (audit_dir / "router_target_diagnostic.csv").write_text(
        "split,oracle_gap_recovery_fraction\nholdout_odd_positions,0.25\n",
        encoding="utf-8",
    )
    (audit_dir / "router_target_nonlinear_diagnostic.csv").write_text(
        "split,oracle_gap_recovery_fraction\nholdout_odd_positions,0.35\n",
        encoding="utf-8",
    )
    (audit_dir / "router_target_contextual_diagnostic.csv").write_text(
        "split,oracle_gap_recovery_fraction\nholdout_odd_positions,0.45\n",
        encoding="utf-8",
    )
    (audit_dir / "router_support_intervention.csv").write_text(
        "split,intervention_loss,intervention_minus_router_loss,oracle_gap_recovery_fraction\n"
        "holdout_odd_positions,3.52,-0.08,0.8\n",
        encoding="utf-8",
    )
    (audit_dir / "contextual_router_support_head.csv").write_text(
        "split,intervention_loss,intervention_minus_router_loss,oracle_gap_recovery_fraction\n"
        "holdout_odd_positions,3.53,-0.07,0.7\n",
        encoding="utf-8",
    )
    (audit_dir / "notes.md").write_text("# audit\n", encoding="utf-8")


def _write_support_width_deconfounding_comparison(comparison_dir: Path) -> None:
    comparison_dir.mkdir(parents=True)
    runs = [
        _support_width_deconfounding_run_entry(
            "char_validation_hep_temporal_clipped_objective_gate",
            num_columns=12,
            top_k=1,
            best_hep_loss=3.586,
            final_loss=3.587,
            used_columns=1,
            dead_columns=11,
            unique_supports=1,
            max_column_fraction=1.0,
        ),
        _support_width_deconfounding_run_entry(
            "char_validation_support_wide_hep_temporal_clipped_objective_gate",
            num_columns=12,
            top_k=2,
            best_hep_loss=3.497,
            final_loss=3.497,
            used_columns=11,
            dead_columns=1,
            unique_supports=19,
            max_column_fraction=0.375,
        ),
        _support_width_deconfounding_run_entry(
            "char_validation_capacity_hep_temporal_clipped_objective_gate",
            num_columns=24,
            top_k=1,
            best_hep_loss=3.586,
            final_loss=3.587,
            used_columns=1,
            dead_columns=23,
            unique_supports=1,
            max_column_fraction=1.0,
        ),
        _support_width_deconfounding_run_entry(
            "char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate",
            num_columns=24,
            top_k=2,
            best_hep_loss=3.490,
            final_loss=3.490,
            used_columns=13,
            dead_columns=11,
            unique_supports=20,
            max_column_fraction=0.385,
        ),
    ]
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
        "runs": runs,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_contextual_support_router_comparison(comparison_dir: Path) -> None:
    comparison_dir.mkdir(parents=True)
    runs = [
        _contextual_support_router_run_entry(
            "char_larger_support_wide_hep_temporal_clipped_objective_gate",
            support_router=None,
            alpha0_loss=3.10,
            best_hep_loss=3.099,
            final_loss=3.10,
            used_columns=12,
            dead_columns=12,
            unique_supports=15,
            max_column_fraction=0.31,
            support_change=0.52,
        ),
        _contextual_support_router_run_entry(
            "char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate",
            support_router="contextual_mlp",
            alpha0_loss=1.92,
            best_hep_loss=1.92,
            final_loss=1.92,
            used_columns=18,
            dead_columns=6,
            unique_supports=44,
            max_column_fraction=0.14,
            support_change=0.17,
        ),
    ]
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "artifact_invariants_passed": True,
        },
        "runs": runs,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_contextual_support_router_promotion_gate_comparison(
    comparison_dir: Path,
    *,
    token_contextual_alpha0_loss: float = 2.90,
) -> None:
    comparison_dir.mkdir(parents=True)
    runs = [
        _contextual_support_router_run_entry(
            "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
            support_router=None,
            alpha0_loss=3.20,
            best_hep_loss=3.20,
            final_loss=3.20,
            used_columns=12,
            dead_columns=12,
            unique_supports=13,
            max_column_fraction=0.47,
            support_change=0.28,
            dataset="tiny_shakespeare_char",
        ),
        _contextual_support_router_run_entry(
            "char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2",
            support_router="contextual_mlp",
            alpha0_loss=1.85,
            best_hep_loss=1.85,
            final_loss=1.85,
            used_columns=19,
            dead_columns=5,
            unique_supports=60,
            max_column_fraction=0.18,
            support_change=0.25,
            dataset="tiny_shakespeare_char",
        ),
        _contextual_support_router_run_entry(
            "token_larger_support_wide_hep_temporal_clipped_objective_gate",
            support_router=None,
            alpha0_loss=3.52,
            best_hep_loss=3.52,
            final_loss=3.52,
            used_columns=14,
            dead_columns=10,
            unique_supports=26,
            max_column_fraction=0.32,
            support_change=0.62,
            dataset="tiny_shakespeare_word",
        ),
        _contextual_support_router_run_entry(
            "token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate",
            support_router="contextual_mlp",
            alpha0_loss=token_contextual_alpha0_loss,
            best_hep_loss=token_contextual_alpha0_loss,
            final_loss=token_contextual_alpha0_loss,
            used_columns=20,
            dead_columns=4,
            unique_supports=52,
            max_column_fraction=0.12,
            support_change=0.32,
            dataset="tiny_shakespeare_word",
        ),
    ]
    summary = {
        "status": "ok",
        "verdict": {
            "status": "pass",
            "invariants_passed": True,
            "artifact_invariants_passed": True,
        },
        "runs": runs,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_contextual_support_router_decision(
    tmp_path: Path,
    *,
    status: str,
    decision: str,
    promote_contextual_support_router_default: bool,
) -> Path:
    report_path = tmp_path / "contextual_router_decision" / "decision_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "promote_contextual_support_router_default": (
                    promote_contextual_support_router_default
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _write_contextual_support_router_gate_configs(tmp_path: Path) -> tuple[Path, ...]:
    configs = (
        (
            "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
            "tiny_shakespeare_char",
            128,
            2,
            None,
        ),
        (
            "char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2",
            "tiny_shakespeare_char",
            128,
            2,
            "contextual_mlp",
        ),
        (
            "token_larger_support_wide_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_word",
            64,
            1,
            None,
        ),
        (
            "token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate",
            "tiny_shakespeare_word",
            64,
            1,
            "contextual_mlp",
        ),
    )
    paths = []
    for experiment_id, dataset, seq_len, seed, support_router in configs:
        path = tmp_path / f"{experiment_id}.yaml"
        router_lines = ""
        if support_router is not None:
            router_lines = (
                "    support_router: contextual_mlp\n"
                "    contextual_router_hidden_dim: 128\n"
            )
        path.write_text(
            (
                "run:\n"
                f"  experiment_id: {experiment_id}\n"
                f"  seed: {seed}\n"
                "  max_steps: 50\n"
                "data:\n"
                f"  dataset: {dataset}\n"
                f"  seq_len: {seq_len}\n"
                "training:\n"
                "  residual_objective: supervised_ce\n"
                "model:\n"
                "  base:\n"
                "    layers: 2\n"
                "    hidden_dim: 96\n"
                "  columns:\n"
                "    num_columns: 24\n"
                "    atoms_per_column: 4\n"
                "    top_k: 2\n"
                "    insertion_sites: 1\n"
                "    support_stress: true\n"
                "    support_stress_preset: false\n"
                f"{router_lines}"
                "inference:\n"
                "  pc_steps: 4\n"
                "  hep_alpha: 0.0\n"
                "  hep_alpha_sweep: \"0.0,0.25,0.5,1.0\"\n"
                "  hep_update_clip_norm: 0.01\n"
                "  hep_settling_objective: temporal_consistency_gradient\n"
                "outputs:\n"
                "  require_summary_json: true\n"
                "  require_metrics_csv: true\n"
                "  require_notes_md: true\n"
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)


def _contextual_support_router_run_entry(
    experiment_id: str,
    *,
    support_router: str | None,
    alpha0_loss: float,
    best_hep_loss: float,
    final_loss: float,
    used_columns: int,
    dead_columns: int,
    unique_supports: int,
    max_column_fraction: float,
    support_change: float,
    dataset: str = "tiny_shakespeare_char",
) -> dict[str, object]:
    run = {
        "artifact_invariants": {
            "metrics_csv": True,
            "notes_md": True,
            "summary_json": True,
        },
        "config_path": f"configs/{experiment_id}.yaml",
        "dataset": dataset,
        "experiment_id": experiment_id,
        "final_residual_loss": final_loss,
        "hep_alpha_sweep": [
            {
                "alpha": 0.0,
                "loss": alpha0_loss,
                "max_logit_delta_from_ordinary": 0.0,
                "pinned_vs_repicked_logit_delta": 0.0,
                "support_change_fraction": support_change,
            },
            {
                "alpha": 1.0,
                "loss": best_hep_loss,
                "max_logit_delta_from_ordinary": 0.001,
                "pinned_vs_repicked_logit_delta": 0.001,
                "support_change_fraction": support_change,
            },
        ],
        "hep_settling_objective": "temporal_consistency_gradient",
        "hep_update_clip_norm": 0.01,
        "invariants": {
            "frozen_base_unchanged": True,
            "hep_alpha_0_equivalence": True,
            "residual_parameters_updated": True,
            "zero_init_identity": True,
        },
        "num_columns": 24,
        "residual_objective": "supervised_ce",
        "status": "ok",
        "support_audit": {
            "dead_columns": dead_columns,
            "max_column_fraction": max_column_fraction,
            "support_positions": 512,
            "total_support_slots": 1024,
            "unique_support_sets": unique_supports,
            "used_columns": used_columns,
        },
        "support_stress": True,
        "support_stress_preset": False,
        "top_k": 2,
        "training_steps": 50,
    }
    if support_router is not None:
        run["contextual_router_hidden_dim"] = 128
        run["support_router"] = support_router
    return run


def _support_width_deconfounding_run_entry(
    experiment_id: str,
    *,
    num_columns: int,
    top_k: int,
    best_hep_loss: float,
    final_loss: float,
    used_columns: int,
    dead_columns: int,
    unique_supports: int,
    max_column_fraction: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "config_path": f"configs/{experiment_id}.yaml",
        "status": "ok",
        "dataset": "tiny_shakespeare_char",
        "num_columns": num_columns,
        "top_k": top_k,
        "residual_objective": "supervised_ce",
        "training_steps": 25,
        "support_stress": True,
        "support_stress_preset": False,
        "hep_update_clip_norm": 0.01,
        "hep_settling_objective": "temporal_consistency_gradient",
        "final_residual_loss": final_loss,
        "invariants": {
            "zero_init_identity": True,
            "frozen_base_unchanged": True,
            "hep_alpha_0_equivalence": True,
            "residual_parameters_updated": True,
        },
        "artifact_invariants": {
            "summary_json": True,
            "metrics_csv": True,
            "notes_md": True,
        },
        "support_audit": {
            "used_columns": used_columns,
            "dead_columns": dead_columns,
            "unique_support_sets": unique_supports,
            "max_column_fraction": max_column_fraction,
            "support_positions": 256,
            "total_support_slots": 256 * top_k,
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
                "support_change_fraction": 0.25 if top_k == 2 else 0.0,
                "pinned_vs_repicked_logit_delta": 0.001,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
