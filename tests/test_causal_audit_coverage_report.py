from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_audit_coverage_report import (
    EXISTING_ARTIFACTS_SUFFICIENT,
    NEW_TRAINING_REQUIRED,
    RANK_MATCHED_TOPK1_ACTIVE_POST_STOP,
    SPECIFIC_MISSING_FIELDS,
    write_causal_audit_coverage_report,
)


class CausalAuditCoverageReportTest(unittest.TestCase):
    def test_existing_aggregate_artifacts_require_field_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "fingerprint"
            matched_dir = root / "matched"
            reports = _write_source_artifacts(root)

            report = write_causal_audit_coverage_report(
                audit_dir,
                matched_dir,
                root / "coverage",
                causal_fingerprint_report_path=reports["fingerprint"],
                bracket_report_path=reports["bracket"],
                rank_matched_report_path=reports["rank"],
                retention_report_path=reports["retention"],
                post_stop_report_path=root / "missing_post_stop_report.json",
                finite_update_matrix_path=root / "missing_finite_update_matrix.csv",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision"], SPECIFIC_MISSING_FIELDS)
            coverage = report["coverage"]
            self.assertEqual(coverage["row_granularity"], "aggregate")
            self.assertEqual(coverage["matched_strata_count"], 2)
            self.assertEqual(coverage["missing_controls_for_deconfounded_matrix"], [])
            self.assertIn(
                "per_token_rows",
                coverage["missing_fields_for_deconfounded_no_training_audit"],
            )
            self.assertIn(
                "active_rank_proxy",
                coverage["missing_fields_for_deconfounded_no_training_audit"],
            )
            self.assertTrue((root / "coverage" / "decision_report.json").is_file())
            self.assertTrue((root / "coverage" / "decision_report.md").is_file())

    def test_complete_matching_fields_are_sufficient_for_no_training_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = _write_source_artifacts(
                root,
                include_per_token_fields=True,
                include_active_rank=True,
                include_residual_norm_bin=True,
            )

            report = write_causal_audit_coverage_report(
                root / "fingerprint",
                root / "matched",
                root / "coverage",
                causal_fingerprint_report_path=reports["fingerprint"],
                bracket_report_path=reports["bracket"],
                rank_matched_report_path=reports["rank"],
                retention_report_path=reports["retention"],
                post_stop_report_path=root / "missing_post_stop_report.json",
                finite_update_matrix_path=root / "missing_finite_update_matrix.csv",
            )

            self.assertEqual(report["decision"], EXISTING_ARTIFACTS_SUFFICIENT)
            self.assertEqual(
                report["coverage"]["missing_fields_for_deconfounded_no_training_audit"],
                [],
            )
            self.assertEqual(
                [
                    row["variant"]
                    for row in report["next_no_training_causal_audit_matrix"][
                        "brackets"
                    ]
                ],
                ["baseline", "rank_matched_topk1_contextual"],
            )

    def test_missing_control_requires_new_focused_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = _write_source_artifacts(root, include_dense_control=False)

            report = write_causal_audit_coverage_report(
                root / "fingerprint",
                root / "matched",
                root / "coverage",
                causal_fingerprint_report_path=reports["fingerprint"],
                bracket_report_path=reports["bracket"],
                rank_matched_report_path=reports["rank"],
                retention_report_path=reports["retention"],
                post_stop_report_path=root / "missing_post_stop_report.json",
                finite_update_matrix_path=root / "missing_finite_update_matrix.csv",
            )

            self.assertEqual(report["decision"], NEW_TRAINING_REQUIRED)
            self.assertIn(
                "norm_matched_dense",
                report["coverage"]["missing_controls_for_deconfounded_matrix"],
            )

    def test_finite_update_matrix_is_consumed_as_explicit_control_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = _write_source_artifacts(
                root,
                include_dense_control=False,
                include_per_token_fields=True,
                include_active_rank=True,
                include_residual_norm_bin=True,
            )
            finite_update_matrix = root / "finite_update_matrix.csv"
            _write_finite_update_matrix(finite_update_matrix)

            report = write_causal_audit_coverage_report(
                root / "fingerprint",
                root / "matched",
                root / "coverage",
                causal_fingerprint_report_path=reports["fingerprint"],
                bracket_report_path=reports["bracket"],
                rank_matched_report_path=reports["rank"],
                retention_report_path=reports["retention"],
                post_stop_report_path=root / "missing_post_stop_report.json",
                finite_update_matrix_path=finite_update_matrix,
            )

            coverage = report["coverage"]
            self.assertEqual(report["decision"], EXISTING_ARTIFACTS_SUFFICIENT)
            self.assertEqual(coverage["missing_controls_for_deconfounded_matrix"], [])
            self.assertTrue(coverage["controls_available"]["norm_matched_dense"])
            self.assertTrue(
                coverage["controls_available"]["finite_update_dense_active_rank"]
            )
            self.assertTrue(
                coverage["controls_available"]["finite_update_random_fixed_topk2"]
            )
            self.assertEqual(coverage["finite_update_control_matrix_row_count"], 4)
            self.assertEqual(
                [
                    row["matrix_role"]
                    for row in report["next_no_training_causal_audit_matrix"][
                        "finite_update_controls"
                    ]
                ],
                [
                    "promoted_contextual_topk2",
                    "rank_matched_contextual_topk1",
                    "random_fixed_topk2",
                    "dense_active_rank",
                ],
            )

    def test_post_stop_report_selects_rank_matched_topk1_active_bracket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = _write_source_artifacts(
                root,
                include_per_token_fields=True,
                include_active_rank=True,
                include_residual_norm_bin=True,
                include_post_stop=True,
            )

            report = write_causal_audit_coverage_report(
                root / "fingerprint",
                root / "matched",
                root / "coverage",
                causal_fingerprint_report_path=reports["fingerprint"],
                bracket_report_path=reports["bracket"],
                rank_matched_report_path=reports["rank"],
                retention_report_path=reports["retention"],
                post_stop_report_path=reports["post_stop"],
                finite_update_matrix_path=root / "missing_finite_update_matrix.csv",
            )

            self.assertEqual(report["decision"], RANK_MATCHED_TOPK1_ACTIVE_POST_STOP)
            coverage = report["coverage"]
            self.assertTrue(coverage["post_stop_rank_matched_topk1_active"])
            self.assertFalse(
                coverage["support_frequency_candidate_percentile_identified"]
            )
            matrix = report["next_no_training_causal_audit_matrix"]
            self.assertEqual(
                matrix["active_bracket"]["variant"],
                "rank_matched_topk1_contextual",
            )
            self.assertTrue(matrix["blocked_claims"]["topk2_causal_cooperation"])
            self.assertTrue(
                matrix["blocked_claims"]["support_frequency_candidate_percentile"]
            )


def _write_source_artifacts(
    root: Path,
    *,
    include_per_token_fields: bool = False,
    include_active_rank: bool = False,
    include_residual_norm_bin: bool = False,
    include_dense_control: bool = True,
    include_post_stop: bool = False,
) -> dict[str, Path]:
    audit_dir = root / "fingerprint"
    matched_dir = root / "matched"
    audit_dir.mkdir(parents=True)
    matched_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "audit": {
            "variants": [
                {"variant": "baseline", "top_k": 2, "alpha0_ce_loss": 3.0},
                {
                    "variant": "rank_matched_topk1_contextual",
                    "top_k": 1,
                    "alpha0_ce_loss": 2.9,
                },
            ],
            "pair_intervention_count": 2,
            "per_token_pair_intervention_count": 2
            if include_per_token_fields
            else 0,
            "functional_churn": [
                {
                    "variant": "baseline",
                    "previous_support_changed_logit_mse_mean": 0.3,
                },
                {
                    "variant": "rank_matched_topk1_contextual",
                    "previous_support_changed_logit_mse_mean": 0.35,
                },
            ],
        },
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    pair_fields = [
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "router_support_count",
        "pair_synergy",
        "fixed_support_residual_stream_l2_delta",
        "fixed_support_logit_mse",
    ]
    rows = [
        {
            "variant": "baseline",
            "intervention": "fixed_dominant_router_support",
            "position_bin": "all",
            "token_class": "all",
            "router_support_count": "3",
            "pair_synergy": "0.2",
            "fixed_support_residual_stream_l2_delta": "1.1",
            "fixed_support_logit_mse": "0.1",
            "token_index": "0",
            "active_rank_proxy": "2",
            "residual_norm_bin": "low",
        },
        {
            "variant": "rank_matched_topk1_contextual",
            "intervention": "fixed_dominant_router_singleton",
            "position_bin": "all",
            "token_class": "all",
            "router_support_count": "4",
            "pair_synergy": "",
            "fixed_support_residual_stream_l2_delta": "1.2",
            "fixed_support_logit_mse": "0.2",
            "token_index": "0",
            "active_rank_proxy": "1",
            "residual_norm_bin": "low",
        },
    ]
    _write_csv(audit_dir / "pair_interventions.csv", pair_fields, rows)
    per_token_fields = [
        "variant",
        "intervention",
        "support",
        "batch_index",
        "position_index",
        "token_index",
        "position_bin",
        "token_class",
        "router_support_count",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
    ]
    if include_active_rank:
        per_token_fields.append("active_rank_proxy")
    if include_residual_norm_bin:
        per_token_fields.extend(["residual_norm", "residual_norm_bin"])
    if include_per_token_fields:
        _write_csv(audit_dir / "per_token_pair_interventions.csv", per_token_fields, rows)
    _write_csv(
        audit_dir / "column_fingerprints.csv",
        [
            "variant",
            "column",
            "router_support_fraction",
            "force_residual_stream_l2_delta",
            "force_logit_mse",
        ],
        [
            {
                "variant": "baseline",
                "column": "0",
                "router_support_fraction": "0.1",
                "force_residual_stream_l2_delta": "1.0",
                "force_logit_mse": "0.2",
            }
        ],
    )
    matched_summary = {
        "status": "pass",
        "decision": "prefer_rank_matched_topk1_for_causal_audits",
        "evidence": {
            "matched_strata_count": 2,
            "matched_strata": ["all|all", "even|all"],
            "topk2_pair_synergy_mean_across_strata": 0.2,
            "topk2_functional_churn_cleaner_than_topk1": True,
            "rank_matched_topk1_router_ce_better_than_topk2": True,
        },
    }
    (matched_dir / "summary.json").write_text(
        json.dumps(matched_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        matched_dir / "matched_strata.csv",
        ["position_bin", "token_class", "topk2_pair_synergy_mean"],
        [
            {
                "position_bin": "all",
                "token_class": "all",
                "topk2_pair_synergy_mean": "0.2",
            }
        ],
    )
    fingerprint = root / "fingerprint_report.json"
    fingerprint.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "diagnose_causal_column_fingerprint_audit",
                "evidence": {
                    "signals": {
                        "random_fixed_topk2_worse_than_learned": True,
                        "norm_matched_dense_worse_than_learned": include_dense_control,
                        "rank_matched_topk1_fingerprint_present": True,
                        "exact_pair_synergy_supported": True,
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bracket = root / "bracket_report.json"
    bracket.write_text(
        json.dumps(
            {"status": "pass", "decision": "select_rank_matched_topk1_causal_audit_bracket"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rank = root / "rank_report.json"
    rank.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "diagnose_rank_matched_topk1_causal_bracket_audit",
                "evidence": {
                    "signals": {
                        "rank_matched_topk1_present": True,
                        "rank_matched_topk1_intervention_present": True,
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    retention = root / "retention_report.json"
    retention.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "diagnose_retention_churn_microtest",
                "evidence": {"signals": {"sparse_beats_dense_after_transfer": True}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    post_stop = root / "post_stop_report.json"
    if include_post_stop:
        post_stop.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "decision": "select_post_stop_rank_matched_topk1_causal_bracket",
                    "rank_matched_topk1_default_causal_audit_bracket": True,
                    "topk2_causal_cooperation_claim_supported": False,
                    "support_frequency_candidate_percentile_ready": False,
                    "support_frequency_candidate_percentile_identified": False,
                    "evidence": {
                        "support_frequency_candidate_artifact": {
                            "artifact_ready": True,
                            "candidate_row_count": 1880,
                            "calipered_candidate_row_count": 0,
                            "unmatched_candidate_row_count": 1880,
                        }
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "fingerprint": fingerprint,
        "bracket": bracket,
        "rank": rank,
        "retention": retention,
        "post_stop": post_stop,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_finite_update_matrix(path: Path) -> None:
    _write_csv(
        path,
        [
            "variant",
            "matrix_role",
            "claim_gate",
            "mean_anchor_ce_abs_delta",
            "mean_anchor_logit_mse",
            "mean_anchor_residual_stream_l2",
            "mean_anchor_support_churn",
            "per_token_commutator_rows_available",
            "available_per_token_strata",
        ],
        [
            {
                "variant": "promoted_contextual_topk2",
                "matrix_role": "promoted_contextual_topk2",
                "claim_gate": "matrix_input_only_not_causal_cooperation_evidence",
                "mean_anchor_ce_abs_delta": "0.01",
                "mean_anchor_logit_mse": "0.24",
                "mean_anchor_residual_stream_l2": "5.1",
                "mean_anchor_support_churn": "0.9",
                "per_token_commutator_rows_available": "True",
                "available_per_token_strata": "position_bin;token_class",
            },
            {
                "variant": "rank_matched_contextual_topk1",
                "matrix_role": "rank_matched_contextual_topk1",
                "claim_gate": "control_matrix_input",
                "mean_anchor_ce_abs_delta": "0.02",
                "mean_anchor_logit_mse": "0.01",
                "mean_anchor_residual_stream_l2": "1.3",
                "mean_anchor_support_churn": "0.0",
                "per_token_commutator_rows_available": "True",
                "available_per_token_strata": "position_bin;token_class",
            },
            {
                "variant": "random_fixed_topk2",
                "matrix_role": "random_fixed_topk2",
                "claim_gate": "control_matrix_input",
                "mean_anchor_ce_abs_delta": "0.05",
                "mean_anchor_logit_mse": "0.35",
                "mean_anchor_residual_stream_l2": "6.7",
                "mean_anchor_support_churn": "0.0",
                "per_token_commutator_rows_available": "True",
                "available_per_token_strata": "position_bin;token_class",
            },
            {
                "variant": "norm_matched_dense_active_rank",
                "matrix_role": "dense_active_rank",
                "claim_gate": "control_matrix_input",
                "mean_anchor_ce_abs_delta": "0.04",
                "mean_anchor_logit_mse": "0.07",
                "mean_anchor_residual_stream_l2": "3.2",
                "mean_anchor_support_churn": "",
                "per_token_commutator_rows_available": "True",
                "available_per_token_strata": "position_bin;token_class",
            },
        ],
    )


if __name__ == "__main__":
    unittest.main()
