from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_next_evidence_selection_report import (
    INSUFFICIENT_EVIDENCE,
    NEXT_EVIDENCE_SELECTED,
    SELECTED_MATCHED_DECONFOUNDING,
    SELECTED_RETENTION_CHURN,
    run_active_topk1_next_evidence_selection_report,
)
from relaleap.experiments.active_topk1_retention_churn_summary import (
    ACTIVE_TOPK1_RETENTION_CHURN_STABLE,
)
from relaleap.experiments.promoted_topk2_finite_update_augmented_causal_gate import (
    BLOCKED as FINITE_UPDATE_TOPK2_BLOCKED,
)
from relaleap.experiments.promoted_topk2_finite_update_control_matrix import (
    FINITE_UPDATE_CONTROL_MATRIX_READY,
)


TOPK2_NOT_SUPPORTED = "topk2_comparative_causal_cooperation_not_supported"


class ActiveTopk1NextEvidenceSelectionReportTest(unittest.TestCase):
    def test_report_selects_retention_churn_when_controls_are_adequate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run active-rank-matched top-k-1 selection",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_active_topk1_next_evidence_selection_report(
                finite_augmented_dir=paths["finite_augmented"],
                functional_retention_dir=paths["functional_retention"],
                retention_stability_dir=paths["retention_stability"],
                deconfounded_dir=paths["deconfounded"],
                finite_control_dir=paths["finite_control"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], NEXT_EVIDENCE_SELECTED)
            self.assertEqual(summary["selected_experiment"], SELECTED_RETENTION_CHURN)
            self.assertFalse(summary["selection_gate"]["requires_gpu_now"])
            self.assertFalse(summary["selection_gate"]["new_training_required"])
            self.assertTrue(summary["selection_gate"]["matched_control_coverage_adequate"])
            self.assertTrue(summary["selection_gate"]["topk2_causal_cooperation_blocked"])
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "minor")
            self.assertFalse(summary["strategy_review"]["notify_ben"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "selected_experiment.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_report_selects_matched_deconfounding_when_control_coverage_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root, include_dense_role=False)

            summary = run_active_topk1_next_evidence_selection_report(
                finite_augmented_dir=paths["finite_augmented"],
                functional_retention_dir=paths["functional_retention"],
                retention_stability_dir=paths["retention_stability"],
                deconfounded_dir=paths["deconfounded"],
                finite_control_dir=paths["finite_control"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], NEXT_EVIDENCE_SELECTED)
            self.assertEqual(summary["selected_experiment"], SELECTED_MATCHED_DECONFOUNDING)
            self.assertFalse(summary["selection_gate"]["matched_control_coverage_adequate"])

    def test_report_fails_closed_when_required_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["finite_augmented"] / "summary.json").unlink()

            summary = run_active_topk1_next_evidence_selection_report(
                finite_augmented_dir=paths["finite_augmented"],
                functional_retention_dir=paths["functional_retention"],
                retention_stability_dir=paths["retention_stability"],
                deconfounded_dir=paths["deconfounded"],
                finite_control_dir=paths["finite_control"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("finite_update_augmented_causal_gate", "summary_json"), fields)


def _write_sources(root: Path, *, include_dense_role: bool = True) -> dict[str, Path]:
    finite_augmented = root / "finite_augmented"
    functional_retention = root / "functional_retention"
    retention_stability = root / "retention_stability"
    deconfounded = root / "deconfounded"
    finite_control = root / "finite_control"
    for path in (
        finite_augmented,
        functional_retention,
        retention_stability,
        deconfounded,
        finite_control,
    ):
        path.mkdir()

    role_counts = {
        "promoted_contextual_topk2": 10,
        "rank_matched_contextual_topk1": 10,
        "random_fixed_topk2": 10,
    }
    if include_dense_role:
        role_counts["dense_active_rank"] = 10
    _write_json(
        finite_augmented / "summary.json",
        {
            "status": "pass",
            "decision": FINITE_UPDATE_TOPK2_BLOCKED,
            "metrics": {
                "augmented_strata_count": 4,
                "augmented_matched_exact_context_count": 32,
                "augmented_mean_topk2_minus_topk1_finite_logit_mse": 0.2,
                "augmented_mean_topk2_minus_dense_finite_logit_mse": 0.1,
                "augmented_mean_topk2_finite_support_churn_fraction": 0.9,
            },
            "role_counts": role_counts,
        },
    )
    _write_json(
        functional_retention / "summary.json",
        {
            "status": "pass",
            "decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
            "evidence": {
                "aggregates": {
                    "mean_topk1_anchor_support_churn_after_transfer": 0.01,
                    "mean_topk2_anchor_support_churn_after_transfer": 0.8,
                    "mean_transfer_improvement_advantage_topk1_vs_topk2": 0.03,
                    "mean_transfer_improvement_advantage_topk1_vs_dense": 0.5,
                    "mean_commutator_anchor_logit_mse_advantage_topk1_vs_topk2": 0.2,
                    "mean_commutator_anchor_logit_mse_advantage_topk1_vs_dense": 0.05,
                },
                "claim_signals": {
                    "support_identity_churn_cleaner_than_topk2": True,
                    "functional_logit_churn_not_higher_than_topk2": True,
                    "finite_update_commutator_not_worse_than_topk2": True,
                    "transfer_improvement_beats_dense_control": True,
                },
            },
        },
    )
    _write_json(
        retention_stability / "summary.json",
        {
            "status": "pass",
            "decision": ACTIVE_TOPK1_RETENTION_CHURN_STABLE,
        },
    )
    _write_json(
        deconfounded / "summary.json",
        {
            "status": "pass",
            "decision": TOPK2_NOT_SUPPORTED,
            "evidence": {
                "metrics": {
                    "matched_exact_context_count": 20,
                    "topk2_incremental_pair_gain_positive_strata_fraction": 0.6,
                    "topk2_fixed_support_cleaner_strata_fraction": 0.6,
                }
            },
        },
    )
    _write_json(
        finite_control / "summary.json",
        {
            "status": "pass",
            "decision": FINITE_UPDATE_CONTROL_MATRIX_READY,
            "metrics": {"topk2_minus_topk1_logit_mse": 0.2},
        },
    )
    return {
        "finite_augmented": finite_augmented,
        "functional_retention": functional_retention,
        "retention_stability": retention_stability,
        "deconfounded": deconfounded,
        "finite_control": finite_control,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
