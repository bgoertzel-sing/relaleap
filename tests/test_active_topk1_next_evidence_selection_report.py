from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_next_evidence_selection_report import (
    INSUFFICIENT_EVIDENCE,
    NEXT_EVIDENCE_SELECTED,
    SELECTED_EXPERIMENT,
    run_active_topk1_next_evidence_selection_report,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


class ActiveTopk1NextEvidenceSelectionReportTest(unittest.TestCase):
    def test_report_selects_context_gate_suppression_calibration_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: run context-conditioned singleton interference decomposition",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_active_topk1_next_evidence_selection_report(
                closeout_dir=paths["closeout"],
                direction_dir=paths["direction"],
                retention_dir=paths["retention"],
                retention_stability_dir=paths["retention_stability"],
                provenance_dir=paths["provenance"],
                causal_bracket_dir=paths["causal_bracket"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], NEXT_EVIDENCE_SELECTED)
            self.assertEqual(summary["selected_experiment"], SELECTED_EXPERIMENT)
            self.assertEqual(summary["claim_status"], COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS)
            self.assertEqual(
                summary["claim_policy"], BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
            )
            self.assertFalse(summary["selection_gate"]["requires_gpu_now"])
            self.assertFalse(summary["selection_gate"]["new_training_required"])
            self.assertIn(
                "offcontext_singleton_interference_remains_present",
                summary["selection_gate"]["selected_because"],
            )
            self.assertIn(
                "offcontext_harm_suppression_metric",
                {row["component"] for row in summary["experiment_design"]["components"]},
            )
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "minor")
            self.assertFalse(summary["strategy_review"]["notify_ben"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "selected_experiment.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_report_fails_closed_when_closeout_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["closeout"] / "summary.json").unlink()

            summary = run_active_topk1_next_evidence_selection_report(
                closeout_dir=paths["closeout"],
                direction_dir=paths["direction"],
                retention_dir=paths["retention"],
                retention_stability_dir=paths["retention_stability"],
                provenance_dir=paths["provenance"],
                causal_bracket_dir=paths["causal_bracket"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("runpod_post_decomposition_closeout", "summary_json"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    closeout = root / "closeout"
    direction = root / "direction"
    retention = root / "retention"
    retention_stability = root / "retention_stability"
    provenance = root / "provenance"
    causal_bracket = root / "causal_bracket"
    for path in (
        closeout,
        direction,
        retention,
        retention_stability,
        provenance,
        causal_bracket,
    ):
        path.mkdir()
    _write_json(
        closeout / "summary.json",
        {
            "status": "pass",
            "decision": RUNPOD_POST_DECOMPOSITION_VALIDATED,
            "claim_status": COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
            "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
            "metric_comparison": [
                _metric("own_context_singleton_gain_mean", 1.0),
                _metric("off_context_singleton_gain_mean", -0.2),
                _metric("context_gated_net_gain_holdout_mean", 0.7),
                _metric("context_gate_gain_minus_ungated_holdout_mean", 0.4),
                _metric("topk2_reference_gain_mean", 0.1),
                _metric("random_singleton_gain_mean", 0.0),
                _metric("exhaustive_singleton_gain_mean", 1.2),
            ],
            "signal_comparison": [
                _signal("own_context_singleton_gain_positive", True),
                _signal("offcontext_singleton_interference_present", True),
                _signal("context_gate_holdout_net_gain_positive", True),
                _signal("context_gate_improves_over_ungated_holdout", True),
                _signal("matched_topk2_reference_present", True),
                _signal("random_control_present", True),
                _signal("exhaustive_control_present", True),
            ],
        },
    )
    _write_json(
        direction / "summary.json",
        {
            "status": "pass",
            "decision": "post_bracket_direction_selected",
        },
    )
    _write_json(
        retention / "summary.json",
        {
            "status": "pass",
            "decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
            "claim_status": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        },
    )
    _write_json(
        retention_stability / "summary.json",
        {
            "status": "pass",
            "decision": "active_topk1_retention_churn_stable_across_local_seeds",
        },
    )
    _write_json(
        provenance / "summary.json",
        {
            "status": "pass",
            "decision": ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
        },
    )
    _write_json(
        causal_bracket / "decision_report.json",
        {
            "status": "pass",
            "decision": "confirm_active_rank_matched_topk1_causal_bracket",
        },
    )
    return {
        "closeout": closeout,
        "direction": direction,
        "retention": retention,
        "retention_stability": retention_stability,
        "provenance": provenance,
        "causal_bracket": causal_bracket,
    }


def _metric(field: str, value: float) -> dict[str, object]:
    return {"field": field, "local": value, "runpod": value, "match": True}


def _signal(field: str, value: bool) -> dict[str, object]:
    return {"field": field, "local": value, "runpod": value, "match": True}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
