from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
    INSUFFICIENT_EVIDENCE,
    POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED,
    run_active_topk1_post_decomposition_decision_report,
)


class ActiveTopk1PostDecompositionDecisionReportTest(unittest.TestCase):
    def test_report_recommends_bounded_runpod_validation(self) -> None:
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

            summary = run_active_topk1_post_decomposition_decision_report(
                interference_dir=paths["interference"],
                backend_provenance_dir=paths["provenance"],
                functional_retention_dir=paths["retention"],
                strategy_review_path=review,
                out_dir=root / "report",
                gpu_backend="runpod",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"], POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED
            )
            self.assertEqual(summary["claim_status"], COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS)
            self.assertEqual(
                summary["claim_policy"], BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
            )
            self.assertTrue(summary["decision_gate"]["validation_warrant"])
            self.assertEqual(summary["backend_plan"]["backend"], "runpod")
            self.assertTrue(summary["backend_plan"]["requires_backend_validation"])
            self.assertIn("runpod_ssh_runner.py bootstrap", summary["backend_plan"]["commands"][0])
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "minor")
            self.assertFalse(summary["strategy_review"]["notify_ben"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "decision_sources.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_report_fails_closed_when_interference_packet_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["interference"] / "summary.json").unlink()

            summary = run_active_topk1_post_decomposition_decision_report(
                interference_dir=paths["interference"],
                backend_provenance_dir=paths["provenance"],
                functional_retention_dir=paths["retention"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
                gpu_backend="runpod",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("context_conditioned_singleton_interference_audit", "summary_json"),
                fields,
            )


def _write_sources(root: Path) -> dict[str, Path]:
    interference = root / "interference"
    provenance = root / "provenance"
    retention = root / "retention"
    interference.mkdir()
    provenance.mkdir()
    retention.mkdir()
    _write_json(
        interference / "summary.json",
        {
            "status": "pass",
            "decision": CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
            "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
            "evidence": {
                "metrics": {
                    "own_context_singleton_gain_mean": 1.0,
                    "off_context_singleton_gain_mean": -0.2,
                    "context_gated_net_gain_holdout_mean": 0.6,
                    "context_gate_gain_minus_ungated_holdout_mean": 0.4,
                    "topk2_reference_gain_mean": 0.1,
                    "random_singleton_gain_mean": 0.0,
                    "exhaustive_singleton_gain_mean": 1.2,
                },
                "signals": {
                    "own_context_singleton_gain_positive": True,
                    "offcontext_singleton_interference_present": True,
                    "context_gate_holdout_net_gain_positive": True,
                    "context_gate_improves_over_ungated_holdout": True,
                    "matched_topk2_reference_present": True,
                    "random_control_present": True,
                    "exhaustive_control_present": True,
                },
            },
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
        retention / "summary.json",
        {
            "status": "pass",
            "decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
        },
    )
    return {"interference": interference, "provenance": provenance, "retention": retention}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
