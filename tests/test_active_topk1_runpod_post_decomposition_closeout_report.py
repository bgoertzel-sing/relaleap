from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
    POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
    run_active_topk1_runpod_post_decomposition_closeout_report,
)


class ActiveTopk1RunpodPostDecompositionCloseoutReportTest(unittest.TestCase):
    def test_closeout_validates_matching_local_and_fetched_runpod_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_interference = _write_interference(root / "local_interference")
            runpod_interference = _write_interference(root / "runpod_interference")
            local_decision = _write_decision(root / "local_decision")
            checked_decision = _write_decision(root / "checked_decision")

            summary = run_active_topk1_runpod_post_decomposition_closeout_report(
                local_interference_dir=local_interference,
                runpod_interference_dir=runpod_interference,
                local_decision_dir=local_decision,
                local_checked_runpod_decision_dir=checked_decision,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], RUNPOD_POST_DECOMPOSITION_VALIDATED)
            self.assertEqual(summary["claim_status"], COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS)
            self.assertEqual(
                summary["claim_policy"], BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
            )
            self.assertTrue(
                summary["prerequisite_sync_provenance"]["synced_ignored_result_packets"]
            )
            self.assertTrue(all(row["match"] for row in summary["metric_comparison"]))
            self.assertTrue(all(row["match"] for row in summary["signal_comparison"]))
            self.assertTrue(all(row["sha256"] for row in summary["artifact_manifest"]))
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "metric_comparison.csv").is_file())
            self.assertTrue((root / "report" / "signal_comparison.csv").is_file())
            self.assertTrue((root / "report" / "artifact_manifest.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_closeout_fails_closed_on_metric_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_interference = _write_interference(root / "local_interference")
            runpod_interference = _write_interference(root / "runpod_interference")
            local_decision = _write_decision(root / "local_decision")
            checked_decision = _write_decision(root / "checked_decision")
            runpod_summary_path = runpod_interference / "summary.json"
            runpod_summary = json.loads(runpod_summary_path.read_text(encoding="utf-8"))
            runpod_summary["evidence"]["metrics"]["own_context_singleton_gain_mean"] = 0.9
            _write_json(runpod_summary_path, runpod_summary)

            summary = run_active_topk1_runpod_post_decomposition_closeout_report(
                local_interference_dir=local_interference,
                runpod_interference_dir=runpod_interference,
                local_decision_dir=local_decision,
                local_checked_runpod_decision_dir=checked_decision,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("own_context_singleton_gain_mean", fields)


def _write_interference(path: Path) -> Path:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
            "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
            "evidence": {
                "metrics": {
                    "context_count": 252,
                    "selected_context_count": 120,
                    "offcontext_context_count": 252,
                    "random_control_context_count": 252,
                    "exhaustive_control_context_count": 252,
                    "source_row_count": 106848,
                    "own_context_singleton_gain_mean": 1.0,
                    "off_context_singleton_gain_mean": -0.1,
                    "context_gated_net_gain_holdout_mean": 0.7,
                    "context_gate_gain_minus_ungated_holdout_mean": 0.4,
                    "topk2_reference_gain_mean": 0.06,
                    "random_singleton_gain_mean": 0.0,
                    "exhaustive_singleton_gain_mean": 1.4,
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
    for name in (
        "singleton_interference_by_context.csv",
        "singleton_interference_by_stratum.csv",
        "context_gate_holdout.csv",
    ):
        (path / name).write_text("field,value\nx,1\n", encoding="utf-8")
    (path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    return path


def _write_decision(path: Path) -> Path:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": POST_DECOMPOSITION_RUNPOD_VALIDATION_RECOMMENDED,
            "claim_status": COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
            "claim_policy": BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
        },
    )
    (path / "decision_sources.csv").write_text("source,status\nx,pass\n", encoding="utf-8")
    (path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    return path


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
