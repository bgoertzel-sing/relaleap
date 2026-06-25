from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_context_gate_suppression_calibration_audit import (
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_context_gate_suppression_calibration_audit,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)


class ActiveTopk1ContextGateSuppressionCalibrationAuditTest(unittest.TestCase):
    def test_audit_calibrates_deployable_gate_against_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            interference = root / "interference"
            closeout = root / "closeout"
            _write_interference(interference)
            _write_closeout(closeout)

            summary = run_active_topk1_context_gate_suppression_calibration_audit(
                interference_dir=interference,
                closeout_dir=closeout,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"], CONTEXT_GATE_SUPPRESSION_CALIBRATION_PASSED
            )
            metrics = summary["evidence"]["metrics"]
            self.assertGreater(metrics["deployable_holdout_net_gain"], 0.0)
            self.assertGreater(metrics["deployable_gain_minus_ungated"], 0.0)
            self.assertGreater(
                metrics["deployable_gain_minus_coverage_matched_random"], 0.0
            )
            self.assertGreaterEqual(
                metrics["deployable_retained_gain_fraction"], 0.8
            )
            self.assertTrue(
                summary["evidence"]["signals"][
                    "deployable_gate_passes_pre_registered_criteria"
                ]
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "policy_metrics.csv").is_file())
            self.assertTrue((root / "out" / "stratum_decisions.csv").is_file())
            self.assertTrue((root / "out" / "bootstrap_intervals.csv").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_audit_fails_closed_when_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_active_topk1_context_gate_suppression_calibration_audit(
                interference_dir=root / "missing",
                closeout_dir=root / "missing_closeout",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("interference_summary", "artifact"), fields)
            self.assertIn(("runpod_closeout", "artifact"), fields)


def _write_interference(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "context_gate_reduces_offcontext_interference",
                "evidence": {
                    "signals": {
                        "own_context_singleton_gain_positive": True,
                        "offcontext_singleton_interference_present": True,
                        "matched_topk2_reference_present": True,
                        "random_control_present": True,
                        "exhaustive_control_present": True,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    index = 0
    strata = [
        ("common_target", "low", "high", 3, True),
        ("common_target", "mid", "high", 3, True),
        ("rare_target", "low", "low", 1, False),
        ("rare_target", "mid", "low", 1, False),
    ]
    for fold in range(3):
        for token_class, norm_bin, gain_bin, count, good in strata:
            for _ in range(count):
                position_index = fold + 3 * index
                rows.append(
                    {
                        "batch_index": 0,
                        "position_index": position_index,
                        "token_index": position_index,
                        "target_token": 10 + index,
                        "position_bin": "all",
                        "token_class": token_class,
                        "residual_norm_bin": norm_bin,
                        "residual_gain_bin": gain_bin,
                        "own_context_singleton_gain": 1.0 if good else -0.6,
                        "off_context_singleton_gain": -0.03 if good else -1.0,
                        "off_context_singleton_harm": 0.03 if good else 1.0,
                        "topk2_reference_gain": 0.1,
                        "random_singleton_gain": 0.0,
                        "exhaustive_singleton_gain": 1.2,
                        "has_selected_context": True,
                        "has_offcontext_match": True,
                        "has_topk2_reference": True,
                        "has_random_control": True,
                        "has_exhaustive_control": True,
                    }
                )
                index += 1
    _write_csv(path / "singleton_interference_by_context.csv", list(rows[0]), rows)
    _write_csv(
        path / "singleton_interference_by_stratum.csv",
        [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "context_count",
        ],
        [
            {
                "position_bin": "all",
                "token_class": "common_target",
                "residual_norm_bin": "low",
                "residual_gain_bin": "high",
                "context_count": 4,
            }
        ],
    )
    _write_csv(
        path / "context_gate_holdout.csv",
        [
            "position_bin",
            "token_class",
            "residual_norm_bin",
            "residual_gain_bin",
            "gate_active",
        ],
        [
            {
                "position_bin": "even",
                "token_class": "common_target",
                "residual_norm_bin": "low",
                "residual_gain_bin": "high",
                "gate_active": True,
            },
            {
                "position_bin": "all",
                "token_class": "common_target",
                "residual_norm_bin": "mid",
                "residual_gain_bin": "high",
                "gate_active": True,
            },
        ],
    )
    (path / "notes.md").write_text("notes\n", encoding="utf-8")


def _write_closeout(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {"status": "pass", "decision": RUNPOD_POST_DECOMPOSITION_VALIDATED},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
