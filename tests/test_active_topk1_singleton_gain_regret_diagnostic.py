from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_singleton_gain_regret_diagnostic import (
    INSUFFICIENT_EVIDENCE,
    LIKELY_REAL_SINGLETON_GAIN_FAILURE,
    MATCHING_ARTIFACT_POSSIBLE,
    run_active_topk1_singleton_gain_regret_diagnostic,
)


class ActiveTopk1SingletonGainRegretDiagnosticTest(unittest.TestCase):
    def test_reports_broad_singleton_gain_failure_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_source_artifacts(
                source_dir,
                [
                    _row(0, -0.4),
                    _row(1, -0.3),
                    _row(2, -0.2),
                    _row(3, -0.1),
                ],
            )
            _write_matched_contexts(deconfounded_dir, [0, 1, 2])

            summary = run_active_topk1_singleton_gain_regret_diagnostic(
                source_audit_dir=source_dir,
                deconfounded_audit_dir=deconfounded_dir,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], LIKELY_REAL_SINGLETON_GAIN_FAILURE)
            metrics = summary["evidence"]["metrics"]
            self.assertEqual(metrics["raw_context_count"], 4)
            self.assertEqual(metrics["matched_deconfounded_context_count"], 3)
            self.assertAlmostEqual(metrics["raw_singleton_gain_mean"], -0.25)
            self.assertTrue(summary["evidence"]["signals"]["broad_negative_singleton_gain"])
            self.assertTrue((root / "out" / "singleton_gain_by_context.csv").is_file())
            self.assertTrue((root / "out" / "singleton_gain_by_stratum.csv").is_file())
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_flags_matching_artifact_when_unmatched_contexts_are_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            deconfounded_dir = root / "deconfounded"
            _write_source_artifacts(
                source_dir,
                [
                    _row(0, -0.5),
                    _row(1, -0.4),
                    _row(2, 0.8),
                    _row(3, 0.7),
                ],
            )
            _write_matched_contexts(deconfounded_dir, [0, 1])

            summary = run_active_topk1_singleton_gain_regret_diagnostic(
                source_audit_dir=source_dir,
                deconfounded_audit_dir=deconfounded_dir,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], MATCHING_ARTIFACT_POSSIBLE)
            metrics = summary["evidence"]["metrics"]
            self.assertLess(metrics["matched_context_singleton_gain_mean"], 0.0)
            self.assertGreater(metrics["unmatched_context_singleton_gain_mean"], 0.0)

    def test_fails_closed_when_source_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_active_topk1_singleton_gain_regret_diagnostic(
                source_audit_dir=root / "missing_source",
                deconfounded_audit_dir=root / "missing_deconfounded",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertGreaterEqual(len(summary["evidence"]["failures"]), 3)


def _row(context_index: int, gain: float) -> dict[str, object]:
    return {
        "variant": "rank_matched_topk1_contextual",
        "intervention": "fixed_dominant_router_singleton",
        "batch_index": 0,
        "position_index": context_index,
        "token_index": context_index,
        "target_token": context_index + 10,
        "position_bin": "even" if context_index % 2 == 0 else "odd",
        "token_class": "common_target",
        "router_support_count": 18,
        "router_loss": 3.0,
        "singleton_left_gain": gain,
        "fixed_support_loss_delta": 1.0 - gain,
        "fixed_support_logit_mse": 0.1,
        "active_rank_proxy": "1",
    }


def _write_source_artifacts(source_dir: Path, rows: list[dict[str, object]]) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "source_ok"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(source_dir / "per_token_pair_interventions.csv", list(rows[0]), rows)


def _write_matched_contexts(deconfounded_dir: Path, context_indices: list[int]) -> None:
    deconfounded_dir.mkdir(parents=True)
    rows = [
        {
            "batch_index": 0,
            "position_index": index,
            "token_index": index,
            "target_token": index + 10,
        }
        for index in context_indices
    ]
    _write_csv(
        deconfounded_dir / "paired_exact_context_deltas.csv",
        ["batch_index", "position_index", "token_index", "target_token"],
        rows,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
