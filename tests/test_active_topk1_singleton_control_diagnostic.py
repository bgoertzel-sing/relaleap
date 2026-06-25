from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_singleton_control_diagnostic import (
    INSUFFICIENT_EVIDENCE,
    LIKELY_ROUTER_SELECTION_FAILURE,
    LIKELY_SINGLETON_CAPACITY_OR_VALUE_FAILURE,
    run_active_topk1_singleton_control_diagnostic,
)


class ActiveTopk1SingletonControlDiagnosticTest(unittest.TestCase):
    def test_reports_router_selection_failure_when_logged_oracle_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            _write_source_artifacts(
                source_dir,
                [
                    _row(0, "fixed_dominant_router_singleton", "2", -0.4, 4.4, True),
                    _row(0, "fixed_best_singleton_swap", "7", 0.3, 3.7, False),
                    _row(1, "fixed_dominant_router_singleton", "2", -0.2, 4.2, True),
                    _row(1, "fixed_best_singleton_swap", "8", 0.4, 3.6, False),
                    _row(2, "fixed_dominant_router_singleton", "3", -0.3, 4.3, True),
                    _row(2, "fixed_best_singleton_swap", "7", 0.2, 3.8, False),
                ],
            )

            summary = run_active_topk1_singleton_control_diagnostic(
                source_audit_dir=source_dir,
                out_dir=root / "out",
                regret_threshold=0.05,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], LIKELY_ROUTER_SELECTION_FAILURE)
            metrics = summary["evidence"]["metrics"]
            self.assertEqual(metrics["context_count"], 3)
            self.assertAlmostEqual(metrics["selected_singleton_gain_mean"], -0.3)
            self.assertAlmostEqual(metrics["logged_oracle_singleton_gain_mean"], 0.3)
            self.assertEqual(
                summary["evidence"]["missing_controls"]["random_singleton_control"],
                "missing: current source artifact does not include random singleton rows",
            )
            provenance = summary["evidence"]["provenance"]
            self.assertEqual(provenance["variant"], "rank_matched_topk1_contextual")
            self.assertIn("positive means", provenance["gain_sign_convention"])
            self.assertIsNotNone(provenance["source_summary_sha256"])
            self.assertTrue((root / "out" / "singleton_control_by_context.csv").is_file())
            self.assertTrue((root / "out" / "singleton_control_by_stratum.csv").is_file())
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_reports_singleton_capacity_or_value_failure_when_oracle_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            _write_source_artifacts(
                source_dir,
                [
                    _row(0, "fixed_dominant_router_singleton", "2", -0.4, 4.4, True),
                    _row(0, "fixed_best_singleton_swap", "7", -0.1, 4.1, False),
                    _row(1, "fixed_dominant_router_singleton", "2", -0.2, 4.2, True),
                    _row(1, "fixed_best_singleton_swap", "8", -0.05, 4.05, False),
                ],
            )

            summary = run_active_topk1_singleton_control_diagnostic(
                source_audit_dir=source_dir,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                LIKELY_SINGLETON_CAPACITY_OR_VALUE_FAILURE,
            )

    def test_fails_closed_when_source_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_active_topk1_singleton_control_diagnostic(
                source_audit_dir=root / "missing_source",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertGreaterEqual(len(summary["evidence"]["failures"]), 2)


def _row(
    context_index: int,
    intervention: str,
    support: str,
    gain: float,
    fixed_loss: float,
    router_matches: bool,
) -> dict[str, object]:
    return {
        "variant": "rank_matched_topk1_contextual",
        "load_balance_weight": 0.0,
        "intervention": intervention,
        "support": support,
        "batch_index": 0,
        "position_index": context_index,
        "token_index": context_index,
        "target_token": context_index + 10,
        "position_bin": "even" if context_index % 2 == 0 else "odd",
        "token_class": "common_target",
        "router_support_count": 18,
        "router_support_matches_fixed": router_matches,
        "empty_loss": 4.0,
        "router_loss": 3.0,
        "singleton_left_loss": fixed_loss,
        "singleton_left_gain": gain,
        "fixed_support_loss": fixed_loss,
        "fixed_support_loss_delta": fixed_loss - 3.0,
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
