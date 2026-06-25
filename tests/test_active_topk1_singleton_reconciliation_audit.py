from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_singleton_reconciliation_audit,
)


class ActiveTopk1SingletonReconciliationAuditTest(unittest.TestCase):
    def test_reconciles_incontext_gain_with_offcontext_interference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            control_dir = root / "control"
            gain_dir = root / "gain"
            _write_source_artifacts(
                source_dir,
                [
                    _row(0, "fixed_dominant_router_singleton", "2", 0.5, True),
                    _row(0, "fixed_best_singleton_swap", "7", 0.7, False),
                    _row(0, "fixed_dominant_router_singleton", "8", -0.3, False),
                    _row(1, "fixed_dominant_router_singleton", "3", 0.4, True),
                    _row(1, "fixed_best_singleton_swap", "9", 0.6, False),
                    _row(1, "fixed_dominant_router_singleton", "6", -0.2, False),
                    _row(2, "fixed_dominant_router_singleton", "4", -0.5, False),
                ],
            )
            _write_summary(control_dir, {"status": "pass", "decision": "mixed_singleton_control_evidence"})
            _write_summary(gain_dir, {"status": "pass", "decision": "likely_real_singleton_gain_failure_mode"})

            summary = run_active_topk1_singleton_reconciliation_audit(
                source_audit_dir=source_dir,
                control_audit_dir=control_dir,
                gain_regret_audit_dir=gain_dir,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
            )
            metrics = summary["evidence"]["metrics"]
            self.assertEqual(metrics["context_count"], 3)
            self.assertEqual(metrics["selected_context_count"], 2)
            self.assertEqual(metrics["offcontext_context_count"], 3)
            self.assertAlmostEqual(metrics["selected_singleton_gain_mean"], 0.45)
            self.assertAlmostEqual(
                metrics["offcontext_fixed_dominant_singleton_gain_mean"], -1.0 / 3.0
            )
            self.assertEqual(
                summary["evidence"]["missing_controls"]["random_singleton_control"],
                "missing: current source artifact does not include random singleton rows",
            )
            provenance = summary["evidence"]["provenance"]
            self.assertIn("positive means", provenance["gain_sign_convention"])
            self.assertIsNotNone(provenance["source_per_token_pair_interventions_sha256"])
            self.assertTrue((root / "out" / "singleton_reconciliation_by_context.csv").is_file())
            self.assertTrue((root / "out" / "singleton_reconciliation_by_stratum.csv").is_file())
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_fails_closed_when_inputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_active_topk1_singleton_reconciliation_audit(
                source_audit_dir=root / "missing_source",
                control_audit_dir=root / "missing_control",
                gain_regret_audit_dir=root / "missing_gain",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertGreaterEqual(len(summary["evidence"]["failures"]), 4)


def _row(
    context_index: int,
    intervention: str,
    support: str,
    gain: float,
    router_matches: bool,
) -> dict[str, object]:
    empty_loss = 4.0
    return {
        "variant": "rank_matched_topk1_contextual",
        "intervention": intervention,
        "support": support,
        "batch_index": 0,
        "position_index": context_index,
        "token_index": context_index,
        "target_token": context_index + 10,
        "position_bin": "even" if context_index % 2 == 0 else "odd",
        "token_class": "common_target",
        "router_support_matches_fixed": router_matches,
        "empty_loss": empty_loss,
        "fixed_support_loss": empty_loss - gain,
        "singleton_left_gain": gain,
    }


def _write_source_artifacts(source_dir: Path, rows: list[dict[str, object]]) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "summary.json").write_text(
        json.dumps({"status": "ok", "decision": None}) + "\n",
        encoding="utf-8",
    )
    _write_csv(source_dir / "per_token_pair_interventions.csv", list(rows[0]), rows)


def _write_summary(path: Path, value: dict[str, object]) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
