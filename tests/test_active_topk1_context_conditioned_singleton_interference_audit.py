from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit import (
    CONTEXT_CONDITIONED_SINGLETON_INTERFERENCE_AUDIT_ESTABLISHED,
    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_context_conditioned_singleton_interference_audit,
)
from relaleap.experiments.active_topk1_post_bracket_research_direction_report import (
    POST_BRACKET_DIRECTION_SELECTED,
    SELECTED_EXPERIMENT,
)


class ActiveTopk1ContextConditionedSingletonInterferenceAuditTest(unittest.TestCase):
    def test_audit_decomposes_context_conditioned_singleton_interference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            direction = root / "direction"
            _write_source(source)
            _write_direction(direction)

            summary = run_active_topk1_context_conditioned_singleton_interference_audit(
                source_audit_dir=source,
                direction_report_dir=direction,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {
                    CONTEXT_CONDITIONED_SINGLETON_INTERFERENCE_AUDIT_ESTABLISHED,
                    CONTEXT_GATE_REDUCES_OFFCONTEXT_INTERFERENCE,
                },
            )
            metrics = summary["evidence"]["metrics"]
            self.assertEqual(metrics["context_count"], 4)
            self.assertEqual(metrics["selected_context_count"], 4)
            self.assertEqual(metrics["offcontext_context_count"], 4)
            self.assertEqual(metrics["random_control_context_count"], 4)
            self.assertEqual(metrics["exhaustive_control_context_count"], 4)
            self.assertAlmostEqual(metrics["own_context_singleton_gain_mean"], 0.475)
            self.assertAlmostEqual(metrics["off_context_singleton_gain_mean"], -0.25)
            self.assertTrue(summary["evidence"]["signals"]["own_context_singleton_gain_positive"])
            self.assertTrue(
                summary["evidence"]["signals"]["offcontext_singleton_interference_present"]
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "singleton_interference_by_context.csv").is_file())
            self.assertTrue((root / "out" / "singleton_interference_by_stratum.csv").is_file())
            self.assertTrue((root / "out" / "context_gate_holdout.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_audit_fails_closed_when_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_active_topk1_context_conditioned_singleton_interference_audit(
                source_audit_dir=root / "missing",
                direction_report_dir=root / "missing_direction",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["evidence"]["failures"]}
            self.assertIn("source_summary_json", fields)
            self.assertIn("source_per_token_pair_interventions_csv", fields)
            self.assertIn("direction_report_summary_json", fields)


def _write_direction(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": POST_BRACKET_DIRECTION_SELECTED,
                "selected_experiment": SELECTED_EXPERIMENT,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_source(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "ok", "decision": None}) + "\n",
        encoding="utf-8",
    )
    rows = []
    selected_gains = [0.5, 0.4, 0.6, 0.4]
    offcontext_gains = [-0.2, -0.3, -0.1, -0.4]
    for context_index, selected_gain in enumerate(selected_gains):
        rows.append(_row(context_index, "baseline", "fixed_dominant_router_support", "2,9", 0.7))
        rows.append(
            _row(
                context_index,
                "rank_matched_topk1_contextual",
                "fixed_dominant_router_singleton",
                "2",
                selected_gain,
                router_matches=True,
            )
        )
        rows.append(
            _row(
                context_index,
                "rank_matched_topk1_contextual",
                "fixed_dominant_router_singleton",
                "8",
                offcontext_gains[context_index],
                router_matches=False,
            )
        )
        rows.append(
            _row(
                context_index,
                "rank_matched_topk1_contextual",
                "fixed_best_singleton_swap",
                "3",
                selected_gain + 0.1,
                router_matches=False,
            )
        )
        rows.append(
            _row(
                context_index,
                "rank_matched_topk1_contextual",
                "fixed_random_singleton_control",
                "4",
                0.05,
                router_matches=False,
            )
        )
        rows.append(
            _row(
                context_index,
                "rank_matched_topk1_contextual",
                "fixed_exhaustive_singleton",
                "3",
                selected_gain + 0.1,
                router_matches=False,
            )
        )
    _write_csv(path / "per_token_pair_interventions.csv", list(rows[0]), rows)


def _row(
    context_index: int,
    variant: str,
    intervention: str,
    support: str,
    gain: float,
    *,
    router_matches: bool = False,
) -> dict[str, object]:
    empty_loss = 4.0
    return {
        "variant": variant,
        "intervention": intervention,
        "support": support,
        "batch_index": 0,
        "position_index": context_index,
        "token_index": context_index,
        "target_token": 10 + context_index,
        "position_bin": "even" if context_index % 2 == 0 else "odd",
        "token_class": "common_target",
        "residual_norm_bin": "low",
        "residual_gain_bin": "high",
        "router_support_matches_fixed": router_matches,
        "empty_loss": empty_loss,
        "router_loss": empty_loss - 0.8,
        "fixed_support_loss": empty_loss - gain,
        "singleton_left_gain": gain,
        "pair_gain": gain,
        "fixed_support_logit_mse": 0.1,
        "fixed_support_residual_stream_l2_delta": 1.0,
        "active_rank_proxy": 1 if variant == "rank_matched_topk1_contextual" else 2,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
