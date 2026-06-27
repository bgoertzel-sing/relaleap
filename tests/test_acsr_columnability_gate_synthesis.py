from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_columnability_gate_synthesis import (
    REQUIRED_ARTIFACTS,
    run_acsr_columnability_gate_synthesis,
)


class ACSRColumnabilityGateSynthesisTest(unittest.TestCase):
    def test_retires_sparse_identity_primary_claim_with_optional_discovery_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            review = root / "latest-review.md"
            _write_source(source, git_dirty=False)
            _write_review(review)

            summary = run_acsr_columnability_gate_synthesis(
                source_dir=source,
                strategy_review=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "retire_sparse_support_identity_primary_claim_keep_support_discovery_followup_optional",
            )
            self.assertEqual(
                summary["claim_status"],
                "sparse_support_identity_primary_claim_retired_locally",
            )
            self.assertTrue(
                summary["aggregate_metrics"]["retire_sparse_identity_primary_claim"]
            )
            self.assertTrue(
                summary["aggregate_metrics"]["secondary_support_discovery_followup_warranted"]
            )
            self.assertIn("Ben should be notified", summary["direction_shift"])
            self.assertIn("do not run RunPod yet", summary["selected_next_step"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

            with (root / "report" / "columnability_synthesis.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            oracle_row = next(row for row in rows if row["criterion"] == "oracle_support_followup")
            self.assertEqual(oracle_row["status"], "secondary_followup_warranted")

    def test_fails_closed_on_dirty_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            review = root / "latest-review.md"
            _write_source(source, git_dirty=True)
            _write_review(review)

            summary = run_acsr_columnability_gate_synthesis(
                source_dir=source,
                strategy_review=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_columnability_gate_synthesis_failed_closed",
            )
            self.assertTrue(
                any(failure["gate"] == "source_provenance_clean" for failure in summary["failures"])
            )


def _write_source(path: Path, *, git_dirty: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "fail",
        "decision": "acsr_common_causal_residual_benchmark_failed_gate",
        "claim_status": "sparse_support_specific_effect_not_separated_from_common_dense_controls",
        "selected_next_step": "synthesize the local columnability/discovery gate before any RunPod repeat or sparse-support identity claim",
        "git_dirty": git_dirty,
        "git_diff_hash": "abc123" if git_dirty else "",
        "arm_count": 8,
        "benchmark_interpretation": {
            "dense_wins_l2_matched": True,
            "teacher_distilled_sparse_beats_default_sparse": False,
            "teacher_distilled_sparse_beats_l2_matched_dense": False,
            "teacher_distilled_gap_vs_default_sparse_ce_delta": 0.23944520950317383,
            "teacher_distilled_gap_vs_l2_matched_dense_ce_delta": 0.3421361446380615,
            "teacher_distilled_mse_margin_vs_shuffled_teacher": 0.0017448607832193375,
            "target_norm_distill_mse_margin_vs_current": -0.00518491305410862,
            "target_norm_distill_gap_vs_default_sparse_ce_delta": 0.21298599243164062,
            "oracle_support_distill_mse_margin_vs_target_norm": 0.004448847845196724,
            "oracle_support_distill_gap_vs_default_sparse_ce_delta": 0.14896631240844727,
            "soft_topk_distill_mse_margin_vs_best_hard_sparse": -0.007986396551132202,
            "soft_topk_distill_gap_vs_default_sparse_ce_delta": 0.2879328727722168,
            "target_norm_distill_beats_token_position_null": True,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "arm_metrics.csv",
        [
            _arm("sparse_contextual_topk2", -0.3168487548828125),
            _arm("rank_flop_matched_causal_dense", -0.4195396900177002),
            _arm("sparse_teacher_distilled_norm_topk2", -0.07740354537963867, mse=0.020887430757284164),
            _arm("sparse_teacher_distilled_target_norm_topk2", -0.10386276245117188, mse=0.026072343811392784),
            _arm("sparse_teacher_distilled_oracle_support_topk2", -0.15957021713256836, mse=0.02165098860859871),
            _arm("sparse_teacher_distilled_soft_temperature_topk2", -0.028915882110595703, mse=0.028873827308416367),
            _arm("sparse_teacher_distilled_shuffled_teacher_null", -0.005171775817871094, mse=0.022632291540503502),
            _arm("sparse_teacher_distilled_token_position_null", -0.07566213607788086, mse=0.026116332039237022),
        ],
    )
    _write_csv(
        path / "gate_criteria.csv",
        [
            {
                "criterion": "sparse_beats_causal_dense",
                "passed": "False",
                "failure_reason": "sparse top-k2 did not beat causal dense",
            }
        ],
    )


def _arm(name: str, delta: float, *, mse: float | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "arm": name,
        "heldout_delta_vs_base_ce": delta,
    }
    if mse is not None:
        row["teacher_residual_mse"] = mse
    return row


def _write_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: major",
                "notify_ben: true",
                "recommended_next_action: Finalize local columnability/discovery synthesis.",
                "verdict: PIVOT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
