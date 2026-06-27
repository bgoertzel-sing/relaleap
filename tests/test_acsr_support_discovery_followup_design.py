from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_support_discovery_followup_design import (
    REQUIRED_ARTIFACTS,
    run_acsr_support_discovery_followup_design,
)


class ACSRSpecSupportDiscoveryFollowupDesignTest(unittest.TestCase):
    def test_records_design_after_sparse_identity_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis"
            benchmark = root / "benchmark"
            review = root / "latest-review.md"
            _write_synthesis(synthesis, retired=True)
            _write_benchmark(benchmark, git_dirty=False)
            _write_review(review)

            summary = run_acsr_support_discovery_followup_design(
                synthesis_dir=synthesis,
                benchmark_dir=benchmark,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "support_discovery_followup_design_recorded_identity_claim_stays_retired",
            )
            self.assertEqual(
                summary["claim_status"],
                "design_only_support_discovery_not_established_sparse_identity_retired",
            )
            self.assertIn("no RunPod", summary["selected_next_step"])
            self.assertIn("Ben should be notified", summary["direction_shift"])
            self.assertTrue(
                any(
                    row["component"] == "oracle_support_headroom"
                    for row in summary["support_discovery_design"]
                )
            )
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_fails_closed_if_identity_not_retired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis"
            benchmark = root / "benchmark"
            review = root / "latest-review.md"
            _write_synthesis(synthesis, retired=False)
            _write_benchmark(benchmark, git_dirty=False)
            _write_review(review)

            summary = run_acsr_support_discovery_followup_design(
                synthesis_dir=synthesis,
                benchmark_dir=benchmark,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "support_discovery_followup_design_failed_closed",
            )
            self.assertTrue(
                any(
                    failure["criterion"] == "identity_claim_retired_before_followup"
                    for failure in summary["failures"]
                )
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_synthesis(path: Path, *, retired: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "decision": "retire_sparse_support_identity_primary_claim_keep_support_discovery_followup_optional",
        "claim_status": (
            "sparse_support_identity_primary_claim_retired_locally"
            if retired
            else "sparse_support_identity_not_retired"
        ),
        "git_commit": "abc123",
        "aggregate_metrics": {
            "dense_minus_sparse_ce_delta": -0.1026909351348877,
            "teacher_distill_gap_vs_default_sparse_ce_delta": 0.23944520950317383,
            "teacher_distill_mse_margin_vs_shuffled_teacher": 0.0017448607832193375,
            "target_norm_mse_margin_vs_current": -0.00518491305410862,
            "oracle_support_mse_margin_vs_target_norm": 0.004448847845196724,
            "soft_topk_mse_margin_vs_best_hard_sparse": -0.007986396551132202,
            "retire_sparse_identity_primary_claim": retired,
            "secondary_support_discovery_followup_warranted": True,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_benchmark(path: Path, *, git_dirty: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "fail",
        "decision": "acsr_common_causal_residual_benchmark_failed_gate",
        "claim_status": "sparse_support_specific_effect_not_separated_from_common_dense_controls",
        "git_dirty": git_dirty,
        "git_diff_hash": "dirty" if git_dirty else "",
        "benchmark_interpretation": {
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
            _arm("sparse_oracle_support", -0.31921982765197754),
            _arm("sparse_teacher_distilled_target_norm_topk2", -0.10386276245117188, mse=0.026072343811392784),
            _arm("sparse_teacher_distilled_oracle_support_topk2", -0.16788244247436523, mse=0.02162349596619606),
            _arm("sparse_teacher_distilled_token_position_null", -0.07566213607788086, mse=0.026116330176591873),
        ],
    )
    _write_csv(
        path / "intervention_fingerprints.csv",
        [
            {
                "arm": "sparse_contextual_topk2",
                "support_overlap_vs_oracle": 0.25,
                "support_overlap_vs_random": 0.12,
            }
        ],
    )


def _arm(name: str, delta: float, *, mse: float | None = None) -> dict[str, object]:
    row: dict[str, object] = {"arm": name, "heldout_delta_vs_base_ce": delta}
    if mse is not None:
        row["teacher_residual_mse"] = mse
    return row


def _write_review(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: major",
                "notify_ben: true",
                "recommended_next_action: keep support discovery local before RunPod",
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
