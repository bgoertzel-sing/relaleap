from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_support_discovery_gate import (
    REQUIRED_ARTIFACTS,
    run_acsr_support_discovery_gate,
)


class ACSRSupportDiscoveryGateTest(unittest.TestCase):
    def test_blocks_deployable_claim_pending_learned_support_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis"
            benchmark = root / "benchmark"
            design = root / "design"
            review = root / "latest-review.md"
            _write_synthesis(synthesis)
            _write_benchmark(benchmark, git_dirty=False)
            _write_design(design)
            _write_review(review)

            summary = run_acsr_support_discovery_gate(
                synthesis_dir=synthesis,
                benchmark_dir=benchmark,
                design_dir=design,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "support_discovery_gate_blocks_deployable_claim_pending_learned_head",
            )
            self.assertEqual(
                summary["claim_status"],
                "deployable_support_discovery_not_established_sparse_identity_retired",
            )
            self.assertIn("do not run RunPod yet", summary["selected_next_step"])
            self.assertIn("Ben should be notified", summary["direction_shift"])
            blocker_names = {row["criterion"] for row in summary["claim_blockers"]}
            self.assertIn("support_head_shuffled_feature_null_present", blocker_names)
            self.assertIn("same_student_forcing_present", blocker_names)
            self.assertIn("oracle_support_headroom_positive", blocker_names)
            self.assertFalse(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

            with (root / "out" / "null_controls.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            missing_null = next(row for row in rows if row["control"] == "learned_support_head_shuffled_feature_null")
            self.assertEqual(missing_null["present"], "False")

    def test_fails_closed_on_dirty_benchmark_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            synthesis = root / "synthesis"
            benchmark = root / "benchmark"
            design = root / "design"
            review = root / "latest-review.md"
            _write_synthesis(synthesis)
            _write_benchmark(benchmark, git_dirty=True)
            _write_design(design)
            _write_review(review)

            summary = run_acsr_support_discovery_gate(
                synthesis_dir=synthesis,
                benchmark_dir=benchmark,
                design_dir=design,
                strategy_review=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "support_discovery_gate_failed_closed")
            self.assertTrue(
                any(failure["criterion"] == "source_provenance_clean" for failure in summary["failures"])
            )


def _write_synthesis(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "decision": "retire_sparse_support_identity_primary_claim_keep_support_discovery_followup_optional",
        "claim_status": "sparse_support_identity_primary_claim_retired_locally",
        "git_commit": "abc123",
        "aggregate_metrics": {
            "retire_sparse_identity_primary_claim": True,
            "secondary_support_discovery_followup_warranted": True,
            "teacher_distill_gap_vs_default_sparse_ce_delta": 0.23944520950317383,
            "oracle_support_mse_margin_vs_target_norm": 0.004448847845196724,
        },
    }
    _write_json(path / "summary.json", summary)


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
    _write_json(path / "summary.json", summary)
    _write_csv(
        path / "arm_metrics.csv",
        [
            _arm("sparse_contextual_topk2", -0.3168487548828125),
            _arm("sparse_oracle_support", -0.3192157745361328),
            _arm("sparse_token_position_null", -0.03207588195800781),
            _arm("sparse_shuffled_support_marginals", -0.022758007049560547),
            _arm("sparse_frequency_matched_random", -0.019070148468017578),
            _arm("sparse_teacher_distilled_norm_topk2", -0.07740354537963867, mse=0.020887430757284164),
            _arm("sparse_teacher_distilled_target_norm_topk2", -0.10386276245117188, mse=0.026072343811392784),
            _arm("sparse_teacher_distilled_token_position_null", -0.07566213607788086, mse=0.026116330176591873),
            _arm("sparse_teacher_distilled_shuffled_teacher_null", -0.005171775817871094, mse=0.022632291540503502),
            _arm("rank_flop_matched_causal_dense", -0.4195396900177002),
            _arm("rank_flop_matched_shuffled_causal_feature_dense_null", 0.013015270233154297),
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


def _write_design(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "decision": "support_discovery_followup_design_recorded_identity_claim_stays_retired",
        "claim_status": "design_only_support_discovery_not_established_sparse_identity_retired",
        "git_commit": "abc123",
    }
    _write_json(path / "summary.json", summary)


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


def _arm(name: str, delta: float, *, mse: float | None = None) -> dict[str, object]:
    row: dict[str, object] = {"arm": name, "heldout_delta_vs_base_ce": delta}
    if mse is not None:
        row["teacher_residual_mse"] = mse
    return row


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
