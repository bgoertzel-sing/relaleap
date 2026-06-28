from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mlp_churn_intervention_fingerprint import _ce_for_target
from relaleap.experiments.sparse_dense_mlp_matched_intervention_decision import (
    REQUIRED_ARTIFACTS,
    run_sparse_dense_mlp_matched_intervention_decision,
)


class SparseDenseMLPMatchedInterventionDecisionTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_sparse_dense_mlp_matched_intervention_decision(
                common_benchmark_dir=root / "missing_common",
                dense_observables_dir=root / "missing_dense",
                mlp_fingerprint_dir=root / "missing_mlp",
                sparse_fingerprint_dir=root / "missing_sparse",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "sparse_dense_mlp_matched_intervention_decision_failed_closed",
            )
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_raw_rows_write_matched_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            dense = root / "dense"
            mlp = root / "mlp"
            sparse = root / "sparse"
            _write_common(common)
            _write_dense(dense)
            _write_summary(mlp, "mlp_churn_intervention_fingerprint_scaled_assay_completed")
            _write_summary(sparse, "sparse_acsr_per_token_churn_fingerprint_available")

            summary = run_sparse_dense_mlp_matched_intervention_decision(
                common_benchmark_dir=common,
                dense_observables_dir=dense,
                mlp_fingerprint_dir=mlp,
                sparse_fingerprint_dir=sparse,
                out_dir=root / "out",
                min_heldout_rows_per_arm=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertGreater(summary["scaled_intervention_row_count"], 0)
            self.assertGreater(summary["matched_comparison_row_count"], 0)
            with (root / "out" / "matched_comparisons.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["challenger_arm"] == "sparse_contextual_topk2" for row in rows))
            self.assertTrue(any(row["challenger_arm"] == "parameter_matched_causal_mlp_control" for row in rows))

    def test_mlp_beats_dense16_but_not_dense24_does_not_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            dense = root / "dense"
            mlp = root / "mlp"
            sparse = root / "sparse"
            _write_common(common)
            dense.mkdir(parents=True)
            rows = []
            rows.extend(_rows_for_arm("dense_rank16_best_norm", residual_scale=1.0, candidate_boost=0.08))
            rows.extend(_rows_for_arm("dense_rank24_best_norm", residual_scale=1.0, candidate_boost=0.18))
            rows.extend(_rows_for_arm("parameter_matched_causal_mlp_control", residual_scale=1.05, candidate_boost=0.12))
            _write_csv(dense / "per_token_observables.csv", rows)
            _write_summary(mlp, "mlp_churn_intervention_fingerprint_scaled_assay_completed")
            _write_summary(sparse, "sparse_acsr_per_token_churn_fingerprint_available")

            summary = run_sparse_dense_mlp_matched_intervention_decision(
                common_benchmark_dir=common,
                dense_observables_dir=dense,
                mlp_fingerprint_dir=mlp,
                sparse_fingerprint_dir=sparse,
                out_dir=root / "out",
                min_heldout_rows_per_arm=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["advancement_row_count"], 0)
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(
                summary["decision"],
                "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail",
            )
            with (root / "out" / "domination_cases.csv").open(newline="", encoding="utf-8") as handle:
                cases = list(csv.DictReader(handle))
            mlp_cases = [row for row in cases if row["challenger_arm"] == "parameter_matched_causal_mlp_control"]
            self.assertTrue(mlp_cases)
            self.assertTrue(all(row["challenger_advances"] == "False" for row in mlp_cases))


def _write_common(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for arm in (
        "sparse_contextual_topk2",
        "sparse_rank_matched_topk1",
        "sparse_teacher_distilled_norm_topk2",
        "sparse_frequency_matched_random_topk1",
    ):
        rows.extend(_rows_for_arm(arm, residual_scale=0.9, candidate_boost=0.15))
    _write_csv(path / "per_token_metrics.csv", rows)


def _write_dense(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    rows.extend(_rows_for_arm("dense_rank16_best_norm", residual_scale=1.0, candidate_boost=0.08))
    rows.extend(_rows_for_arm("dense_rank24_best_norm", residual_scale=1.1, candidate_boost=0.10))
    rows.extend(_rows_for_arm("parameter_matched_causal_mlp_control", residual_scale=1.8, candidate_boost=0.22))
    _write_csv(path / "per_token_observables.csv", rows)


def _rows_for_arm(arm: str, *, residual_scale: float, candidate_boost: float) -> list[dict[str, object]]:
    rows = []
    for index in range(6):
        target = index % 3
        base_logits = [0.2, -0.1, 0.4]
        candidate_logits = list(base_logits)
        candidate_logits[target] += candidate_boost
        rows.append(
            {
                "arm": arm,
                "split": "heldout" if index % 2 else "anchor",
                "token_index": index,
                "position_index": index,
                "base_ce_loss": _ce_for_target(base_logits, target),
                "ce_loss": _ce_for_target(candidate_logits, target),
                "delta_vs_base_ce": _ce_for_target(candidate_logits, target) - _ce_for_target(base_logits, target),
                "residual_update_l2": residual_scale,
                "logit_mse_vs_base": candidate_boost * candidate_boost / 3.0,
                "prediction_changed_vs_base": "False",
                "residual_update_vector": json.dumps([residual_scale, 0.0]),
                "base_logits": json.dumps(base_logits),
                "candidate_logits": json.dumps(candidate_logits),
            }
        )
    return rows


def _write_summary(path: Path, decision: str) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"status": "pass", "decision": decision}) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
