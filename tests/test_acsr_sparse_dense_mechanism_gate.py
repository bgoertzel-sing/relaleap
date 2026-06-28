from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_sparse_dense_mechanism_gate import (
    REQUIRED_ARTIFACTS,
    run_acsr_sparse_dense_mechanism_gate,
)


class ACSRSparseDenseMechanismGateTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_acsr_sparse_dense_mechanism_gate(
                acsr_dir=root / "missing_acsr",
                dense_matrix_dir=root / "missing_dense",
                dense_synthesis_path=root / "missing_dense_summary.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_sparse_dense_mechanism_gate_failed_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_available_sources_block_on_missing_dense_mechanism_observables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            acsr = root / "acsr"
            dense = root / "dense"
            _write_acsr_packet(acsr)
            _write_dense_matrix(dense)
            dense_synthesis = root / "dense_synthesis.json"
            dense_synthesis.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "decision": "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            review = root / "latest-review.md"
            review.write_text(
                "strategic_change_level: minor\n"
                "notify_ben: false\n"
                "recommended_next_action: Replace artifact-contract loop with sparse-vs-dense gate\n",
                encoding="utf-8",
            )

            summary = run_acsr_sparse_dense_mechanism_gate(
                acsr_dir=acsr,
                dense_matrix_dir=dense,
                dense_synthesis_path=dense_synthesis,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "acsr_sparse_dense_mechanism_gate_blocked")
            self.assertEqual(
                summary["claim_status"],
                "sparse_mechanism_claim_blocked_by_dense_observable_gap",
            )
            self.assertFalse(summary["promotion_allowed"])
            self.assertIn("dense-control extractor", summary["selected_next_step"])
            self.assertTrue(
                any(
                    row["criterion"] == "dense_mechanism_observables_present"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            self.assertTrue(
                any(
                    row["arm"] == "dense_rank16_best_norm"
                    and "anchor_kl_or_logit_mse" in row["missing_mechanism_fields"]
                    for row in summary["mechanism_metrics"]
                )
            )
            metrics_text = (root / "out" / "mechanism_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("acsr_mlp_predicted_future", metrics_text)
            self.assertIn("dense_rank24_best_norm", metrics_text)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Promotion allowed: `False`", notes)


def _write_acsr_packet(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "anticipatory_contextual_support_routing_smoke_completed"})
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "router_metrics.csv",
        [
            _router_row("full_context_contextual_topk2_teacher", 2.1, 0.0),
            _router_row("causal_feature_safe_contextual_topk2", 2.4, 0.3),
            _router_row("acsr_mlp_predicted_future", 2.2, 0.1),
            _router_row("shuffled_predicted_features", 3.1, 1.0),
            _router_row("token_position_only_predicted_features", 2.7, 0.5),
            _router_row("random_fixed_topk2", 3.4, 1.3),
            _router_row("parameter_matched_causal_mlp_control", 2.19, 0.09),
        ],
    )
    _write_csv(
        path / "same_student_metrics.csv",
        [
            {
                "comparison": "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features",
                "acsr_minus_control_ce_loss": -0.9,
            }
        ],
    )
    _write_csv(
        path / "retention_churn_metrics.csv",
        [
            {
                "phase": "second_context_transfer",
                "variant": "acsr_mlp_predicted_future",
                "anchor_logit_mse_after_transfer": 0.01,
                "anchor_support_churn_after_transfer": 0.2,
                "anchor_ce_drift": -0.05,
            }
        ],
    )
    _write_csv(
        path / "feature_perturbation.csv",
        [{"control_type": "future_perturbation_negative", "passed": True}],
    )
    _write_csv(
        path / "parameter_counts.csv",
        [
            {
                "component": "residual_columns",
                "active_parameter_count": 768,
                "status": "available",
            },
            {
                "component": "acsr_mlp_predictor_plus_contextual_router",
                "active_parameter_count": 113228,
                "status": "available",
            },
            {
                "component": "parameter_matched_causal_mlp_control",
                "active_parameter_count": 113078,
                "status": "available",
            },
        ],
    )


def _write_dense_matrix(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "dense_rank_norm_matrix_completed"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "rank_summary.csv",
        [
            {
                "rank": 16,
                "best_arm": "dense_causal_rank16_norm_scale_1.00",
                "best_delta_minus_sparse_topk2": -0.01,
                "beats_sparse_topk2": True,
            },
            {
                "rank": 24,
                "best_arm": "dense_causal_rank24_norm_scale_1.00",
                "best_delta_minus_sparse_topk2": -0.05,
                "beats_sparse_topk2": True,
            },
        ],
    )
    _write_csv(
        path / "matrix_metrics.csv",
        [
            _dense_row("dense_causal_rank16_norm_scale_1.00", 16, 3.8, 1.0, 6192),
            _dense_row("dense_causal_rank24_norm_scale_1.00", 24, 3.7, 1.1, 9288),
        ],
    )
    _write_csv(
        path / "per_token_metrics.csv",
        [{"arm": "dense_causal_rank16_norm_scale_1.00", "token_index": 0}],
    )


def _router_row(variant: str, ce_loss: float, oracle_regret: float) -> dict[str, object]:
    return {
        "variant": variant,
        "top_k": 2,
        "ce_loss": ce_loss,
        "oracle_regret": oracle_regret,
    }


def _dense_row(
    arm: str,
    rank: int,
    heldout_ce_loss: float,
    heldout_residual_update_l2: float,
    active_params_proxy: int,
) -> dict[str, object]:
    return {
        "arm": arm,
        "rank": rank,
        "heldout_ce_loss": heldout_ce_loss,
        "heldout_residual_update_l2": heldout_residual_update_l2,
        "active_params_proxy": active_params_proxy,
        "heldout_delta_minus_sparse_topk2": -0.01,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
