from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from relaleap.experiments.synthetic_task_free_continual_learning_probe import (
    REQUIRED_ARTIFACTS,
    run_synthetic_task_free_continual_learning_probe,
)


class SyntheticTaskFreeContinualLearningProbeTest(unittest.TestCase):
    def test_probe_supports_sparse_retention_candidate_when_topk1_beats_dense(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            decision = root / "decision"
            _write_config(config)
            _write_decision(decision)

            with mock.patch(
                "relaleap.experiments.synthetic_task_free_continual_learning_probe"
                ".run_retention_churn_microtest",
                return_value=_microtest(
                    dataset="synthetic_rule_stream",
                    topk1_anchor_ce_drift=-0.08,
                    dense_anchor_ce_drift=0.02,
                ),
            ):
                summary = run_synthetic_task_free_continual_learning_probe(
                    config_path=config,
                    decision_dir=decision,
                    out_dir=root / "out",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "synthetic_sparse_retention_candidate_supported")
            self.assertEqual(
                summary["claim_status"],
                "rank_matched_topk1_retention_advantage_candidate_not_promoted",
            )
            self.assertLess(summary["primary_result"]["topk1_minus_dense_anchor_ce_drift"], 0.0)
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertTrue(
                any(
                    row["variant"] == "frozen_base_anchor"
                    for row in summary["variant_metrics"]
                )
            )
            self.assertTrue(summary["primary_result"]["is_true_synthetic_cl_dataset"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_probe_labels_non_synthetic_dataset_as_confounded_slice_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            decision = root / "decision"
            _write_config(config)
            _write_decision(decision)

            with mock.patch(
                "relaleap.experiments.synthetic_task_free_continual_learning_probe"
                ".run_retention_churn_microtest",
                return_value=_microtest(dataset="tiny_shakespeare_word"),
            ):
                summary = run_synthetic_task_free_continual_learning_probe(
                    config_path=config,
                    decision_dir=decision,
                    out_dir=root / "out",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "confounded_slice_retention_signal_recorded")
            self.assertEqual(summary["claim_status"], "confounded_slice_retention_not_synthetic_cl")
            self.assertEqual(
                summary["selected_next_step"],
                "implement_mechanism_factorized_local_continual_learning_probe",
            )
            self.assertFalse(summary["primary_result"]["is_true_synthetic_cl_dataset"])
            self.assertTrue(
                any(
                    row["criterion"] == "dataset_is_true_synthetic_cl"
                    and not row["passed"]
                    and row["severity"] == "interpretation"
                    for row in summary["gate_criteria"]
                )
            )

    def test_probe_fails_closed_when_required_arm_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            decision = root / "decision"
            _write_config(config)
            _write_decision(decision)
            packet = _microtest()
            packet["audit"]["variants"] = [
                row
                for row in packet["audit"]["variants"]
                if row["variant"] != "random_fixed_topk2"
            ]

            with mock.patch(
                "relaleap.experiments.synthetic_task_free_continual_learning_probe"
                ".run_retention_churn_microtest",
                return_value=packet,
            ):
                summary = run_synthetic_task_free_continual_learning_probe(
                    config_path=config,
                    decision_dir=decision,
                    out_dir=root / "out",
                )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "synthetic_task_free_continual_learning_probe_failed_closed",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "required_arms_present" and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_probe_fails_closed_when_branch_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            decision = root / "decision"
            _write_config(config)
            _write_decision(decision, selected_next_step="different_branch")

            summary = run_synthetic_task_free_continual_learning_probe(
                config_path=config,
                decision_dir=decision,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(
                    row["criterion"] == "heldout_post_probe_selected_this_branch"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )


def _write_config(path: Path) -> None:
    path.write_text(
        """
run:
  experiment_id: test_synthetic_task_free_probe
  seed: 1
  max_steps: 2

data:
  dataset: tiny_shakespeare_word
  seq_len: 16

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 32
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_decision(
    out_dir: Path,
    *,
    selected_next_step: str = "implement_synthetic_task_free_continual_learning_dense_vs_sparse_probe",
) -> None:
    out_dir.mkdir(parents=True)
    payload = {
        "status": "pass",
        "decision": "heldout_context_post_probe_decision_recorded",
        "selected_next_step": selected_next_step,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _microtest(
    *,
    dataset: str = "synthetic_rule_stream",
    topk1_anchor_ce_drift: float = -0.08,
    dense_anchor_ce_drift: float = 0.02,
) -> dict[str, object]:
    return {
        "status": "ok",
        "audit": {
            "dataset": dataset,
            "empty_anchor_ce": 3.0,
            "empty_transfer_ce": 3.1,
            "variants": [
                _variant(
                    "rank_matched_contextual_topk1",
                    "sparse",
                    top_k=1,
                    anchor_ce_drift=topk1_anchor_ce_drift,
                    anchor_logit_mse=0.03,
                    transfer_improvement=0.32,
                    support_churn=0.01,
                ),
                _variant(
                    "promoted_contextual_topk2",
                    "sparse",
                    top_k=2,
                    anchor_ce_drift=-0.04,
                    anchor_logit_mse=0.06,
                    transfer_improvement=0.34,
                    support_churn=0.50,
                ),
                _variant(
                    "random_fixed_topk2",
                    "sparse_fixed",
                    top_k=2,
                    anchor_ce_drift=0.04,
                    anchor_logit_mse=0.05,
                    transfer_improvement=0.10,
                    support_churn=0.0,
                ),
                _variant(
                    "norm_matched_dense_active_rank",
                    "dense",
                    top_k=0,
                    anchor_ce_drift=dense_anchor_ce_drift,
                    anchor_logit_mse=0.04,
                    transfer_improvement=0.30,
                    support_churn="",
                ),
            ],
        },
    }


def _variant(
    variant: str,
    kind: str,
    *,
    top_k: int,
    anchor_ce_drift: float,
    anchor_logit_mse: float,
    transfer_improvement: float,
    support_churn: float | str,
) -> dict[str, object]:
    return {
        "variant": variant,
        "kind": kind,
        "support_router": "contextual_mlp" if kind != "dense" else "none",
        "top_k": top_k,
        "num_columns": 8 if kind != "dense" else "",
        "stored_parameters": 256,
        "active_parameters_proxy": 64,
        "anchor_ce_drift": anchor_ce_drift,
        "transfer_ce_improvement": transfer_improvement,
        "anchor_logit_mse_drift": anchor_logit_mse,
        "commutator_anchor_logit_mse": anchor_logit_mse / 2.0,
        "commutator_transfer_logit_mse": anchor_logit_mse / 3.0,
        "anchor_residual_stream_l2_drift": 0.2,
        "anchor_residual_norm_after_transfer": 1.0,
        "commutator_anchor_residual_stream_l2": 0.1,
        "anchor_support_churn_after_transfer": support_churn,
        "anchor_used_columns_after_transfer": 4,
        "anchor_unique_support_sets_after_transfer": 8,
    }


if __name__ == "__main__":
    unittest.main()
