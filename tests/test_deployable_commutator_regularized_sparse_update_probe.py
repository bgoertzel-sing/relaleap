from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.deployable_commutator_regularized_sparse_update_probe import (
    CANDIDATE,
    REQUIRED_ARTIFACTS,
    run_deployable_commutator_regularized_sparse_update_probe,
)


class DeployableCommutatorRegularizedSparseUpdateProbeTests(unittest.TestCase):
    def test_records_candidate_but_blocks_gpu_when_dense_control_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = _write_sources(root)
            with patch(
                "relaleap.experiments.deployable_commutator_regularized_sparse_update_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(out_dir)

                summary = run_deployable_commutator_regularized_sparse_update_probe(
                    config_path=root / "config.yaml",
                    pregate_path=sources["pregate"],
                    flat_value_path=sources["flat"],
                    order_probe_path=sources["order"],
                    strategy_review_path=sources["review"],
                    out_dir=root / "out",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "deployable_commutator_regularized_sparse_update_probe_recorded_gpu_blocked",
            )
            self.assertEqual(summary["claim_status"], "deployable_sparse_update_not_established")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            candidate = {row["variant"]: row for row in summary["arm_metrics"]}[CANDIDATE]
            self.assertEqual(candidate["commutator_value_penalty_weight"], 0.1)
            failed = {row["criterion"] for row in summary["claim_failures"]}
            self.assertIn("candidate_beats_dense_and_random_controls", failed)
            self.assertTrue(all((root / "out" / name).is_file() for name in REQUIRED_ARTIFACTS))

    def test_missing_pregate_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = _write_sources(root)
            sources["pregate"].unlink()
            with patch(
                "relaleap.experiments.deployable_commutator_regularized_sparse_update_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(out_dir)

                summary = run_deployable_commutator_regularized_sparse_update_probe(
                    config_path=root / "config.yaml",
                    pregate_path=sources["pregate"],
                    flat_value_path=sources["flat"],
                    order_probe_path=sources["order"],
                    strategy_review_path=sources["review"],
                    out_dir=root / "out",
                )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "deployable_commutator_regularized_sparse_update_probe_failed_closed",
            )
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "pregate": root / "pregate.json",
        "flat": root / "flat.json",
        "order": root / "order.json",
        "review": root / "latest-review.md",
    }
    _write_json(
        paths["pregate"],
        {
            "status": "pass",
            "selected_next_action": "implement_local_deployable_commutator_regularized_sparse_update_probe",
        },
    )
    _write_json(
        paths["flat"],
        {
            "status": "pass",
            "decision": "same_router_flat_value_commutator_mitigation_probe_gpu_blocked",
            "claim_status": "flat_value_commutator_mitigation_not_established",
            "variant_rows": [{"variant": "flat_value_commutator_penalty_probe", "variant_passes": False}],
        },
    )
    _write_json(
        paths["order"],
        {
            "status": "pass",
            "decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
        },
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: none",
                "notify_ben: false",
                "recommended_next_action: run local deployable sparse update probe",
                "verdict: PAUSE-RECOVER",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _mock_microtest(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _token_row("low", "0,1", "2,3"),
        _token_row("medium", "0,1", "1,2"),
        _token_row("high", "0,1", "0,1"),
    ]
    with (out_dir / "per_token_commutator.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "variant,split,symmetric_kl,forward_support,reverse_support\n"
            + "\n".join(rows)
            + "\n"
        )
    (out_dir / "summary.json").write_text("{}", encoding="utf-8")
    variants = [
        _variant("promoted_contextual_topk2", commutator=0.30, transfer=1.0, drift=0.01, norm=2.0),
        _variant("deployable_commutator_regularized_sparse_update", commutator=0.20, transfer=0.9, drift=0.02, norm=1.9, candidate=True),
        _variant("norm_matched_dense_active_rank", commutator=0.15, transfer=0.7, drift=0.01, norm=2.0),
        _variant("norm_matched_dense_stored_rank", commutator=0.14, transfer=0.7, drift=0.01, norm=2.1),
        _variant("random_fixed_topk2", commutator=0.25, transfer=0.5, drift=0.0, norm=2.0),
    ]
    return {"status": "ok", "experiment_id": "mock_microtest", "audit": {"variants": variants}}


def _variant(
    name: str,
    *,
    commutator: float,
    transfer: float,
    drift: float,
    norm: float,
    candidate: bool = False,
) -> dict[str, object]:
    row = {
        "variant": name,
        "kind": "sparse" if "dense" not in name else "dense",
        "stored_parameters": 100,
        "active_parameters_proxy": 32,
        "commutator_anchor_logit_mse": commutator,
        "commutator_transfer_logit_mse": commutator,
        "anchor_ce_drift": drift,
        "anchor_logit_mse_drift": drift,
        "transfer_ce_improvement": transfer,
        "anchor_residual_norm_before_transfer": norm,
        "anchor_residual_norm_after_transfer": norm,
        "parameter_delta_after_anchor": 1.0,
        "parameter_delta_during_transfer": 1.0,
    }
    if candidate:
        row.update(
            {
                "freeze_router_during_transfer": True,
                "gradient_clip_norm": 0.25,
                "value_gradient_clip_norm": 0.10,
                "value_gradient_low_rank": 2,
                "commutator_value_penalty_weight": 0.10,
            }
        )
    return row


def _token_row(label: str, forward: str, reverse: str) -> str:
    _ = label
    return f"deployable_commutator_regularized_sparse_update,anchor,0.05,\"{forward}\",\"{reverse}\""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
