from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
    INSUFFICIENT_EVIDENCE,
    run_promoted_topk2_finite_update_order_control_audit,
)
from relaleap.experiments.promoted_topk2_functional_churn_control_audit import (
    FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
)


class PromotedTopk2FiniteUpdateOrderControlAuditTest(unittest.TestCase):
    def test_reports_ce_bounded_material_residual_order_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            functional = root / "functional"
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            microtest = root / "microtest"
            fingerprint = root / "fingerprint"
            _write_functional(functional)
            _write_probe(seed1, "seed1", topk2_logit=0.20, topk1_logit=0.01)
            _write_probe(seed2, "seed2", topk2_logit=0.28, topk1_logit=0.01)
            _write_microtest(microtest)
            _write_fingerprint(fingerprint)

            summary = run_promoted_topk2_finite_update_order_control_audit(
                functional_churn_dir=functional,
                probe_dirs=(seed1, seed2),
                microtest_dirs=(microtest,),
                fingerprint_dir=fingerprint,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
            )
            self.assertTrue(
                summary["signals"]["topk2_absolute_commutator_ce_bounded"]
            )
            self.assertTrue(
                summary["signals"]["topk2_absolute_commutator_residual_material"]
            )
            self.assertGreater(
                summary["metrics"][
                    "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio"
                ],
                20.0,
            )
            self.assertLess(
                summary["metrics"][
                    "topk2_to_random_fixed_topk2_mean_commutator_anchor_logit_mse_ratio"
                ],
                1.0,
            )
            self.assertEqual(len(summary["microtest_packet_rows"]), 1)
            self.assertTrue(summary["signals"]["order_averaged_rows_available"])
            self.assertAlmostEqual(
                summary["metrics"][
                    "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio"
                ],
                0.25,
            )
            self.assertAlmostEqual(
                summary["metrics"][
                    "topk2_mean_order_averaged_anchor_ce_delta_vs_forward"
                ],
                0.003,
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "variant_commutator.csv").is_file())
            self.assertTrue((root / "out" / "token_strata.csv").is_file())
            self.assertTrue((root / "out" / "token_correlations.csv").is_file())
            self.assertIn(
                "Existing artifacts do not expose finite-update KL deltas.",
                summary["source_limitations"],
            )

    def test_fails_closed_when_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_promoted_topk2_finite_update_order_control_audit(
                functional_churn_dir=root / "missing_functional",
                probe_dirs=(root / "missing_seed1", root / "missing_seed2"),
                microtest_dirs=(),
                fingerprint_dir=root / "missing_fingerprint",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("artifact", fields)
            self.assertIn("decision", fields)
            self.assertIn("token_strata_rows", fields)


def _write_functional(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_probe(path: Path, packet: str, *, topk2_logit: float, topk1_logit: float) -> None:
    path.mkdir(parents=True)
    variants = [
        _variant(
            "promoted_contextual_topk2",
            top_k=2,
            support_churn=0.9,
            ce_delta=0.02,
            logit_mse=topk2_logit,
            residual_l2=5.0,
        ),
        _variant(
            "rank_matched_contextual_topk1",
            top_k=1,
            support_churn=0.0,
            ce_delta=0.02,
            logit_mse=topk1_logit,
            residual_l2=1.3,
        ),
        _variant(
            "norm_matched_dense_active_rank",
            top_k=0,
            support_churn="",
            ce_delta=0.03,
            logit_mse=0.08,
            residual_l2=3.3,
            kind="dense",
        ),
    ]
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "active_topk1_retention_churn_probe_established",
                "config_path": f"configs/{packet}.yaml",
                "audit": {"variants": variants},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_microtest(path: Path) -> None:
    path.mkdir(parents=True)
    variants = [
        _variant(
            "promoted_contextual_topk2",
            top_k=2,
            support_churn=0.88,
            ce_delta=0.03,
            logit_mse=0.22,
            residual_l2=4.3,
        ),
        _variant(
            "rank_matched_contextual_topk1",
            top_k=1,
            support_churn=0.0,
            ce_delta=0.02,
            logit_mse=0.01,
            residual_l2=1.4,
        ),
        _variant(
            "random_fixed_topk2",
            top_k=2,
            support_churn=0.0,
            ce_delta=0.02,
            logit_mse=0.36,
            residual_l2=6.8,
            kind="sparse_fixed",
        ),
        _variant(
            "norm_matched_dense_active_rank",
            top_k=0,
            support_churn="",
            ce_delta=0.02,
            logit_mse=0.06,
            residual_l2=2.9,
            kind="dense",
        ),
    ]
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "config_path": "configs/token_larger.yaml",
                "audit": {"variants": variants},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _variant(
    name: str,
    *,
    top_k: int,
    support_churn: float | str,
    ce_delta: float,
    logit_mse: float,
    residual_l2: float,
    kind: str = "sparse",
) -> dict[str, object]:
    return {
        "variant": name,
        "kind": kind,
        "top_k": top_k,
        "num_columns": 24,
        "active_parameters_proxy": 768,
        "anchor_ce_drift": -0.9,
        "anchor_logit_mse_drift": 0.15,
        "anchor_residual_stream_l2_drift": 4.5,
        "anchor_support_churn_after_transfer": support_churn,
        "commutator_anchor_ce_abs_delta": ce_delta,
        "commutator_transfer_ce_abs_delta": ce_delta,
        "commutator_anchor_logit_mse": logit_mse,
        "commutator_transfer_logit_mse": logit_mse,
        "commutator_anchor_residual_stream_l2": residual_l2,
        "commutator_transfer_residual_stream_l2": residual_l2,
        "commutator_anchor_support_churn": support_churn,
        "commutator_transfer_support_churn": support_churn,
        "order_averaged_anchor_ce_delta_vs_forward": 0.003,
        "order_averaged_transfer_ce_delta_vs_forward": 0.002,
        "order_averaged_anchor_ce_delta_vs_best_order": 0.004,
        "order_averaged_transfer_ce_delta_vs_best_order": 0.003,
        "order_averaged_anchor_logit_mse_to_forward": logit_mse * 0.25,
        "order_averaged_transfer_logit_mse_to_forward": logit_mse * 0.25,
        "order_averaged_anchor_residual_stream_l2_to_forward": residual_l2 * 0.5,
        "order_averaged_transfer_residual_stream_l2_to_forward": residual_l2 * 0.5,
    }


def _write_fingerprint(path: Path) -> None:
    path.mkdir(parents=True)
    fieldnames = [
        "variant",
        "intervention",
        "position_bin",
        "token_class",
        "router_support_count",
        "fixed_support_loss_delta",
        "fixed_support_logit_mse",
        "fixed_support_residual_stream_l2_delta",
        "residual_norm",
        "residual_norm_bin",
        "residual_gain",
        "residual_gain_bin",
        "pair_synergy",
    ]
    rows = [
        {
            "variant": "baseline",
            "intervention": "fixed_dominant_router_support",
            "position_bin": "even",
            "token_class": "rare_target",
            "router_support_count": 18,
            "fixed_support_loss_delta": 1.0,
            "fixed_support_logit_mse": 0.2,
            "fixed_support_residual_stream_l2_delta": 5.0,
            "residual_norm": 4.0,
            "residual_norm_bin": "low",
            "residual_gain": 0.2,
            "residual_gain_bin": "high",
            "pair_synergy": 0.1,
        },
        {
            "variant": "baseline",
            "intervention": "fixed_dominant_router_support",
            "position_bin": "odd",
            "token_class": "common_target",
            "router_support_count": 16,
            "fixed_support_loss_delta": 1.2,
            "fixed_support_logit_mse": 0.3,
            "fixed_support_residual_stream_l2_delta": 6.0,
            "residual_norm": 5.0,
            "residual_norm_bin": "mid",
            "residual_gain": 0.1,
            "residual_gain_bin": "mid",
            "pair_synergy": 0.2,
        },
    ]
    with (path / "per_token_pair_interventions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
