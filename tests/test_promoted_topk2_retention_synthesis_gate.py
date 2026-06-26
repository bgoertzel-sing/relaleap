from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)
from relaleap.experiments.promoted_topk2_functional_churn_control_audit import (
    FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
)
from relaleap.experiments.promoted_topk2_retention_synthesis_gate import (
    CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
    INSUFFICIENT_EVIDENCE,
    RETENTION_SEPARABILITY_RISK_MITIGATION_RECOMMENDED,
    run_promoted_topk2_retention_synthesis_gate,
)
from relaleap.experiments.promoted_topk2_support_selection_quality_audit import (
    PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
)


class PromotedTopk2RetentionSynthesisGateTest(unittest.TestCase):
    def test_recommends_mitigation_after_replicated_retention_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            finite = root / "finite"
            functional = root / "functional"
            support = root / "support"
            deconfounded = root / "deconfounded"
            _write_microtest(seed1, topk2_transfer=0.90, topk1_transfer=0.92)
            _write_microtest(seed2, topk2_transfer=0.95, topk1_transfer=0.99)
            _write_json(
                finite / "summary.json",
                {
                    "status": "pass",
                    "decision": FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
                    "metrics": {
                        "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": 24.0
                    },
                },
            )
            _write_json(
                functional / "summary.json",
                {
                    "status": "pass",
                    "decision": FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
                },
            )
            _write_json(
                support / "summary.json",
                {
                    "status": "pass",
                    "decision": PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
                    "metrics": {
                        "oracle_support_regret": 0.0025,
                        "oracle_support_regret_positive_fraction": 0.05,
                    },
                },
            )
            _write_json(
                deconfounded / "summary.json",
                {
                    "status": "pass",
                    "decision": "topk2_comparative_causal_cooperation_not_supported",
                    "evidence": {
                        "topk2_ce_deficit_vs_topk1": 0.04,
                        "topk2_incremental_pair_gain_positive_strata_fraction": 0.65,
                        "topk2_fixed_support_cleaner_strata_fraction": 0.65,
                        "topk2_functional_churn_cleaner_strata_fraction": 0.61,
                    },
                },
            )

            summary = run_promoted_topk2_retention_synthesis_gate(
                microtest_dirs=(seed1, seed2),
                finite_update_dir=finite,
                functional_churn_dir=functional,
                support_selection_dir=support,
                deconfounded_dir=deconfounded,
                context_gate_dir=None,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                RETENTION_SEPARABILITY_RISK_MITIGATION_RECOMMENDED,
            )
            self.assertTrue(summary["signals"]["topk2_transfer_beats_random_and_dense"])
            self.assertTrue(summary["signals"]["topk2_high_support_churn_replicated"])
            self.assertTrue(summary["signals"]["topk2_commutator_risk_replicated"])
            self.assertTrue(summary["signals"]["topk2_causal_cooperation_not_supported"])
            self.assertGreater(
                summary["metrics"]["min_topk2_to_topk1_commutator_anchor_logit_mse_ratio"],
                20.0,
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "retention_seed_metrics.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_fails_closed_when_microtest_packet_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_promoted_topk2_retention_synthesis_gate(
                microtest_dirs=(root / "missing_seed1", root / "missing_seed2"),
                finite_update_dir=root / "missing_finite",
                functional_churn_dir=root / "missing_functional",
                support_selection_dir=root / "missing_support",
                deconfounded_dir=root / "missing_deconfounded",
                context_gate_dir=None,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("artifact", fields)
            self.assertIn("status", fields)
            self.assertIn("required_variants", fields)

    def test_keeps_topk2_default_when_topk1_gate_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            finite = root / "finite"
            functional = root / "functional"
            support = root / "support"
            deconfounded = root / "deconfounded"
            context_gate = root / "context_gate"
            _write_microtest(seed1, topk2_transfer=0.91, topk1_transfer=0.92)
            _write_microtest(seed2, topk2_transfer=0.95, topk1_transfer=0.96)
            _write_json(
                finite / "summary.json",
                {
                    "status": "pass",
                    "decision": FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
                    "metrics": {
                        "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": 24.0
                    },
                },
            )
            _write_json(
                functional / "summary.json",
                {
                    "status": "pass",
                    "decision": FUNCTIONAL_CHURN_BOUNDED_WITH_COMMUTATOR_RISK,
                },
            )
            _write_json(
                support / "summary.json",
                {
                    "status": "pass",
                    "decision": PROMOTED_TOPK2_SUPPORT_SELECTION_QUALITY_ESTABLISHED,
                    "metrics": {
                        "oracle_support_regret": 0.0025,
                        "oracle_support_regret_positive_fraction": 0.05,
                    },
                },
            )
            _write_json(
                deconfounded / "summary.json",
                {
                    "status": "pass",
                    "decision": "topk2_comparative_causal_cooperation_not_supported",
                    "evidence": {"topk2_ce_deficit_vs_topk1": 0.04},
                },
            )
            _write_json(
                context_gate / "summary.json",
                {
                    "status": "pass",
                    "decision": "deployable_context_gate_suppression_calibration_failed",
                },
            )

            summary = run_promoted_topk2_retention_synthesis_gate(
                microtest_dirs=(seed1, seed2),
                finite_update_dir=finite,
                functional_churn_dir=functional,
                support_selection_dir=support,
                deconfounded_dir=deconfounded,
                context_gate_dir=context_gate,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
            )
            self.assertTrue(summary["signals"]["topk1_transfer_competitive"])
            self.assertTrue(summary["signals"]["topk1_context_gate_failed"])
            self.assertIn("finite-update order-symmetrization", summary["next_step"])


def _write_microtest(path: Path, *, topk2_transfer: float, topk1_transfer: float) -> None:
    _write_json(
        path / "summary.json",
        {
            "status": "ok",
            "config_path": "configs/token_larger.yaml",
            "device": "cuda",
            "cuda_device_name": "test gpu",
            "audit": {
                "variants": [
                    _variant(
                        "promoted_contextual_topk2",
                        transfer=topk2_transfer,
                        churn=0.85,
                        commutator=0.24,
                        ce_delta=0.02,
                    ),
                    _variant(
                        "rank_matched_contextual_topk1",
                        transfer=topk1_transfer,
                        churn=0.005,
                        commutator=0.01,
                        ce_delta=0.02,
                    ),
                    _variant(
                        "random_fixed_topk2",
                        transfer=0.62,
                        churn=0.0,
                        commutator=0.35,
                        ce_delta=0.03,
                    ),
                    _variant(
                        "norm_matched_dense_active_rank",
                        transfer=0.45,
                        churn="",
                        commutator=0.07,
                        ce_delta=0.04,
                    ),
                ]
            },
        },
    )


def _variant(
    name: str,
    *,
    transfer: float,
    churn: float | str,
    commutator: float,
    ce_delta: float,
) -> dict[str, object]:
    return {
        "variant": name,
        "transfer_ce_improvement": transfer,
        "anchor_ce_drift": -0.9,
        "anchor_support_churn_after_transfer": churn,
        "commutator_anchor_logit_mse": commutator,
        "commutator_anchor_ce_abs_delta": ce_delta,
        "commutator_anchor_residual_stream_l2": 4.0,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
