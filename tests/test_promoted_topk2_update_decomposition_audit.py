from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.promoted_topk2_update_decomposition_audit import (
    DECOMPOSITION_INSUFFICIENT,
    MIXED_UPDATE_SENSITIVITY,
    ROUTER_UPDATE_DOMINATED,
    VALUE_UPDATE_DOMINATED,
    run_promoted_topk2_update_decomposition_audit,
)


class PromotedTopk2UpdateDecompositionAuditTest(unittest.TestCase):
    def test_runs_decomposition_microtest_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_update_decomposition
  seed: 1
  max_steps: 2

data:
  dataset: tiny_shakespeare_char
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

            summary = run_promoted_topk2_update_decomposition_audit(
                config_path=config_path,
                out_dir=root / "audit",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {
                    VALUE_UPDATE_DOMINATED,
                    ROUTER_UPDATE_DOMINATED,
                    MIXED_UPDATE_SENSITIVITY,
                },
            )
            self.assertEqual(len(summary["decomposition_rows"]), 2)
            rows_by_group = {
                row["transfer_update_group"]: row
                for row in summary["decomposition_rows"]
            }
            self.assertIn("router_only", rows_by_group)
            self.assertIn("value_only", rows_by_group)
            self.assertTrue((root / "audit" / "summary.json").is_file())
            self.assertTrue((root / "audit" / "decomposition_rows.csv").is_file())
            self.assertTrue((root / "audit" / "variant_metrics.csv").is_file())
            saved = json.loads((root / "audit" / "summary.json").read_text())
            self.assertEqual(saved["decision"], summary["decision"])

    def test_identifies_router_dominated_mocked_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "relaleap.experiments.promoted_topk2_update_decomposition_audit.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(
                    out_dir,
                    router_commutator=0.16,
                    value_commutator=0.04,
                )

                summary = run_promoted_topk2_update_decomposition_audit(
                    config_path=root / "config.yaml",
                    out_dir=root / "audit",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ROUTER_UPDATE_DOMINATED)

    def test_fails_closed_when_rows_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "relaleap.experiments.promoted_topk2_update_decomposition_audit.run_retention_churn_microtest"
            ) as mocked:
                mocked.return_value = {"status": "ok", "audit": {"variants": []}}

                summary = run_promoted_topk2_update_decomposition_audit(
                    config_path=root / "config.yaml",
                    out_dir=root / "audit",
                )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], DECOMPOSITION_INSUFFICIENT)


def _mock_microtest(
    out_dir: Path,
    *,
    router_commutator: float,
    value_commutator: float,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant_metrics.csv").write_text("", encoding="utf-8")
    (out_dir / "phase_metrics.csv").write_text("", encoding="utf-8")
    variants = [
        _variant(
            "promoted_contextual_topk2",
            commutator=0.20,
            transfer=1.0,
            group="full",
        ),
        _variant(
            "router_only_transfer_topk2",
            commutator=router_commutator,
            transfer=0.7,
            group="router_only",
        ),
        _variant(
            "value_only_transfer_topk2",
            commutator=value_commutator,
            transfer=0.9,
            group="value_only",
        ),
    ]
    return {"status": "ok", "audit": {"variants": variants}}


def _variant(
    name: str,
    *,
    commutator: float,
    transfer: float,
    group: str,
) -> dict[str, object]:
    return {
        "variant": name,
        "commutator_anchor_logit_mse": commutator,
        "commutator_transfer_logit_mse": commutator,
        "transfer_ce_improvement": transfer,
        "anchor_ce_drift": 0.0,
        "anchor_support_churn_after_transfer": 0.5,
        "commutator_anchor_support_churn": 0.5,
        "anchor_used_columns_after_transfer": 4,
        "transfer_update_group": group,
    }


if __name__ == "__main__":
    unittest.main()
