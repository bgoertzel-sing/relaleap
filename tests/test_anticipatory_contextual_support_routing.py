from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing import (
    REQUIRED_ARTIFACTS,
    run_anticipatory_contextual_support_routing,
)


class AnticipatoryContextualSupportRoutingSmokeTest(unittest.TestCase):
    def test_local_smoke_writes_fail_closed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "acsr_smoke.yaml"
            out_dir = root / "audit"
            _write_tiny_config(config)

            summary = run_anticipatory_contextual_support_routing(
                config_path=config,
                out_dir=out_dir,
                max_steps=2,
                predictor_steps=3,
            )

            self.assertIn(summary["status"], {"pass", "fail"})
            self.assertIn("future_perturbation_invariance", summary["gates"])
            self.assertIn("leaky_future_positive_control", summary["gates"])
            self.assertIn("sequence_heldout_available", summary["gates"])
            self.assertIn("margin_fragility_available", summary["gates"])
            self.assertEqual(summary["train_steps"], 2)
            self.assertEqual(summary["predictor_steps"], 3)
            self.assertEqual(summary["top_k"], 2)
            self.assertEqual(summary["best_predictor"], "mlp_causal")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            written = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["decision"], summary["decision"])
            router_metrics = (out_dir / "router_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("acsr_mlp_predicted_future", router_metrics)
            self.assertIn("acsr_gru_predicted_future", router_metrics)
            self.assertIn("shuffled_predicted_features", router_metrics)
            self.assertIn("parameter_matched_causal_mlp_control", router_metrics)
            predictor_metrics = (out_dir / "predictor_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("gru_causal", predictor_metrics)
            self.assertIn("parameter_matched_causal_mlp_control", predictor_metrics)
            same_student = (out_dir / "same_student_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("dual_student_cross_forcing", same_student)
            self.assertIn(
                "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features",
                same_student,
            )
            self.assertIn(
                "acsr_gru_predicted_future_support_vs_token_position_only_predicted_features",
                same_student,
            )
            support_agreement = (out_dir / "support_agreement.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "acsr_mlp_predicted_future_support_vs_parameter_matched_causal_mlp_control",
                support_agreement,
            )
            perturbation = (out_dir / "feature_perturbation.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("future_positions_do_not_change_prefix", perturbation)
            self.assertIn("leaky_future_positive", perturbation)
            sequence = (out_dir / "sequence_heldout_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("sequence_suffix_holdout", sequence)
            margin = (out_dir / "margin_fragility.csv").read_text(encoding="utf-8")
            self.assertIn("feature_noise_flip_rate", margin)
            parameter_counts = (out_dir / "parameter_counts.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("parameter_matched_causal_mlp_control", parameter_counts)
            retention = (out_dir / "retention_churn_metrics.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("second_context_transfer", retention)
            self.assertIn("anchor_support_churn_after_transfer", retention)


def _write_tiny_config(path: Path) -> None:
    path.write_text(
        """
run:
  experiment_id: acsr_unit_smoke
  seed: 3
  max_steps: 2
  learning_rate: 0.01

data:
  dataset: tiny_shakespeare_word
  seq_len: 8

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 16
  columns:
    num_columns: 6
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_stress: true
    support_stress_preset: false
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16
""".strip()
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
