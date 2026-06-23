from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.run import _read_config
from relaleap.experiments.run import run
from relaleap.smoke import (
    ResidualColumns,
    forward_with_hep_alpha,
    _hep_support_instability,
    run_phase0_smoke,
)


CONFIG = {
    "run": {"experiment_id": "test_smoke", "seed": 1, "max_steps": 2},
    "data": {"dataset": "tiny_shakespeare_char", "seq_len": 32},
    "model": {
        "base": {"layers": 2, "hidden_dim": 32},
        "columns": {
            "num_columns": 8,
            "atoms_per_column": 4,
            "top_k": 1,
            "insertion_sites": 1,
        },
    },
    "inference": {"pc_steps": 1, "hep_alpha": 0.0},
    "outputs": {
        "require_summary_json": True,
        "require_metrics_csv": True,
        "require_notes_md": True,
    },
}


class Phase0SmokeTest(unittest.TestCase):
    def test_phase0_invariants_pass(self) -> None:
        try:
            result = run_phase0_smoke(CONFIG)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.invariants["zero_init_identity"])
        self.assertTrue(result.invariants["frozen_base_unchanged"])
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertTrue(result.invariants["residual_parameters_updated"])
        self.assertAlmostEqual(result.base_loss, result.zero_init_loss)
        self.assertAlmostEqual(result.initial_loss, result.zero_init_loss)
        self.assertEqual(result.training_steps, CONFIG["run"]["max_steps"])
        self.assertEqual(result.residual_objective, "supervised_ce")
        self.assertEqual(len(result.to_metric_rows()), CONFIG["run"]["max_steps"] + 1)
        self.assertEqual(result.to_metric_rows()[0]["phase"], "initial")
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce"
                for row in result.to_metric_rows()
            )
        )
        self.assertTrue(
            all(row["phase"] == "residual_update" for row in result.to_metric_rows()[1:])
        )
        self.assertEqual(result.to_metric_rows()[-1]["residual_loss"], result.post_step_loss)
        self.assertEqual(result.support_audit["support_positions"], 4 * 32)
        self.assertEqual(result.support_audit["top_k"], 1)
        self.assertEqual(result.support_audit["num_columns"], 8)
        self.assertEqual(result.support_audit["total_support_slots"], 4 * 32)
        self.assertEqual(
            sum(result.support_audit["column_counts"]),
            result.support_audit["total_support_slots"],
        )
        self.assertEqual(
            sum(result.support_audit["support_set_counts"].values()),
            result.support_audit["support_positions"],
        )
        self.assertIn("support_audit", result.to_summary())

    def test_residual_columns_break_zero_score_ties_deterministically(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        residual = ResidualColumns(
            hidden_dim=4,
            num_columns=4,
            atoms_per_column=2,
            top_k=2,
        )
        hidden = torch.zeros(2, 3, 4)

        output, support = residual(hidden, return_support=True)

        self.assertTrue(torch.equal(output, hidden))
        self.assertTrue(torch.equal(support[..., 0], torch.zeros_like(support[..., 0])))
        self.assertTrue(torch.equal(support[..., 1], torch.ones_like(support[..., 1])))

    def test_contextual_residual_router_preserves_zero_init_identity(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        residual = ResidualColumns(
            hidden_dim=4,
            num_columns=5,
            atoms_per_column=2,
            top_k=2,
            support_router="contextual_mlp",
            contextual_router_hidden_dim=8,
        )
        hidden = torch.randn(2, 3, 4)

        output, support = residual(hidden, return_support=True)

        self.assertTrue(torch.equal(output, hidden))
        self.assertEqual(support.shape, (2, 3, 2))
        self.assertEqual(residual.support_router, "contextual_mlp")

    def test_phase0_accepts_contextual_support_router(self) -> None:
        contextual_config = copy.deepcopy(CONFIG)
        contextual_config["model"]["columns"]["top_k"] = 2
        contextual_config["model"]["columns"]["support_router"] = "contextual_mlp"
        contextual_config["model"]["columns"]["contextual_router_hidden_dim"] = 16

        try:
            result = run_phase0_smoke(contextual_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.support_router, "contextual_mlp")
        self.assertEqual(result.contextual_router_hidden_dim, 16)
        self.assertEqual(result.to_summary()["support_router"], "contextual_mlp")
        self.assertEqual(result.support_audit["top_k"], 2)

    def test_pc_logit_mse_objective_toggle(self) -> None:
        pc_config = copy.deepcopy(CONFIG)
        pc_config["training"] = {"residual_objective": "pc_logit_mse"}

        try:
            result = run_phase0_smoke(pc_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "pc_logit_mse")
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_metric_rows()[0]["phase"], "initial")
        self.assertTrue(
            all(
                row["phase"] == "pc_residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "pc_logit_mse"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_pc_logit_mse_ce_anchor_objective_toggle(self) -> None:
        anchored_config = copy.deepcopy(CONFIG)
        anchored_config["training"] = {
            "residual_objective": "pc_logit_mse_ce_anchor",
            "ce_anchor_weight": 0.1,
        }

        try:
            result = run_phase0_smoke(anchored_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "pc_logit_mse_ce_anchor")
        self.assertEqual(result.ce_anchor_weight, 0.1)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["ce_anchor_weight"], 0.1)
        self.assertTrue(
            all(
                row["phase"] == "pc_residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "pc_logit_mse_ce_anchor"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_supervised_ce_confidence_penalty_objective_toggle(self) -> None:
        confidence_config = copy.deepcopy(CONFIG)
        confidence_config["training"] = {
            "residual_objective": "supervised_ce_confidence_penalty",
            "confidence_penalty_weight": 0.02,
        }

        try:
            result = run_phase0_smoke(confidence_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "supervised_ce_confidence_penalty")
        self.assertEqual(result.confidence_penalty_weight, 0.02)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["confidence_penalty_weight"], 0.02)
        self.assertTrue(
            all(
                row["phase"] == "residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce_confidence_penalty"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_supervised_ce_margin_penalty_objective_toggle(self) -> None:
        margin_config = copy.deepcopy(CONFIG)
        margin_config["training"] = {
            "residual_objective": "supervised_ce_margin_penalty",
            "margin_penalty_weight": 0.02,
            "target_logit_margin": 0.3,
        }

        try:
            result = run_phase0_smoke(margin_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "supervised_ce_margin_penalty")
        self.assertEqual(result.margin_penalty_weight, 0.02)
        self.assertEqual(result.target_logit_margin, 0.3)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["margin_penalty_weight"], 0.02)
        self.assertEqual(result.to_summary()["target_logit_margin"], 0.3)
        self.assertTrue(
            all(
                row["phase"] == "residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce_margin_penalty"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_supervised_ce_label_smoothing_objective_toggle(self) -> None:
        label_smoothing_config = copy.deepcopy(CONFIG)
        label_smoothing_config["training"] = {
            "residual_objective": "supervised_ce_label_smoothing",
            "label_smoothing_weight": 0.05,
        }

        try:
            result = run_phase0_smoke(label_smoothing_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "supervised_ce_label_smoothing")
        self.assertEqual(result.label_smoothing_weight, 0.05)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["label_smoothing_weight"], 0.05)
        self.assertTrue(
            all(
                row["phase"] == "residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce_label_smoothing"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_supervised_ce_focal_objective_toggle(self) -> None:
        focal_config = copy.deepcopy(CONFIG)
        focal_config["training"] = {
            "residual_objective": "supervised_ce_focal",
            "focal_gamma": 2.0,
        }

        try:
            result = run_phase0_smoke(focal_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "supervised_ce_focal")
        self.assertEqual(result.focal_gamma, 2.0)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["focal_gamma"], 2.0)
        self.assertTrue(
            all(
                row["phase"] == "residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce_focal"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_supervised_ce_temporal_consistency_objective_toggle(self) -> None:
        temporal_config = copy.deepcopy(CONFIG)
        temporal_config["training"] = {
            "residual_objective": "supervised_ce_temporal_consistency",
            "temporal_consistency_weight": 0.02,
        }

        try:
            result = run_phase0_smoke(temporal_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(
            result.residual_objective,
            "supervised_ce_temporal_consistency",
        )
        self.assertEqual(result.temporal_consistency_weight, 0.02)
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_summary()["temporal_consistency_weight"], 0.02)
        self.assertTrue(
            all(
                row["phase"] == "residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce_temporal_consistency"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_hep_alpha_sweep_records_nonzero_alpha_rows(self) -> None:
        hep_config = copy.deepcopy(CONFIG)
        hep_config["run"]["max_steps"] = 1
        hep_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,0.5,1.0",
        }

        try:
            result = run_phase0_smoke(hep_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertEqual(
            [entry["alpha"] for entry in result.hep_alpha_sweep],
            [0.0, 0.5, 1.0],
        )
        self.assertLessEqual(
            result.hep_alpha_sweep[0]["max_logit_delta_from_ordinary"],
            1e-6,
        )
        sweep_rows = [
            row for row in result.to_metric_rows() if row["phase"] == "hep_sweep"
        ]
        self.assertEqual([row["hep_alpha"] for row in sweep_rows], [0.0, 0.5, 1.0])
        self.assertTrue(
            all(row["hep_loss"] != "" for row in sweep_rows)
        )

    def test_residual_columns_can_pin_selected_support(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        residual = ResidualColumns(
            hidden_dim=2,
            num_columns=3,
            atoms_per_column=1,
            top_k=1,
        )
        with torch.no_grad():
            residual.column_scores.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                )
            )
            residual.atom_values.copy_(
                torch.tensor(
                    [
                        [[-2.0, 5.0]],
                        [[0.0, 10.0]],
                        [[1.0, 1.0]],
                    ]
                )
            )

        hidden = torch.tensor([[[1.0, 0.0]]])
        settled, support = residual(hidden, return_support=True)
        repicked = residual(settled)
        pinned = residual(settled, support_indices=support)

        self.assertEqual(support.tolist(), [[[0]]])
        self.assertNotEqual(repicked.tolist(), pinned.tolist())
        self.assertEqual(pinned.tolist(), [[[-3.0, 10.0]]])

    def test_support_instability_detects_repicked_settling(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        class IdentityBase:
            def encode(self, inputs):
                return inputs

            def decode(self, hidden):
                return hidden

        residual = ResidualColumns(
            hidden_dim=2,
            num_columns=3,
            atoms_per_column=1,
            top_k=1,
        )
        with torch.no_grad():
            residual.column_scores.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                )
            )
            residual.atom_values.copy_(
                torch.tensor(
                    [
                        [[-2.0, 5.0]],
                        [[0.0, 10.0]],
                        [[1.0, 1.0]],
                    ]
                )
            )

        diagnostic = _hep_support_instability(
            IdentityBase(),
            residual,
            torch.tensor([[[1.0, 0.0]]]),
            pc_steps=2,
            hep_alpha=1.0,
        )

        self.assertTrue(diagnostic["support_changed"])
        self.assertEqual(diagnostic["support_changed_positions"], 1)
        self.assertEqual(diagnostic["support_transition_count"], 1)
        self.assertEqual(diagnostic["support_change_fraction"], 1.0)
        self.assertGreater(diagnostic["pinned_vs_repicked_logit_delta"], 0.0)

    def test_hep_update_clip_bounds_settling_divergence(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        class IdentityBase:
            def encode(self, inputs):
                return inputs

            def decode(self, hidden):
                return hidden

        residual = ResidualColumns(
            hidden_dim=2,
            num_columns=3,
            atoms_per_column=1,
            top_k=1,
        )
        with torch.no_grad():
            residual.column_scores.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                )
            )
            residual.atom_values.copy_(
                torch.tensor(
                    [
                        [[-2.0, 5.0]],
                        [[0.0, 10.0]],
                        [[1.0, 1.0]],
                    ]
                )
            )

        inputs = torch.tensor([[[1.0, 0.0]]])
        unclipped = _hep_support_instability(
            IdentityBase(),
            residual,
            inputs,
            pc_steps=2,
            hep_alpha=1.0,
        )
        clipped = _hep_support_instability(
            IdentityBase(),
            residual,
            inputs,
            pc_steps=2,
            hep_alpha=1.0,
            hep_update_clip_norm=1.0,
        )
        unclipped_logits = forward_with_hep_alpha(
            IdentityBase(),
            residual,
            inputs,
            pc_steps=2,
            hep_alpha=1.0,
        )
        clipped_logits = forward_with_hep_alpha(
            IdentityBase(),
            residual,
            inputs,
            pc_steps=2,
            hep_alpha=1.0,
            hep_update_clip_norm=1.0,
        )

        self.assertEqual(clipped["hep_update_clip_norm"], 1.0)
        self.assertLess(
            clipped["pinned_vs_repicked_logit_delta"],
            unclipped["pinned_vs_repicked_logit_delta"],
        )
        self.assertLess(
            float((clipped_logits - unclipped_logits).abs().max().item()),
            10.0,
        )

    def test_pinned_support_config_is_reported(self) -> None:
        pinned_config = copy.deepcopy(CONFIG)
        pinned_config["run"]["max_steps"] = 1
        pinned_config["model"]["columns"]["pinned_support"] = True
        pinned_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,0.5",
        }

        try:
            result = run_phase0_smoke(pinned_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.pinned_support)
        self.assertTrue(result.to_summary()["pinned_support"])
        self.assertIn("support_change_fraction", result.support_instability)
        self.assertIn("support_instability", result.to_summary())
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])

    def test_support_stress_config_records_nonzero_repicking(self) -> None:
        stress_config = copy.deepcopy(CONFIG)
        stress_config["run"]["max_steps"] = 1
        stress_config["run"]["experiment_id"] = "test_support_stress"
        stress_config["model"]["columns"]["support_stress"] = True
        stress_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
        }

        try:
            result = run_phase0_smoke(stress_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.support_stress)
        self.assertTrue(result.invariants["zero_init_identity"])
        self.assertTrue(result.invariants["frozen_base_unchanged"])
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertGreater(result.support_instability["support_change_fraction"], 0.0)
        self.assertGreater(
            result.support_instability["pinned_vs_repicked_logit_delta"],
            0.0,
        )
        self.assertIn("support_stress", [row["phase"] for row in result.to_metric_rows()])
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertGreater(alpha1["support_change_fraction"], 0.0)
        self.assertGreater(alpha1["pinned_vs_repicked_logit_delta"], 0.0)

    def test_support_stress_without_preset_preserves_learned_residual_values(
        self,
    ) -> None:
        stress_config = copy.deepcopy(CONFIG)
        stress_config["run"]["max_steps"] = 2
        stress_config["run"]["experiment_id"] = "test_support_stress_no_preset"
        stress_config["model"]["columns"]["support_stress"] = True
        stress_config["model"]["columns"]["support_stress_preset"] = False
        stress_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
            "hep_update_clip_norm": 0.01,
            "hep_settling_objective": "temporal_consistency_gradient",
        }

        try:
            result = run_phase0_smoke(stress_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        phases = [row["phase"] for row in result.to_metric_rows()]
        residual_update_loss = [
            row["residual_loss"]
            for row in result.to_metric_rows()
            if row["phase"] == "residual_update"
        ][-1]
        alpha0 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 0.0
        ][0]

        self.assertTrue(result.support_stress)
        self.assertFalse(result.support_stress_preset)
        self.assertFalse(result.to_summary()["support_stress_preset"])
        self.assertNotIn("support_stress", phases)
        self.assertAlmostEqual(result.post_step_loss, residual_update_loss)
        self.assertAlmostEqual(alpha0["loss"], residual_update_loss)
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])

    def test_default_support_stress_config_uses_temporal_clipped_hep(self) -> None:
        config = _read_config(Path("configs/char_smoke_hep_support_stress.yaml"))

        inference = config["inference"]
        self.assertEqual(inference["hep_update_clip_norm"], 0.01)
        self.assertEqual(
            inference["hep_settling_objective"],
            "temporal_consistency_gradient",
        )

    def test_seed2_focal_objective_gate_configs_match_gate_requirements(self) -> None:
        expected = {
            "char_xxlarge_hep_temporal_clipped_objective_gate_seed2": {
                "path": (
                    "configs/"
                    "char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml"
                ),
                "dataset": "tiny_shakespeare_char",
                "seq_len": 192,
                "hidden_dim": 160,
                "num_columns": 40,
                "max_steps": 70,
                "residual_objective": "supervised_ce",
            },
            "char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2": {
                "path": (
                    "configs/"
                    "char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml"
                ),
                "dataset": "tiny_shakespeare_char",
                "seq_len": 192,
                "hidden_dim": 160,
                "num_columns": 40,
                "max_steps": 70,
                "residual_objective": "supervised_ce_focal",
            },
            "token_larger_hep_temporal_clipped_objective_gate_seed2": {
                "path": (
                    "configs/"
                    "token_larger_hep_temporal_clipped_objective_gate_seed2.yaml"
                ),
                "dataset": "tiny_shakespeare_word",
                "seq_len": 64,
                "hidden_dim": 96,
                "num_columns": 24,
                "max_steps": 50,
                "residual_objective": "supervised_ce",
            },
            "token_larger_focal_hep_temporal_clipped_objective_gate_seed2": {
                "path": (
                    "configs/"
                    "token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml"
                ),
                "dataset": "tiny_shakespeare_word",
                "seq_len": 64,
                "hidden_dim": 96,
                "num_columns": 24,
                "max_steps": 50,
                "residual_objective": "supervised_ce_focal",
            },
        }

        for experiment_id, fields in expected.items():
            with self.subTest(experiment_id=experiment_id):
                config = _read_config(Path(fields["path"]))
                self.assertEqual(config["run"]["experiment_id"], experiment_id)
                self.assertEqual(config["run"]["seed"], 2)
                self.assertEqual(config["run"]["max_steps"], fields["max_steps"])
                self.assertEqual(config["data"]["dataset"], fields["dataset"])
                self.assertEqual(config["data"]["seq_len"], fields["seq_len"])
                self.assertEqual(
                    config["training"]["residual_objective"],
                    fields["residual_objective"],
                )
                if fields["residual_objective"] == "supervised_ce_focal":
                    self.assertEqual(config["training"]["focal_gamma"], 2.0)
                self.assertEqual(
                    config["model"]["base"]["hidden_dim"],
                    fields["hidden_dim"],
                )
                self.assertEqual(
                    config["model"]["columns"]["num_columns"],
                    fields["num_columns"],
                )
                self.assertFalse(config["model"]["columns"]["support_stress_preset"])
                self.assertEqual(config["inference"]["pc_steps"], 4)
                self.assertEqual(config["inference"]["hep_update_clip_norm"], 0.01)
                self.assertEqual(
                    config["inference"]["hep_settling_objective"],
                    "temporal_consistency_gradient",
                )

    def test_hep_update_clip_config_is_reported(self) -> None:
        clipped_config = copy.deepcopy(CONFIG)
        clipped_config["run"]["max_steps"] = 1
        clipped_config["run"]["experiment_id"] = "test_hep_update_clip"
        clipped_config["model"]["columns"]["support_stress"] = True
        clipped_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
            "hep_update_clip_norm": 0.01,
        }

        try:
            result = run_phase0_smoke(clipped_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.hep_update_clip_norm, 0.01)
        self.assertEqual(result.to_summary()["hep_update_clip_norm"], 0.01)
        self.assertEqual(result.support_instability["hep_update_clip_norm"], 0.01)
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertLessEqual(alpha1["max_logit_delta_from_ordinary"], 0.1)

    def test_supervised_gradient_hep_improves_clipped_support_stress_loss(self) -> None:
        guided_config = copy.deepcopy(CONFIG)
        guided_config["run"]["max_steps"] = 1
        guided_config["run"]["experiment_id"] = "test_guided_clipped_hep"
        guided_config["model"]["columns"]["support_stress"] = True
        guided_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
            "hep_update_clip_norm": 0.01,
            "hep_settling_objective": "supervised_ce_gradient",
        }

        try:
            result = run_phase0_smoke(guided_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.hep_settling_objective, "supervised_ce_gradient")
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        alpha0 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 0.0
        ][0]
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertLess(alpha1["loss"], alpha0["loss"])
        self.assertLessEqual(alpha1["max_logit_delta_from_ordinary"], 0.1)

    def test_prediction_entropy_hep_is_label_free_and_reported(self) -> None:
        entropy_config = copy.deepcopy(CONFIG)
        entropy_config["run"]["max_steps"] = 1
        entropy_config["run"]["experiment_id"] = "test_entropy_clipped_hep"
        entropy_config["model"]["columns"]["support_stress"] = True
        entropy_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
            "hep_update_clip_norm": 0.01,
            "hep_settling_objective": "prediction_entropy_gradient",
        }

        try:
            result = run_phase0_smoke(entropy_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.hep_settling_objective, "prediction_entropy_gradient")
        self.assertEqual(
            result.to_summary()["hep_settling_objective"],
            "prediction_entropy_gradient",
        )
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertLessEqual(alpha1["max_logit_delta_from_ordinary"], 0.1)

    def test_temporal_consistency_hep_is_label_free_and_reported(self) -> None:
        temporal_config = copy.deepcopy(CONFIG)
        temporal_config["run"]["max_steps"] = 1
        temporal_config["run"]["experiment_id"] = "test_temporal_clipped_hep"
        temporal_config["model"]["columns"]["support_stress"] = True
        temporal_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
            "hep_update_clip_norm": 0.01,
            "hep_settling_objective": "temporal_consistency_gradient",
        }

        try:
            result = run_phase0_smoke(temporal_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.hep_settling_objective, "temporal_consistency_gradient")
        self.assertEqual(
            result.to_summary()["hep_settling_objective"],
            "temporal_consistency_gradient",
        )
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertLessEqual(alpha1["max_logit_delta_from_ordinary"], 0.1)

    def test_word_tokenized_dataset_runs_same_artifact_contract(self) -> None:
        token_config = copy.deepcopy(CONFIG)
        token_config["run"]["max_steps"] = 1
        token_config["run"]["experiment_id"] = "test_word_tokenized"
        token_config["data"] = {
            "dataset": "tiny_shakespeare_word",
            "seq_len": 16,
        }

        try:
            result = run_phase0_smoke(token_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.dataset, "tiny_shakespeare_word")
        self.assertEqual(result.seq_len, 16)
        self.assertGreater(result.vocab_size, 10)
        self.assertTrue(result.invariants["zero_init_identity"])
        self.assertTrue(result.invariants["frozen_base_unchanged"])
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertEqual(
            result.to_summary()["dataset"],
            "tiny_shakespeare_word",
        )

    def test_runner_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  experiment_id: test_smoke",
                        "  seed: 1",
                        "  max_steps: 2",
                        "data:",
                        "  dataset: tiny_shakespeare_char",
                        "  seq_len: 32",
                        "training:",
                        "  residual_objective: supervised_ce",
                        "model:",
                        "  base:",
                        "    layers: 2",
                        "    hidden_dim: 32",
                        "  columns:",
                        "    num_columns: 8",
                        "    atoms_per_column: 4",
                        "    top_k: 1",
                        "    insertion_sites: 1",
                        "inference:",
                        "  pc_steps: 1",
                        "  hep_alpha: 0.0",
                        "outputs:",
                        "  require_summary_json: true",
                        "  require_metrics_csv: true",
                        "  require_notes_md: true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run(config_path, tmp_path / "out")
            if summary["error"] and "torch" in summary["error"]:
                self.skipTest(summary["error"])

            self.assertEqual(summary["status"], "ok")
            self.assertTrue((tmp_path / "out" / "summary.json").is_file())
            self.assertTrue((tmp_path / "out" / "metrics.csv").is_file())
            self.assertTrue((tmp_path / "out" / "notes.md").is_file())
            saved = json.loads((tmp_path / "out" / "summary.json").read_text())
            self.assertTrue(saved["artifact_invariants"]["summary_json"])
            self.assertTrue(saved["phase0"]["invariants"]["zero_init_identity"])
            self.assertIn("base_loss", saved["phase0"])
            self.assertIn("post_step_loss", saved["phase0"])
            self.assertEqual(saved["phase0"]["residual_objective"], "supervised_ce")
            self.assertEqual(saved["phase0"]["training_steps"], 2)
            self.assertEqual(saved["phase0"]["num_columns"], 8)
            self.assertEqual(saved["phase0"]["atoms_per_column"], 4)
            self.assertEqual(saved["phase0"]["top_k"], 1)
            self.assertEqual(saved["phase0"]["support_audit"]["support_positions"], 128)
            self.assertEqual(saved["phase0"]["support_audit"]["top_k"], 1)
            self.assertEqual(saved["phase0"]["support_audit"]["num_columns"], 8)

            with (tmp_path / "out" / "metrics.csv").open(newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["phase"] for row in metric_rows],
                ["initial", "residual_update", "residual_update"],
            )
            self.assertEqual([int(row["step"]) for row in metric_rows], [0, 1, 2])
            self.assertNotIn("smoke_loss", metric_rows[0])
            self.assertIn("base_loss", metric_rows[0])
            self.assertIn("residual_loss", metric_rows[0])
            self.assertIn("residual_objective", metric_rows[0])
            self.assertIn("hep_alpha", metric_rows[0])
            self.assertIn("hep_loss", metric_rows[0])
            self.assertIn("hep_support_change_fraction", metric_rows[0])
            self.assertIn("hep_pinned_vs_repicked_logit_delta", metric_rows[0])
            self.assertEqual(metric_rows[0]["residual_objective"], "supervised_ce")
            self.assertEqual(
                float(metric_rows[-1]["residual_loss"]),
                saved["final_smoke_loss"],
            )


if __name__ == "__main__":
    unittest.main()
