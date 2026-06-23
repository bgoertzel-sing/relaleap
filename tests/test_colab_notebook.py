from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/relaleap_colab_smoke.ipynb")


class ColabNotebookTest(unittest.TestCase):
    def test_checkout_cell_forces_current_origin_main(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        checkout_cells = [
            source
            for source in sources
            if "git clone https://github.com/bgoertzel-sing/relaleap.git" in source
        ]

        self.assertEqual(len(checkout_cells), 1)
        checkout_cell = checkout_cells[0]
        self.assertIn("git fetch origin main --prune", checkout_cell)
        self.assertIn("git reset --hard origin/main", checkout_cell)
        self.assertIn(
            "configs/char_larger_hep_support_stress_clipped.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_hep_support_stress_clipped.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_pc_hep_support_stress_temporal_clipped.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_label_smoothing_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_focal_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_focal_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_focal_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xlarge_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xlarge_focal_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xxlarge_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_support_wide_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_hep_temporal_clipped_objective_gate_seed3.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_hep_temporal_clipped_objective_gate_seed3.yaml",
            checkout_cell,
        )
        self.assertIn(
            "configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml",
            checkout_cell,
        )

    def test_run_cell_executes_token_larger_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/token_larger_hep_support_stress_clipped.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_support_stress_entropy_clipped.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_support_stress_temporal_clipped.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_support_stress_guided_clipped.yaml",
            run_cell,
        )
        self.assertIn("token_larger_hep_support_stress_temporal_clipped", evidence_cell)
        self.assertIn("tiny_shakespeare_word", evidence_cell)

    def test_run_cell_executes_xxlarge_focal_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_char_xxlarge_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_xxlarge_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_xxlarge_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "char_xxlarge_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "Xxlarge focal objective-gate comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "Xxlarge focal objective-gate artifact check:",
            evidence_cell,
        )
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_token_larger_seed2_focal_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_token_larger_focal_temporal_clipped_objective_gate_seed2"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "token_larger_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_focal_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn(
            "Token larger focal seed2 objective-gate comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "Token larger focal seed2 objective-gate artifact check:",
            evidence_cell,
        )
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_xxlarge_seed2_focal_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "char_xxlarge_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn(
            "char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn(
            "Xxlarge focal seed2 objective-gate comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "Xxlarge focal seed2 objective-gate artifact check:",
            evidence_cell,
        )
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_validation_pc_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_pc_vs_supervised_temporal_clipped_hep" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_support_stress_temporal_clipped.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_pc_hep_support_stress_temporal_clipped.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_hep_support_stress_temporal_clipped",
            evidence_cell,
        )
        self.assertIn(
            "char_validation_pc_hep_support_stress_temporal_clipped",
            evidence_cell,
        )
        self.assertIn(
            "Validation PC-vs-supervised temporal clipped comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_objective_gate_validation_pc_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_pc_vs_supervised_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "char_validation_pc_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_capacity_support_diagnostic_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_residual_capacity_support_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "Residual capacity/support validation comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "char_validation_support_wide_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("support_stress_preset'] is False", evidence_cell)
        self.assertIn(
            "Objective-gate validation PC-vs-supervised temporal clipped comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "Path('results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate')",
            evidence_cell,
        )

    def test_run_cell_executes_larger_token_support_width_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_support_width_larger_char_token_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "Support-width larger char/token comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "char_larger_support_wide_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_support_wide_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("tiny_shakespeare_word", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_seed2_larger_token_support_width_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_larger_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml",
            run_cell,
        )
        self.assertIn(
            "Support-width seed2 larger char/token comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
            evidence_cell,
        )
        self.assertIn("tiny_shakespeare_word", evidence_cell)
        self.assertIn("support_width_seed2_check['status'] == 'pass'", evidence_cell)

    def test_run_cell_executes_seed3_larger_token_support_width_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_larger_hep_temporal_clipped_objective_gate_seed3.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate_seed3.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml",
            run_cell,
        )
        self.assertIn(
            "Support-width seed3 larger char/token comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3",
            evidence_cell,
        )
        self.assertIn("tiny_shakespeare_word", evidence_cell)
        self.assertIn("support_width_seed3_check['status'] == 'pass'", evidence_cell)

    def test_run_cell_executes_post_support_width_capacity_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_post_support_width_capacity_larger_token_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_capacity_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "Post-support-width capacity larger char/token comparison status:",
            evidence_cell,
        )
        self.assertIn(
            "char_larger_capacity_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_capacity_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("num_columns'] == 48", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)

    def test_run_cell_executes_anchored_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_pc_anchor_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_pc_anchor_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "pc_logit_mse_ce_anchor",
            evidence_cell,
        )
        self.assertIn(
            "Anchored objective-gate validation PC comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_confidence_penalty_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_confidence_penalty_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_confidence_penalty_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "supervised_ce_confidence_penalty",
            evidence_cell,
        )
        self.assertIn(
            "Confidence-penalty objective-gate validation comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_margin_penalty_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_margin_penalty_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_margin_penalty_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "supervised_ce_margin_penalty",
            evidence_cell,
        )
        self.assertIn(
            "Margin-penalty objective-gate validation comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_label_smoothing_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_label_smoothing_temporal_clipped_objective_gate"
            in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_label_smoothing_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_label_smoothing_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "supervised_ce_label_smoothing",
            evidence_cell,
        )
        self.assertIn(
            "Label-smoothing objective-gate validation comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_focal_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_validation_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_validation_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_validation_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_validation_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("supervised_ce_focal", evidence_cell)
        self.assertIn(
            "Focal objective-gate validation comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_extended_focal_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_extended_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_extended_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_extended_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_extended_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "char_extended_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("supervised_ce_focal", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)
        self.assertIn(
            "Extended focal objective-gate comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_larger_focal_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_larger_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_larger_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_larger_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "char_larger_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("supervised_ce_focal", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)
        self.assertIn(
            "Larger focal objective-gate comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_token_larger_focal_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_token_larger_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/token_larger_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/token_larger_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "token_larger_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "token_larger_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("tiny_shakespeare_word", evidence_cell)
        self.assertIn("supervised_ce_focal", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)
        self.assertIn(
            "Token larger focal objective-gate comparison status:",
            evidence_cell,
        )

    def test_run_cell_executes_xlarge_focal_objective_gate_colab_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        run_cells = [
            source
            for source in sources
            if "colab_char_xlarge_focal_temporal_clipped_objective_gate" in source
        ]

        self.assertEqual(len(run_cells), 2)
        run_cell, evidence_cell = run_cells
        self.assertIn(
            "--config configs/char_xlarge_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "--config configs/char_xlarge_focal_hep_temporal_clipped_objective_gate.yaml",
            run_cell,
        )
        self.assertIn(
            "char_xlarge_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn(
            "char_xlarge_focal_hep_temporal_clipped_objective_gate",
            evidence_cell,
        )
        self.assertIn("supervised_ce_focal", evidence_cell)
        self.assertIn("support_stress_preset'] is False", evidence_cell)
        self.assertIn(
            "Xlarge focal objective-gate comparison status:",
            evidence_cell,
        )


if __name__ == "__main__":
    unittest.main()
