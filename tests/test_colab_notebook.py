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
        self.assertIn(
            "Objective-gate validation PC-vs-supervised temporal clipped comparison status:",
            evidence_cell,
        )

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


if __name__ == "__main__":
    unittest.main()
