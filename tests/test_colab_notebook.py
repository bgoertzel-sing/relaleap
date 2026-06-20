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


if __name__ == "__main__":
    unittest.main()
