from __future__ import annotations

import unittest

from tools.colab_playwright_runner import COMPLETION_TEXT, _validate_evidence_text


class ColabPlaywrightRunnerTest(unittest.TestCase):
    def test_validate_evidence_accepts_rendered_success_markers(self) -> None:
        _validate_evidence_text(
            "\n".join(
                [
                    "cuda_available: True",
                    '"status": "pass"',
                    "Accepted HEP alpha: {'alpha': 0.25}",
                    COMPLETION_TEXT,
                ]
            )
        )

    def test_validate_evidence_rejects_source_only_completion_text(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cuda_available: True"):
            _validate_evidence_text(
                "\n".join(
                    [
                        "assert baseline_comparison['status'] == 'pass'",
                        "print('Accepted HEP alpha:', accepted)",
                        f"print('{COMPLETION_TEXT}')",
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
