from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.colab_playwright_runner import (
    ARTIFACT_BUNDLE_BEGIN,
    ARTIFACT_BUNDLE_END,
    COMPLETION_TEXT,
    _extract_colab_artifact_bundle,
    _validate_evidence_text,
)


class ColabPlaywrightRunnerTest(unittest.TestCase):
    def test_validate_evidence_accepts_rendered_success_markers(self) -> None:
        _validate_evidence_text(
            "\n".join(
                [
                    "cuda_available: True",
                    '"status": "pass"',
                    "Accepted HEP alpha: {'alpha': 0.25}",
                    '"pinned_support": true',
                    "Pinned HEP status: ok",
                    "Support-stress comparison status: ok",
                    "Support-stress artifact check: pass",
                    "Clipped support-stress comparison status: ok",
                    "Clipped support-stress artifact check: pass",
                    "Guided clipped support-stress comparison status: ok",
                    "Guided clipped support-stress artifact check: pass",
                    "Temporal clipped support-stress comparison status: ok",
                    "Temporal clipped support-stress artifact check: pass",
                    "char_smoke_hep_support_stress_clipped",
                    "char_smoke_hep_support_stress_entropy_clipped",
                    "char_smoke_hep_support_stress_temporal_clipped",
                    "char_smoke_hep_support_stress_guided_clipped",
                    "char_smoke_pinned_hep_support_stress",
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
                        "print('Pinned HEP status:', pinned_summary['status'])",
                        f"print('{COMPLETION_TEXT}')",
                    ]
                )
            )

    def test_extract_colab_artifact_bundle_writes_safe_paths(self) -> None:
        bundle = _zip_base64(
            {
                "results/comparisons/colab_phase0/summary.json": "{}\n",
                "results/comparisons/colab_phase0/notes.md": "# Notes\n",
            }
        )
        evidence = "\n".join(
            [
                ARTIFACT_BUNDLE_BEGIN,
                bundle[:20],
                bundle[20:],
                ARTIFACT_BUNDLE_END,
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = _extract_colab_artifact_bundle(evidence, Path(tmpdir))

            self.assertEqual(len(extracted), 2)
            self.assertEqual(
                (Path(tmpdir) / "results/comparisons/colab_phase0/summary.json")
                .read_text(encoding="utf-8"),
                "{}\n",
            )

    def test_extract_colab_artifact_bundle_rejects_unsafe_paths(self) -> None:
        bundle = _zip_base64({"../outside.txt": "bad\n"})
        evidence = "\n".join([ARTIFACT_BUNDLE_BEGIN, bundle, ARTIFACT_BUNDLE_END])

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "Unsafe Colab artifact path"):
                _extract_colab_artifact_bundle(evidence, Path(tmpdir))


def _zip_base64(files: dict[str, str]) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
