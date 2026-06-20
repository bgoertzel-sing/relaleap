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
    _confirm_run_modals,
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
                    "Temporal clipped seed2 support-stress comparison status: ok",
                    "Temporal clipped seed2 support-stress artifact check: pass",
                    "Temporal clipped seed3 support-stress comparison status: ok",
                    "Temporal clipped seed3 support-stress artifact check: pass",
                    "Temporal clipped seed4 support-stress comparison status: ok",
                    "Temporal clipped seed4 support-stress artifact check: pass",
                    "Temporal clipped validation support-stress comparison status: ok",
                    "Temporal clipped validation support-stress artifact check: pass",
                    "Temporal clipped extended support-stress comparison status: ok",
                    "Temporal clipped extended support-stress artifact check: pass",
                    "Temporal clipped larger support-stress comparison status: ok",
                    "Temporal clipped larger support-stress artifact check: pass",
                    "Temporal clipped token larger support-stress comparison status: ok",
                    "Temporal clipped token larger support-stress artifact check: pass",
                    "char_smoke_hep_support_stress_clipped",
                    "char_smoke_hep_support_stress_entropy_clipped",
                    "char_smoke_hep_support_stress_temporal_clipped",
                    "char_smoke_hep_support_stress_guided_clipped",
                    "char_smoke_hep_support_stress_clipped_seed2",
                    "char_smoke_hep_support_stress_entropy_clipped_seed2",
                    "char_smoke_hep_support_stress_temporal_clipped_seed2",
                    "char_smoke_hep_support_stress_guided_clipped_seed2",
                    "char_smoke_hep_support_stress_clipped_seed3",
                    "char_smoke_hep_support_stress_entropy_clipped_seed3",
                    "char_smoke_hep_support_stress_temporal_clipped_seed3",
                    "char_smoke_hep_support_stress_guided_clipped_seed3",
                    "char_smoke_hep_support_stress_clipped_seed4",
                    "char_smoke_hep_support_stress_entropy_clipped_seed4",
                    "char_smoke_hep_support_stress_temporal_clipped_seed4",
                    "char_smoke_hep_support_stress_guided_clipped_seed4",
                    "char_validation_hep_support_stress_clipped",
                    "char_validation_hep_support_stress_entropy_clipped",
                    "char_validation_hep_support_stress_temporal_clipped",
                    "char_validation_hep_support_stress_guided_clipped",
                    "char_extended_hep_support_stress_clipped",
                    "char_extended_hep_support_stress_entropy_clipped",
                    "char_extended_hep_support_stress_temporal_clipped",
                    "char_extended_hep_support_stress_guided_clipped",
                    "char_larger_hep_support_stress_clipped",
                    "char_larger_hep_support_stress_entropy_clipped",
                    "char_larger_hep_support_stress_temporal_clipped",
                    "char_larger_hep_support_stress_guided_clipped",
                    "token_larger_hep_support_stress_clipped",
                    "token_larger_hep_support_stress_entropy_clipped",
                    "token_larger_hep_support_stress_temporal_clipped",
                    "token_larger_hep_support_stress_guided_clipped",
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


class ConfirmRunModalsTest(unittest.IsolatedAsyncioTestCase):
    async def test_completion_wait_can_skip_run_all_prompts(self) -> None:
        page = _RecordingPage()

        clicked = await _confirm_run_modals(
            page,
            max_rounds=1,
            timeout_ms=1,
            include_run_all=False,
        )

        self.assertFalse(clicked)
        self.assertNotIn("Run anyway", page.role_labels)
        self.assertNotIn("Run all", page.role_labels)
        self.assertNotIn("Run all cells", page.role_labels)
        self.assertNotIn("Run anyway", page.text_labels)
        self.assertNotIn("Run all", page.text_labels)
        self.assertNotIn("Run all cells", page.text_labels)
        self.assertIn("Reconnect", page.role_labels)
        self.assertIn("Reconnect", page.text_labels)


class _RecordingPage:
    def __init__(self) -> None:
        self.role_labels: list[str] = []
        self.text_labels: list[str] = []

    def get_by_role(self, role: str, name: str, exact: bool = False) -> "_FailingTarget":
        self.role_labels.append(name)
        return _FailingTarget()

    def get_by_text(self, label: str, exact: bool = False) -> "_FailingTarget":
        self.text_labels.append(label)
        return _FailingTarget()

    async def wait_for_timeout(self, timeout_ms: int) -> None:
        return None


class _FailingTarget:
    @property
    def first(self) -> "_FailingTarget":
        return self

    async def click(self, timeout: int) -> None:
        raise RuntimeError("not visible")


def _zip_base64(files: dict[str, str]) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
