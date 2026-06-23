from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from tools.colab_playwright_runner import (
    ARTIFACT_BUNDLE_BEGIN,
    ARTIFACT_BUNDLE_END,
    COLAB_NOTEBOOK_URL,
    COMPLETION_TEXT,
    FOCUSED_TARGET_COMPARISON_DIR,
    _colab_notebook_url,
    _confirm_run_modals,
    _extract_colab_artifact_bundle,
    _validate_evidence_text,
    _validate_focused_target_artifact_bundle,
    _validate_pinned_support_evidence,
    _wait_for_completion,
)


class ColabPlaywrightRunnerTest(unittest.TestCase):
    def test_colab_notebook_url_includes_revision_cache_buster(self) -> None:
        with patch("tools.colab_playwright_runner._current_git_revision") as revision:
            revision.return_value = "abc123def456"

            url = _colab_notebook_url()

        self.assertEqual(url, f"{COLAB_NOTEBOOK_URL}?relaleap_rev=abc123def456")

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
                    "Validation PC-vs-supervised temporal clipped comparison status: ok",
                    "Validation PC-vs-supervised temporal clipped artifact check: pass",
                    "Objective-gate validation PC-vs-supervised temporal clipped comparison status: ok",
                    "Objective-gate validation PC-vs-supervised temporal clipped artifact check: pass",
                    "Anchored objective-gate validation PC comparison status: ok",
                    "Anchored objective-gate validation PC artifact check: pass",
                    "Confidence-penalty objective-gate validation comparison status: ok",
                    "Confidence-penalty objective-gate validation artifact check: pass",
                    "Margin-penalty objective-gate validation comparison status: ok",
                    "Margin-penalty objective-gate validation artifact check: pass",
                    "Label-smoothing objective-gate validation comparison status: ok",
                    "Label-smoothing objective-gate validation artifact check: pass",
                    "Focal objective-gate validation comparison status: ok",
                    "Focal objective-gate validation artifact check: pass",
                    "Larger focal objective-gate comparison status: ok",
                    "Larger focal objective-gate artifact check: pass",
                    "Token larger focal objective-gate comparison status: ok",
                    "Token larger focal objective-gate artifact check: pass",
                    "Token larger focal seed2 objective-gate comparison status: ok",
                    "Token larger focal seed2 objective-gate artifact check: pass",
                    "Xlarge focal objective-gate comparison status: ok",
                    "Xlarge focal objective-gate artifact check: pass",
                    "Xxlarge focal objective-gate comparison status: ok",
                    "Xxlarge focal objective-gate artifact check: pass",
                    "Xxlarge focal seed2 objective-gate comparison status: ok",
                    "Xxlarge focal seed2 objective-gate artifact check: pass",
                    "results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2",
                    "Temporal clipped extended support-stress comparison status: ok",
                    "Temporal clipped extended support-stress artifact check: pass",
                    "Temporal clipped larger support-stress comparison status: ok",
                    "Temporal clipped larger support-stress artifact check: pass",
                    "Temporal clipped token larger support-stress comparison status: ok",
                    "Temporal clipped token larger support-stress artifact check: pass",
                    "Residual capacity/support validation comparison status: ok",
                    "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate",
                    "Support-width larger char/token comparison status: ok",
                    "Support-width larger char/token artifact check: pass",
                    "results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate",
                    "Support-width seed2 larger char/token comparison status: ok",
                    "Support-width seed2 larger char/token artifact check: pass",
                    "results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2",
                    "Post-support-width capacity larger char/token comparison status: ok",
                    "Post-support-width capacity larger char/token artifact check: pass",
                    "results/comparisons/colab_post_support_width_capacity_larger_token_objective_gate",
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
                    "char_validation_pc_hep_support_stress_temporal_clipped",
                    "char_validation_hep_temporal_clipped_objective_gate",
                    "char_validation_pc_hep_temporal_clipped_objective_gate",
                    "char_validation_pc_anchor_hep_temporal_clipped_objective_gate",
                    "char_validation_confidence_penalty_hep_temporal_clipped_objective_gate",
                    "char_validation_margin_penalty_hep_temporal_clipped_objective_gate",
                    "char_validation_label_smoothing_hep_temporal_clipped_objective_gate",
                    "char_validation_focal_hep_temporal_clipped_objective_gate",
                    "char_validation_support_wide_hep_temporal_clipped_objective_gate",
                    "char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate",
                    "char_larger_hep_temporal_clipped_objective_gate",
                    "char_larger_support_wide_hep_temporal_clipped_objective_gate",
                    "char_larger_capacity_hep_temporal_clipped_objective_gate",
                    "char_larger_hep_temporal_clipped_objective_gate_seed2",
                    "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
                    "char_larger_focal_hep_temporal_clipped_objective_gate",
                    "token_larger_hep_temporal_clipped_objective_gate",
                    "token_larger_support_wide_hep_temporal_clipped_objective_gate",
                    "token_larger_capacity_hep_temporal_clipped_objective_gate",
                    "token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
                    "token_larger_focal_hep_temporal_clipped_objective_gate",
                    "token_larger_hep_temporal_clipped_objective_gate_seed2",
                    "token_larger_focal_hep_temporal_clipped_objective_gate_seed2",
                    "char_xlarge_hep_temporal_clipped_objective_gate",
                    "char_xlarge_focal_hep_temporal_clipped_objective_gate",
                    "char_xxlarge_hep_temporal_clipped_objective_gate",
                    "char_xxlarge_focal_hep_temporal_clipped_objective_gate",
                    "char_xxlarge_hep_temporal_clipped_objective_gate_seed2",
                    "char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2",
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

    def test_validate_evidence_rejects_rendered_python_traceback(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Colab rendered Python error marker 'Traceback",
        ):
            _validate_evidence_text(
                "\n".join(
                    [
                        "summary_status: ok",
                        "verdict_status: pass",
                        "KeyError Traceback (most recent call last)",
                        "/tmp/ipykernel_12222/3779044106.py in <cell line: 0>()",
                        "169 assert token_temporal_check['status'] == 'pass'",
                        "170 token_temporal_runs = {run['experiment_id']: run for run in token_temporal_summary['runs']}",
                        "171 assert token_temporal_runs['token_larger_hep_support_stress_clipped']['dataset'] == 'tiny_shakespeare_word'",
                    ]
                )
            )

    def test_validate_pinned_support_accepts_artifact_bundle_summary(self) -> None:
        bundle = _zip_base64(
            {
                "results/runs/colab_char_smoke_pinned_hep/summary.json": (
                    '{"phase0": {"pinned_support": true}}\n'
                )
            }
        )

        _validate_pinned_support_evidence(
            "\n".join([ARTIFACT_BUNDLE_BEGIN, bundle, ARTIFACT_BUNDLE_END])
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

    def test_extract_colab_artifact_bundle_accepts_later_valid_bundle(self) -> None:
        bundle = _zip_base64(
            {
                "results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/summary.json": "{}\n",
                "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/summary.json": "{}\n",
                "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/metrics.csv": "step,loss\n",
                "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/notes.md": "# Notes\n",
                "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/artifact_check.json": "{}\n",
            }
        )
        evidence = "\n".join(
            [
                "truncated-base64-fragment",
                ARTIFACT_BUNDLE_END,
                ARTIFACT_BUNDLE_BEGIN,
                bundle,
                ARTIFACT_BUNDLE_END,
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = _extract_colab_artifact_bundle(evidence, Path(tmpdir))

            self.assertEqual(len(extracted), 5)
            self.assertTrue(
                (
                    Path(tmpdir)
                    / "results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/summary.json"
                ).is_file()
            )

    def test_extract_colab_artifact_bundle_rejects_unsafe_paths(self) -> None:
        bundle = _zip_base64({"../outside.txt": "bad\n"})
        evidence = "\n".join([ARTIFACT_BUNDLE_BEGIN, bundle, ARTIFACT_BUNDLE_END])

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "Unsafe Colab artifact path"):
                _extract_colab_artifact_bundle(evidence, Path(tmpdir))

    def test_focused_target_bundle_rejects_stale_support_schema(self) -> None:
        summary = _focused_summary(include_support_schema=False)
        bundle = _zip_base64(
            {
                f"{FOCUSED_TARGET_COMPARISON_DIR}/summary.json": (
                    json_dumps(summary)
                ),
                f"{FOCUSED_TARGET_COMPARISON_DIR}/artifact_check.json": (
                    '{"status": "pass"}\n'
                ),
                f"{FOCUSED_TARGET_COMPARISON_DIR}/metrics.csv": "step,loss\n",
                f"{FOCUSED_TARGET_COMPARISON_DIR}/notes.md": "# Notes\n",
            }
        )
        evidence = "\n".join([ARTIFACT_BUNDLE_BEGIN, bundle, ARTIFACT_BUNDLE_END])

        with self.assertRaisesRegex(RuntimeError, "focused support-width artifact schema"):
            _validate_focused_target_artifact_bundle(evidence)

    def test_focused_target_bundle_accepts_support_width_schema(self) -> None:
        summary = _focused_summary(include_support_schema=True)
        bundle = _zip_base64(
            {
                f"{FOCUSED_TARGET_COMPARISON_DIR}/summary.json": (
                    json_dumps(summary)
                ),
                f"{FOCUSED_TARGET_COMPARISON_DIR}/artifact_check.json": (
                    '{"status": "pass"}\n'
                ),
                f"{FOCUSED_TARGET_COMPARISON_DIR}/metrics.csv": "step,loss\n",
                f"{FOCUSED_TARGET_COMPARISON_DIR}/notes.md": "# Notes\n",
            }
        )
        evidence = "\n".join([ARTIFACT_BUNDLE_BEGIN, bundle, ARTIFACT_BUNDLE_END])

        _validate_focused_target_artifact_bundle(evidence)


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

    async def test_completion_wait_can_skip_generic_runtime_controls(self) -> None:
        page = _RecordingPage()

        clicked = await _confirm_run_modals(
            page,
            max_rounds=1,
            timeout_ms=1,
            include_run_all=False,
            include_generic_runtime_controls=False,
        )

        self.assertFalse(clicked)
        self.assertIn("Reconnect", page.role_labels)
        self.assertNotIn("Connect", page.role_labels)
        self.assertNotIn("OK", page.role_labels)
        self.assertNotIn("Ok", page.role_labels)
        self.assertIn("Reconnect", page.text_labels)
        self.assertNotIn("Connect", page.text_labels)
        self.assertNotIn("OK", page.text_labels)
        self.assertNotIn("Ok", page.text_labels)

    async def test_completion_wait_fails_after_repeated_runtime_prompts(self) -> None:
        async def never_completed(*args, **kwargs):
            raise TimeoutError("not done")

        async def no_rendered_output(*args, **kwargs):
            return ""

        async def no_obstructive_modal(*args, **kwargs):
            return False

        async def runtime_prompt_confirmed(*args, **kwargs):
            return True

        async def write_evidence(page, evidence_out):
            evidence_out.write_text("partial evidence\n", encoding="utf-8")

        snapshots = []

        async def write_debug_snapshot(page, label):
            snapshots.append(label)

        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_out = Path(tmpdir) / "evidence.txt"
            with (
                patch(
                    "tools.colab_playwright_runner._wait_for_rendered_output_text",
                    never_completed,
                ),
                patch(
                    "tools.colab_playwright_runner._rendered_output_text",
                    no_rendered_output,
                ),
                patch(
                    "tools.colab_playwright_runner._dismiss_obstructive_modals",
                    no_obstructive_modal,
                ),
                patch(
                    "tools.colab_playwright_runner._confirm_run_modals",
                    runtime_prompt_confirmed,
                ),
                patch(
                    "tools.colab_playwright_runner._write_evidence",
                    write_evidence,
                ),
                patch(
                    "tools.colab_playwright_runner._write_debug_snapshot",
                    write_debug_snapshot,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "manual Chrome/Colab runtime resolution is required",
                ):
                    await _wait_for_completion(
                        _RecordingPage(),
                        timeout_minutes=1.0,
                        evidence_out=evidence_out,
                    )

            self.assertEqual(
                evidence_out.read_text(encoding="utf-8"),
                "partial evidence\n",
            )
            self.assertEqual(snapshots, ["runtime_prompt_blocked"])


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


def json_dumps(value) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _focused_summary(*, include_support_schema: bool) -> dict:
    specs = [
        ("char_validation_hep_temporal_clipped_objective_gate", 12, 1),
        ("char_validation_capacity_hep_temporal_clipped_objective_gate", 24, 1),
        ("char_validation_support_wide_hep_temporal_clipped_objective_gate", 12, 2),
        (
            "char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate",
            24,
            2,
        ),
    ]
    runs = []
    for experiment_id, num_columns, top_k in specs:
        run = {
            "experiment_id": experiment_id,
            "status": "ok",
            "config_path": f"configs/{experiment_id}.yaml",
        }
        if include_support_schema:
            run.update(
                {
                    "num_columns": num_columns,
                    "top_k": top_k,
                    "support_audit": {
                        "num_columns": num_columns,
                        "top_k": top_k,
                        "used_columns": 1 if top_k == 1 else 8,
                        "dead_columns": num_columns - (1 if top_k == 1 else 8),
                        "unique_support_sets": 1 if top_k == 1 else 12,
                        "total_support_slots": 256 * top_k,
                        "support_positions": 256,
                    },
                }
            )
        return_run = run
        runs.append(return_run)
    return {
        "status": "ok",
        "verdict": {"status": "pass"},
        "runs": runs,
    }


if __name__ == "__main__":
    unittest.main()
