#!/usr/bin/env python3
"""Best-effort Colab browser automation for RelaLeap.

This is intentionally a temporary bridge. It opens the GitHub-backed Colab
notebook and tries to connect and run all cells through the web UI. Expect this
to need occasional selector fixes when Colab changes.

Usage:
  python tools/colab_playwright_runner.py --manual-login
  python tools/colab_playwright_runner.py --run-all
  python tools/colab_playwright_runner.py --browser-channel chrome --manual-login
  python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --wait-completion
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from datetime import datetime, timezone
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode
import zipfile


COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/"
    "bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb"
)
COMPLETION_TEXT = "RelaLeap Colab Phase 0 comparison completed."
ARTIFACT_BUNDLE_BEGIN = "RELALEAP_ARTIFACT_BUNDLE_ZIP_BASE64_BEGIN"
ARTIFACT_BUNDLE_END = "RELALEAP_ARTIFACT_BUNDLE_ZIP_BASE64_END"
FOCUSED_TARGET_COMPARISON_DIR = (
    "results/comparisons/"
    "colab_contextual_support_router_promotion_gate_larger_char_token"
)
FOCUSED_TARGET_RUN_SCHEMA = {
    "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2": {
        "num_columns": 24,
        "top_k": 2,
        "support_router": "linear",
    },
    "char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2": {
        "num_columns": 24,
        "top_k": 2,
        "support_router": "contextual_mlp",
        "contextual_router_hidden_dim": 128,
    },
    "token_larger_support_wide_hep_temporal_clipped_objective_gate": {
        "num_columns": 24,
        "top_k": 2,
        "support_router": "linear",
    },
    "token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate": {
        "num_columns": 24,
        "top_k": 2,
        "support_router": "contextual_mlp",
        "contextual_router_hidden_dim": 128,
    },
}
FOCUSED_TARGET_MARKERS = (
    "cuda_available: True",
    '"status": "pass"',
    FOCUSED_TARGET_COMPARISON_DIR,
    "char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2",
    "char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2",
    "token_larger_support_wide_hep_temporal_clipped_objective_gate",
    "token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate",
    COMPLETION_TEXT,
)
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "KeyError",
    "ModuleNotFoundError",
    "FileNotFoundError",
    "RuntimeError",
)
OUTPUT_SELECTORS = (
    "colab-static-output-renderer",
    ".output",
    ".stream.output_text",
)


def _current_git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    revision = completed.stdout.strip()
    return revision or None


def _colab_notebook_url() -> str:
    revision = _current_git_revision()
    if revision is None:
        return COLAB_NOTEBOOK_URL
    return f"{COLAB_NOTEBOOK_URL}?{urlencode({'relaleap_rev': revision})}"


async def _click_first(page, labels: list[str], timeout_ms: int = 5_000) -> bool:
    for label in labels:
        try:
            target = page.get_by_text(label, exact=False).first
            await target.click(timeout=timeout_ms)
            print(f"clicked: {label}")
            return True
        except Exception:
            pass
    return False


async def _write_debug_snapshot(page, label: str) -> None:
    out_dir = Path("results/colab_bridge_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{stamp}_{label}"
    try:
        await page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception as exc:
        print(f"Could not write screenshot: {type(exc).__name__}: {exc}")
    try:
        base.with_suffix(".html").write_text(await page.content())
    except Exception as exc:
        print(f"Could not write HTML snapshot: {type(exc).__name__}: {exc}")


async def _write_evidence(page, evidence_out: Path) -> None:
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    output_text = await _rendered_output_text(page)
    body_text = await page.locator("body").inner_text(timeout=10_000)
    evidence_out.write_text(
        "\n".join(
            [
                "# Rendered Colab output",
                output_text.strip(),
                "",
                "# Full page text",
                body_text.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote Colab evidence: {evidence_out}")


async def _rendered_output_text(page) -> str:
    chunks: list[str] = []
    for selector in OUTPUT_SELECTORS:
        locator = page.locator(selector)
        try:
            count = await locator.count()
        except Exception:
            continue
        for index in range(count):
            try:
                text = await locator.nth(index).inner_text(timeout=2_000)
            except Exception:
                continue
            text = text.strip()
            if text and text not in chunks:
                chunks.append(text)
    return "\n\n".join(chunks)


async def _wait_for_rendered_output_text(page, text: str, timeout_ms: int) -> None:
    selectors = ", ".join(OUTPUT_SELECTORS)
    await page.locator(selectors).filter(has_text=text).first.wait_for(
        timeout=timeout_ms
    )


def _validate_evidence_text(text: str) -> None:
    _raise_for_rendered_python_error(text)
    missing = [marker for marker in FOCUSED_TARGET_MARKERS if marker not in text]
    if missing:
        raise RuntimeError(
            "Colab completed, but evidence is missing required rendered output "
            f"marker(s): {', '.join(missing)}"
        )


def _validate_pinned_support_evidence(text: str) -> None:
    if '"pinned_support": true' in text:
        return

    begin = text.find(ARTIFACT_BUNDLE_BEGIN)
    end = text.find(ARTIFACT_BUNDLE_END)
    if begin < 0 or end < 0 or end <= begin:
        raise RuntimeError(
            "Colab completed, but evidence is missing required pinned-support "
            "marker and artifact bundle."
        )

    encoded = text[begin + len(ARTIFACT_BUNDLE_BEGIN) : end]
    encoded = "".join(encoded.split())
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        summary_name = "results/runs/colab_char_smoke_pinned_hep/summary.json"
        try:
            summary = json.loads(archive.read(summary_name))
        except KeyError as exc:
            raise RuntimeError(
                "Colab completed, but artifact bundle is missing required "
                f"pinned-support summary: {summary_name}"
            ) from exc

    pinned_support = summary.get("phase0", {}).get("pinned_support")
    if pinned_support is not True:
        raise RuntimeError(
            "Colab completed, but pinned-support summary did not report "
            "phase0.pinned_support true."
        )


def _validate_focused_target_artifact_bundle(text: str) -> None:
    archive_bytes = _find_colab_artifact_bundle_bytes(text)
    if archive_bytes is None:
        raise RuntimeError(
            "Colab completed, but no focused artifact bundle was found."
        )

    summary_name = f"{FOCUSED_TARGET_COMPARISON_DIR}/summary.json"
    check_name = f"{FOCUSED_TARGET_COMPARISON_DIR}/artifact_check.json"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        try:
            summary = json.loads(archive.read(summary_name))
        except KeyError as exc:
            raise RuntimeError(
                "Colab completed, but artifact bundle is missing required "
                f"focused summary: {summary_name}"
            ) from exc
        try:
            artifact_check = json.loads(archive.read(check_name))
        except KeyError as exc:
            raise RuntimeError(
                "Colab completed, but artifact bundle is missing required "
                f"focused artifact check: {check_name}"
            ) from exc

    _validate_focused_target_summary(summary, artifact_check)


def _validate_focused_target_summary(
    summary: dict[str, object],
    artifact_check: dict[str, object],
) -> None:
    failures: list[str] = []
    if summary.get("status") != "ok":
        failures.append(f"summary.status={summary.get('status')!r}")
    verdict = summary.get("verdict") if isinstance(summary.get("verdict"), dict) else {}
    if verdict.get("status") != "pass":
        failures.append(f"summary.verdict.status={verdict.get('status')!r}")
    if artifact_check.get("status") != "pass":
        failures.append(f"artifact_check.status={artifact_check.get('status')!r}")

    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    by_experiment = {
        run.get("experiment_id"): run for run in runs if isinstance(run, dict)
    }
    missing = sorted(set(FOCUSED_TARGET_RUN_SCHEMA) - set(by_experiment))
    if missing:
        failures.append(f"missing focused run(s): {', '.join(missing)}")

    for experiment_id, expected in FOCUSED_TARGET_RUN_SCHEMA.items():
        run = by_experiment.get(experiment_id)
        if not isinstance(run, dict):
            continue
        for field, expected_value in expected.items():
            if run.get(field) != expected_value:
                failures.append(
                    f"{experiment_id}.{field}={run.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        support_audit = run.get("support_audit")
        if not isinstance(support_audit, dict):
            failures.append(f"{experiment_id}.support_audit=missing")
            continue
        for field in (
            "used_columns",
            "dead_columns",
            "unique_support_sets",
            "total_support_slots",
            "support_positions",
        ):
            if support_audit.get(field) is None:
                failures.append(f"{experiment_id}.support_audit.{field}=missing")
        for field in ("num_columns", "top_k"):
            expected_value = expected[field]
            if support_audit.get(field) != expected_value:
                failures.append(
                    f"{experiment_id}.support_audit.{field}="
                    f"{support_audit.get(field)!r}, expected {expected_value!r}"
                )

    if failures:
        preview = "; ".join(failures[:8])
        if len(failures) > 8:
            preview = f"{preview}; ... ({len(failures)} total)"
        raise RuntimeError(
            "Colab completed, but focused contextual-router artifact schema is "
            f"stale or invalid: {preview}"
        )


def _raise_for_rendered_python_error(text: str) -> None:
    for marker in ERROR_MARKERS:
        index = text.find(marker)
        if index < 0:
            continue
        start = max(0, index - 800)
        end = min(len(text), index + 2400)
        excerpt = text[start:end].strip()
        raise RuntimeError(f"Colab rendered Python error marker {marker!r}:\n{excerpt}")


def _extract_colab_artifact_bundle(
    text: str,
    destination_root: Path = Path("."),
) -> list[Path]:
    archive_bytes = _find_colab_artifact_bundle_bytes(text)
    if archive_bytes is None:
        return []

    root = destination_root.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe Colab artifact path: {member.filename}")
            if member.header_offset < 0:
                continue
            target = (root / member_path).resolve()
            if root not in [target, *target.parents]:
                raise RuntimeError(f"Colab artifact escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                content = archive.read(member)
            except (ValueError, zipfile.BadZipFile):
                continue
            target.write_bytes(content)
            extracted.append(target)
    return extracted


def _find_colab_artifact_bundle_bytes(text: str) -> bytes | None:
    candidates = []
    for begin in re.finditer(re.escape(ARTIFACT_BUNDLE_BEGIN), text):
        for end in re.finditer(re.escape(ARTIFACT_BUNDLE_END), text[begin.end() :]):
            candidates.append(text[begin.end() : begin.end() + end.start()])
            break
    for end in re.finditer(re.escape(ARTIFACT_BUNDLE_END), text):
        candidates.append(_base64_block_before_marker(text, end.start()))
        prefix = text[: end.start()]
        for match in reversed(list(re.finditer(r"UEsDB[A-Za-z0-9+/=\s]*", prefix))):
            candidates.append(match.group(0))

    saw_marker = ARTIFACT_BUNDLE_BEGIN in text or ARTIFACT_BUNDLE_END in text
    saw_empty_marker = False
    for candidate in candidates:
        encoded = "".join(candidate.split())
        if not encoded:
            saw_empty_marker = True
            continue
        try:
            archive_bytes = base64.b64decode(encoded, validate=True)
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    continue
            return archive_bytes
        except (binascii.Error, zipfile.BadZipFile, RuntimeError):
            continue
        except ValueError:
            try:
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    if _archive_has_readable_focused_target(archive):
                        return archive_bytes
            except (ValueError, zipfile.BadZipFile, RuntimeError):
                continue

    if saw_empty_marker:
        raise RuntimeError("Colab artifact bundle marker was present but empty.")
    if saw_marker:
        raise RuntimeError("Colab artifact bundle marker was present but no valid zip was found.")
    return None


def _base64_block_before_marker(text: str, marker_start: int) -> str:
    lines = text[:marker_start].splitlines()
    block: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if block:
                break
            continue
        if re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
            block.append(stripped)
            continue
        break
    return "\n".join(reversed(block))


def _archive_has_readable_focused_target(archive: zipfile.ZipFile) -> bool:
    required = [
        f"{FOCUSED_TARGET_COMPARISON_DIR}/summary.json",
        f"{FOCUSED_TARGET_COMPARISON_DIR}/metrics.csv",
        f"{FOCUSED_TARGET_COMPARISON_DIR}/notes.md",
        f"{FOCUSED_TARGET_COMPARISON_DIR}/artifact_check.json",
    ]
    for name in required:
        try:
            archive.read(name)
        except (KeyError, ValueError, zipfile.BadZipFile):
            return False
    return True


async def _wait_for_completion(page, timeout_minutes: float, evidence_out: Path) -> None:
    timeout_seconds = timeout_minutes * 60
    started = asyncio.get_running_loop().time()
    runtime_prompt_confirmations = 0
    max_runtime_prompt_confirmations = 3
    print(f"waiting up to {timeout_minutes:g} minute(s) for Colab completion text")
    while True:
        try:
            await _wait_for_rendered_output_text(page, COMPLETION_TEXT, timeout_ms=3_000)
            break
        except Exception as exc:
            rendered_text = await _rendered_output_text(page)
            try:
                _raise_for_rendered_python_error(rendered_text)
            except RuntimeError:
                await _write_evidence(page, evidence_out)
                raise
            elapsed = asyncio.get_running_loop().time() - started
            if elapsed >= timeout_seconds:
                await _write_evidence(page, evidence_out)
                raise TimeoutError(
                    f"Timed out waiting for Colab completion text: {COMPLETION_TEXT!r}"
                ) from exc
            await _dismiss_obstructive_modals(page)
            confirmed_runtime_prompt = await _confirm_run_modals(
                page,
                max_rounds=1,
                timeout_ms=750,
                include_run_all=False,
                include_generic_runtime_controls=False,
            )
            if confirmed_runtime_prompt:
                runtime_prompt_confirmations += 1
                if runtime_prompt_confirmations >= max_runtime_prompt_confirmations:
                    await _write_debug_snapshot(page, "runtime_prompt_blocked")
                    await _write_evidence(page, evidence_out)
                    raise RuntimeError(
                        "Colab runtime prompts persisted after "
                        f"{max_runtime_prompt_confirmations} non-interactive "
                        "confirmation attempt(s); manual Chrome/Colab runtime "
                        "resolution is required before rerunning the bridge."
                    )

    await _write_evidence(page, evidence_out)
    text = evidence_out.read_text()
    _validate_evidence_text(text)
    _validate_focused_target_artifact_bundle(text)
    extracted = _extract_colab_artifact_bundle(text)
    if extracted:
        print(f"extracted {len(extracted)} Colab artifact file(s) from rendered bundle.")
    else:
        print("No rendered Colab artifact bundle found to extract.")
    print("Colab completion detected and evidence passed visible-output checks.")


async def _trigger_run_all(page, method: str) -> None:
    if method in {"shortcut", "both"}:
        print("triggering run all with keyboard shortcut: Meta+F9")
        await page.keyboard.press("Meta+F9")
        await page.wait_for_timeout(2_000)

    if method in {"menu", "both"}:
        ran = await _click_first(page, ["Runtime"])
        if ran:
            await page.wait_for_timeout(1_000)
            ran = await _click_first(page, ["Run all", "Run all cells"])
        if not ran:
            print("Could not trigger Run all via menus.")


async def _confirm_run_modals(
    page,
    max_rounds: int = 12,
    timeout_ms: int = 1_500,
    include_run_all: bool = True,
    include_generic_runtime_controls: bool = True,
) -> bool:
    run_all_labels = [
        "Run anyway",
        "Run all",
        "Run all cells",
    ]
    explicit_runtime_labels = [
        "Resume",
        "Reconnect",
        "Connect to a hosted runtime",
        "Restart runtime",
        "Restart and run all",
        "Continue",
        "Yes",
    ]
    generic_runtime_labels = [
        "Connect",
        "OK",
        "Ok",
    ]
    runtime_labels = [*explicit_runtime_labels]
    if include_generic_runtime_controls:
        runtime_labels.extend(generic_runtime_labels)
    labels = [*run_all_labels, *runtime_labels] if include_run_all else runtime_labels
    any_clicked = False
    for _ in range(max_rounds):
        clicked = False
        for label in labels:
            try:
                button = page.get_by_role("button", name=label, exact=False).first
                await button.click(timeout=timeout_ms)
                print(f"confirmed modal button: {label}")
                clicked = True
                any_clicked = True
                break
            except Exception:
                pass
        if not clicked:
            clicked = await _click_first(page, labels, timeout_ms=timeout_ms)
            any_clicked = any_clicked or clicked
        if not clicked:
            await page.wait_for_timeout(1_000)
    return any_clicked


async def _dismiss_obstructive_modals(page) -> bool:
    modal_patterns = [
        "Share notebook",
        "Share this notebook",
        "Sharing settings",
    ]
    dismiss_labels = [
        "Cancel",
        "Close",
        "Dismiss",
        "Not now",
        "No thanks",
        "Maybe later",
        "Skip",
    ]
    dismissed = False
    for pattern in modal_patterns:
        try:
            modal_text = page.get_by_text(pattern, exact=False).first
            await modal_text.wait_for(timeout=500)
        except Exception:
            continue

        for label in dismiss_labels:
            try:
                button = page.get_by_role("button", name=label, exact=False).first
                await button.click(timeout=500)
                print(f"dismissed obstructive modal: {pattern} via {label}")
                dismissed = True
                break
            except Exception:
                pass

        if not dismissed:
            try:
                await page.keyboard.press("Escape")
                print(f"dismissed obstructive modal: {pattern} via Escape")
                dismissed = True
            except Exception:
                pass
    return dismissed


async def _operate_page(
    page,
    manual_login: bool,
    run_all: bool,
    run_method: str,
    debug_snapshot: bool,
    wait_completion: bool,
    completion_timeout_minutes: float,
    evidence_out: Path,
) -> None:
    notebook_url = _colab_notebook_url()
    await page.goto(notebook_url, wait_until="domcontentloaded")
    print(f"opened: {notebook_url}")

    if manual_login:
        print()
        print("A browser window is open. Log into Google/Colab there if needed.")
        print("When the notebook is visible, return here and press Enter.")
        input()

    if run_all:
        await _dismiss_obstructive_modals(page)
        connected = await _click_first(page, ["Connect", "Reconnect"])
        if not connected:
            print("Could not find a Connect button; it may already be connected.")

        await page.wait_for_timeout(8_000)
        await _dismiss_obstructive_modals(page)
        await _trigger_run_all(page, method=run_method)
        await page.wait_for_timeout(3_000)
        await _dismiss_obstructive_modals(page)
        await _confirm_run_modals(page)
        await page.wait_for_timeout(5_000)
        await _dismiss_obstructive_modals(page)
        if debug_snapshot:
            await _write_debug_snapshot(page, "after_run_all")

        print("Run-all was requested. Watch the browser for completion/errors.")
        if wait_completion:
            await _wait_for_completion(
                page,
                timeout_minutes=completion_timeout_minutes,
                evidence_out=evidence_out,
            )
        else:
            print("Completion detection is disabled; pass --wait-completion to enable it.")


async def automate(
    manual_login: bool,
    run_all: bool,
    headed: bool,
    browser_channel: str | None,
    profile_dir: Path,
    cdp_url: str | None,
    run_method: str,
    debug_snapshot: bool,
    wait_completion: bool,
    completion_timeout_minutes: float,
    evidence_out: Path,
    pause_before_close: bool,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        print("Playwright is not installed. Try: python -m pip install playwright")
        print("Then, if needed: python -m playwright install chromium")
        raise

    async with async_playwright() as p:
        if cdp_url:
            browser = await p.chromium.connect_over_cdp(cdp_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            await _operate_page(
                page,
                manual_login=manual_login,
                run_all=run_all,
                run_method=run_method,
                debug_snapshot=debug_snapshot,
                wait_completion=wait_completion,
                completion_timeout_minutes=completion_timeout_minutes,
                evidence_out=evidence_out,
            )
            if pause_before_close:
                print("Leaving connected Chrome open. Press Enter to detach automation.")
                input()
            else:
                print("Detaching automation from connected Chrome.")
            await browser.close()
            return

        browser_type = p.chromium
        context = await browser_type.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=not headed,
            channel=browser_channel,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await _operate_page(
            page,
            manual_login=manual_login,
            run_all=run_all,
            run_method=run_method,
            debug_snapshot=debug_snapshot,
            wait_completion=wait_completion,
            completion_timeout_minutes=completion_timeout_minutes,
            evidence_out=evidence_out,
        )

        if headed and pause_before_close:
            print("Leaving browser open. Press Enter to close this automation context.")
            input()
        await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open/run RelaLeap Colab notebook.")
    parser.add_argument("--manual-login", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--run-method",
        choices=["shortcut", "menu", "both"],
        default="shortcut",
        help="How to trigger Colab Run all. Keyboard shortcut is most reliable on current Colab.",
    )
    parser.add_argument(
        "--debug-snapshot",
        action="store_true",
        help="Write a screenshot and HTML snapshot after requesting Run all.",
    )
    parser.add_argument(
        "--wait-completion",
        action="store_true",
        help="Wait for the final RelaLeap Colab completion text and fail if it is not observed.",
    )
    parser.add_argument(
        "--completion-timeout-minutes",
        type=float,
        default=45.0,
        help="Maximum minutes to wait for Colab completion text.",
    )
    parser.add_argument(
        "--evidence-out",
        default="results/colab_bridge_evidence/latest_colab_output.txt",
        help="Path to save visible Colab page text after completion or timeout.",
    )
    parser.add_argument(
        "--pause-before-close",
        action="store_true",
        help="Pause for Enter before detaching/closing. Intended for manual debugging only.",
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chrome", "chrome-beta", "chrome-dev", "msedge"],
        default=None,
        help="Use an installed browser channel instead of bundled Chromium.",
    )
    parser.add_argument(
        "--profile-dir",
        default=".colab-browser-profile",
        help="Persistent browser profile directory for launched browser mode.",
    )
    parser.add_argument(
        "--cdp-url",
        default=None,
        help="Connect to an already-running Chrome via remote debugging, e.g. http://127.0.0.1:9222.",
    )
    args = parser.parse_args()

    import asyncio

    try:
        asyncio.run(
            automate(
                manual_login=args.manual_login,
                run_all=args.run_all,
                headed=not args.headless,
                browser_channel=args.browser_channel,
                profile_dir=Path(args.profile_dir).resolve(),
                cdp_url=args.cdp_url,
                run_method=args.run_method,
                debug_snapshot=args.debug_snapshot,
                wait_completion=args.wait_completion,
                completion_timeout_minutes=args.completion_timeout_minutes,
                evidence_out=Path(args.evidence_out),
                pause_before_close=args.pause_before_close,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        message = str(exc).splitlines()[0]
        print("Colab bridge failed before completion.")
        print(f"{type(exc).__name__}: {message}")
        print(
            "Open the notebook manually and run all cells if browser "
            "automation is blocked by auth, session, quota, UI, or sandbox state."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
