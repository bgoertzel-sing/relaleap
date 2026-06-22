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
from datetime import datetime, timezone
import io
import sys
from pathlib import Path
import zipfile


COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/"
    "bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb"
)
COMPLETION_TEXT = "RelaLeap Colab Phase 0 comparison completed."
ARTIFACT_BUNDLE_BEGIN = "RELALEAP_ARTIFACT_BUNDLE_ZIP_BASE64_BEGIN"
ARTIFACT_BUNDLE_END = "RELALEAP_ARTIFACT_BUNDLE_ZIP_BASE64_END"
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
    required = [
        'cuda_available: True',
        '"status": "pass"',
        "Accepted HEP alpha:",
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
        "char_validation_pc_hep_support_stress_temporal_clipped",
        "char_validation_hep_temporal_clipped_objective_gate",
        "char_validation_pc_hep_temporal_clipped_objective_gate",
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
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            "Colab completed, but evidence is missing required rendered output "
            f"marker(s): {', '.join(missing)}"
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
    begin = text.find(ARTIFACT_BUNDLE_BEGIN)
    end = text.find(ARTIFACT_BUNDLE_END)
    if begin < 0 or end < 0 or end <= begin:
        return []

    encoded = text[begin + len(ARTIFACT_BUNDLE_BEGIN) : end]
    encoded = "".join(encoded.split())
    if not encoded:
        raise RuntimeError("Colab artifact bundle marker was present but empty.")

    root = destination_root.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe Colab artifact path: {member.filename}")
            target = (root / member_path).resolve()
            if root not in [target, *target.parents]:
                raise RuntimeError(f"Colab artifact escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            extracted.append(target)
    return extracted


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
    await page.goto(COLAB_NOTEBOOK_URL, wait_until="domcontentloaded")
    print(f"opened: {COLAB_NOTEBOOK_URL}")

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
