#!/usr/bin/env python3
"""Best-effort Colab browser automation for RelaLeap.

This is intentionally a temporary bridge. It opens the GitHub-backed Colab
notebook and tries to connect and run all cells through the web UI. Expect this
to need occasional selector fixes when Colab changes.

Usage:
  python tools/colab_playwright_runner.py --manual-login
  python tools/colab_playwright_runner.py --run-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/"
    "bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb"
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


async def automate(manual_login: bool, run_all: bool, headed: bool) -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        print("Playwright is not installed. Try: python -m pip install playwright")
        print("Then, if needed: python -m playwright install chromium")
        raise

    profile_dir = Path(".colab-browser-profile").resolve()
    async with async_playwright() as p:
        browser_type = p.chromium
        context = await browser_type.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=not headed,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(COLAB_NOTEBOOK_URL, wait_until="domcontentloaded")
        print(f"opened: {COLAB_NOTEBOOK_URL}")

        if manual_login:
            print()
            print("A browser window is open. Log into Google/Colab there if needed.")
            print("When the notebook is visible, return here and press Enter.")
            input()

        if run_all:
            connected = await _click_first(page, ["Connect", "Reconnect"])
            if not connected:
                print("Could not find a Connect button; it may already be connected.")

            await page.wait_for_timeout(8_000)
            ran = await _click_first(page, ["Runtime"])
            if ran:
                await page.wait_for_timeout(1_000)
                ran = await _click_first(page, ["Run all", "Run all cells"])
            if not ran:
                print("Could not trigger Run all via menus; trying keyboard shortcut.")
                await page.keyboard.press("Meta+F9")

            await page.wait_for_timeout(3_000)
            await _click_first(page, ["Run anyway", "Run all", "Yes"])

            print("Run-all was requested. Watch the browser for completion/errors.")
            print("This helper does not yet reliably detect Colab completion.")

        if headed:
            print("Leaving browser open. Press Enter to close this automation context.")
            input()
        await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open/run RelaLeap Colab notebook.")
    parser.add_argument("--manual-login", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    import asyncio

    try:
        asyncio.run(
            automate(
                manual_login=args.manual_login,
                run_all=args.run_all,
                headed=not args.headless,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

