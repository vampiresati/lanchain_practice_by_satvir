import os
import time
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import Locator, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

load_dotenv()

SESSION_DIR = Path("whatsapp-session")
WHATSAPP_WEB_URL = "https://web.whatsapp.com/"

CHAT_LIST_SELECTOR = "#pane-side"
CONVERSATION_SELECTOR = "#main"


def _launch_whatsapp(
    session_dir: str | Path = SESSION_DIR,
    headless: bool = False,
    login_timeout: int = 120000,
):
    """Open WhatsApp Web in a persistent browser context and wait until it is usable."""

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch_persistent_context(
        str(session_dir),
        headless=headless,
        viewport={"width": 1280, "height": 900},
    )

    # A persistent context already ships with one page; opening a second one
    # leaves a stray blank tab around and makes WhatsApp complain about
    # multiple sessions.
    page = browser.pages[0] if browser.pages else browser.new_page()

    try:
        page.goto(WHATSAPP_WEB_URL, wait_until="domcontentloaded")
        print("Waiting for WhatsApp Web...")
        print("Scan the QR code if prompted.")
        _wait_until_logged_in(page, timeout=login_timeout)
    except Exception:
        _close_whatsapp(playwright, browser)
        raise

    return playwright, browser, page


def _wait_until_logged_in(page: Page, timeout: int = 120000) -> None:
    """Block until the chat list is rendered (i.e. QR login is complete)."""

    page.wait_for_selector(CHAT_LIST_SELECTOR, timeout=timeout, state="visible")


def _close_whatsapp(playwright, browser) -> None:
    try:
        browser.close()
    finally:
        playwright.stop()


def _box_text(target: Locator) -> str:
    """Read back the current value of an <input> or a contenteditable box."""

    if target.evaluate("el => el.tagName") == "INPUT":
        return target.input_value()
    return target.inner_text()


def _type_text(target: Locator, text: str) -> None:
    """
    Type into a WhatsApp text box.

    The message composer is a rich-text (Lexical) contenteditable: Locator.fill()
    sets the DOM text without emitting the keystrokes the app listens for, so the
    message is never registered. Real key events are required. The search box is
    a plain <input>, where typing works too, so one path covers both.
    """

    target.click()
    target.press("ControlOrMeta+a")
    target.press("Backspace")
    if text:
        target.press_sequentially(text, delay=25)


def _wait_for_composer(page: Page, timeout: int = 30000) -> Locator:
    composer = page.locator(f"{CONVERSATION_SELECTOR} footer [contenteditable='true']").last
    composer.wait_for(timeout=timeout, state="visible")
    return composer


def _wait_for_search_box(page: Page, timeout: int = 30000) -> Locator:
    """
    Find the chat search box.

    Current WhatsApp Web renders this as a real
    <input type="text" role="textbox" data-tab="3" aria-label="Search or start a new chat">.
    Older builds used a contenteditable div, which is kept here as a fallback.
    """

    candidate_selectors = [
        "input[data-tab='3']",
        "input[type='text'][aria-label*='Search' i]",
        "input[type='text'][placeholder*='Search' i]",
        "[role='textbox'][data-tab='3']",
        # Legacy contenteditable variants.
        "div[role='textbox'][title='Search input textbox']",
        "div[contenteditable='true'][data-tab='3']",
        "div[contenteditable='true'][data-tab='2']",
    ]

    deadline = time.time() + (timeout / 1000)
    while time.time() < deadline:
        for selector in candidate_selectors:
            locator = page.locator(selector)
            for index in range(locator.count()):
                candidate = locator.nth(index)
                try:
                    if not candidate.is_visible():
                        continue
                    # Skip the message composer, which matches some of these too.
                    if candidate.evaluate("el => !!el.closest('footer')"):
                        continue
                    return candidate
                except Exception:
                    continue

        page.wait_for_timeout(500)

    raise RuntimeError(
        "Could not find the WhatsApp search box after "
        f"{timeout / 1000:.0f}s. WhatsApp Web may have changed its markup again — "
        "inspect the search field and add its selector to _wait_for_search_box()."
    )


def _click_search_result(page: Page, chat_name: str, timeout: int = 15000) -> bool:
    """Try the known ways of clicking a chat in the search results pane."""

    pane = page.locator(CHAT_LIST_SELECTOR)
    candidates = [
        pane.get_by_title(chat_name, exact=True),
        pane.get_by_role("gridcell", name=chat_name, exact=True),
        pane.get_by_text(chat_name, exact=True),
    ]

    deadline = time.time() + (timeout / 1000)
    while time.time() < deadline:
        for candidate in candidates:
            target = candidate.first
            try:
                if target.count() and target.is_visible():
                    target.click()
                    return True
            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue
        page.wait_for_timeout(500)

    return False


def _open_chat_by_name(page: Page, chat_name: str, login_timeout: int = 120000) -> None:
    if not chat_name.strip():
        raise ValueError("chat_name must not be empty")

    _wait_until_logged_in(page, timeout=login_timeout)
    search_box = _wait_for_search_box(page)

    _type_text(search_box, chat_name)
    if chat_name not in _box_text(search_box):
        # Some builds swallow synthetic keystrokes; fall back to a direct set.
        search_box.fill(chat_name)

    if not _click_search_result(page, chat_name):
        raise RuntimeError(f"Could not find a chat or group named '{chat_name}'.")

    # Wait for the conversation pane itself, not just any composer.
    page.wait_for_selector(CONVERSATION_SELECTOR, timeout=30000, state="visible")
    _wait_for_composer(page)


def _extract_last_visible_message(page: Page, timeout: int = 30000) -> str:
    """Return the text of the newest message in the open conversation."""

    rows = page.locator(f"{CONVERSATION_SELECTOR} div[role='row']")
    try:
        rows.last.wait_for(timeout=timeout, state="visible")
    except PlaywrightTimeoutError:
        raise RuntimeError("Could not find any visible messages in the selected chat.")

    # Walk backwards: the last rows can be date dividers or system notices that
    # carry no message text.
    for index in range(rows.count() - 1, -1, -1):
        row = rows.nth(index)
        for selector in ("span.selectable-text", ".copyable-text"):
            texts = [
                text.strip()
                for text in row.locator(selector).all_inner_texts()
                if text.strip()
            ]
            if texts:
                return texts[-1]

    raise RuntimeError("Could not find any visible messages in the selected chat.")


def send_whatsapp_message(
    phone: str,
    message: str,
    session_dir: str | Path = SESSION_DIR,
    headless: bool = False,
    wait_before_send: int = 5,
) -> None:
    """
    Send a WhatsApp message to a phone number using WhatsApp Web.

    Args:
        phone: Phone number in international format without '+'.
               Example: '15551234567'
        message: Message to send.
        session_dir: Directory where Playwright stores the login session.
        headless: Whether to run the browser headlessly.
        wait_before_send: Seconds to wait before pressing Enter.
    """

    encoded = urllib.parse.quote(message)
    playwright, browser, page = _launch_whatsapp(session_dir=session_dir, headless=headless)

    try:
        page.goto(
            f"https://web.whatsapp.com/send?phone={phone}&text={encoded}",
            wait_until="domcontentloaded",
        )

        message_box = _wait_for_composer(page)

        if wait_before_send:
            print(f"Sending in {wait_before_send} seconds...")
            time.sleep(wait_before_send)

        message_box.click()
        message_box.press("Enter")
        _wait_until_sent(page, message_box)
    finally:
        _close_whatsapp(playwright, browser)


def send_whatsapp_group_message(
    group_name: str,
    message: str,
    session_dir: str | Path = SESSION_DIR,
    headless: bool = False,
    wait_before_send: int = 2,
) -> None:
    """
    Send a WhatsApp message to a specific group by its visible chat name.
    """

    playwright, browser, page = _launch_whatsapp(session_dir=session_dir, headless=headless)

    try:
        _open_chat_by_name(page, group_name)
        message_box = _wait_for_composer(page)
        _type_text(message_box, message)

        if wait_before_send:
            print(f"Sending to group '{group_name}' in {wait_before_send} seconds...")
            time.sleep(wait_before_send)

        message_box.press("Enter")
        _wait_until_sent(page, message_box)
    finally:
        _close_whatsapp(playwright, browser)


def _wait_until_sent(page: Page, message_box: Locator, timeout: int = 15000) -> None:
    """Wait for the composer to clear, which means WhatsApp accepted the message."""

    deadline = time.time() + (timeout / 1000)
    while time.time() < deadline:
        try:
            if not message_box.inner_text().strip():
                return
        except Exception:
            return
        page.wait_for_timeout(250)

    raise RuntimeError("Message does not appear to have been sent (composer still has text).")


def read_last_message_from_group(
    group_name: str,
    session_dir: str | Path = SESSION_DIR,
    headless: bool = False,
) -> str:
    """
    Open a specific group and return the last visible message text.
    """

    playwright, browser, page = _launch_whatsapp(session_dir=session_dir, headless=headless)

    try:
        _open_chat_by_name(page, group_name)
        return _extract_last_visible_message(page)
    finally:
        _close_whatsapp(playwright, browser)


if __name__ == "__main__":
    contact = os.getenv("WHATSAPP_NUMBER")
    message = os.getenv("WHATSAPP_MESSAGE")

    # send_whatsapp_message(
    #     phone=contact,
    #     message=message,
    #     headless=False,
    # )
    last_message = read_last_message_from_group(
        group_name="ai_commands",
        headless=False,
    )
    print(last_message)
