from dotenv import load_dotenv
import os

from playwright.sync_api import sync_playwright
from pathlib import Path
import urllib.parse
import time

load_dotenv()

contact = os.getenv("WHATSAPP_NUMBER")
message = os.getenv("WHATSAPP_MESSAGE")

if not contact or not message:
    raise ValueError("WHATSAPP_NUMBER and WHATSAPP_MESSAGE must be set in .env")



SESSION_DIR = Path("whatsapp-session")


def send_whatsapp_message(
    phone: str,
    message: str,
    session_dir: str | Path = SESSION_DIR,
    headless: bool = False,
    wait_before_send: int = 5,
) -> None:
    """
    Send a WhatsApp message using WhatsApp Web.

    Args:
        phone: Phone number in international format without '+'.
               Example: '15551234567'
        message: Message to send.
        session_dir: Directory where Playwright stores the login session.
        headless: Whether to run the browser headlessly.
        wait_before_send: Seconds to wait before pressing Enter.
    """

    encoded = urllib.parse.quote(message)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            str(session_dir),
            headless=headless,
        )

        page = browser.new_page()

        page.goto(
            f"https://web.whatsapp.com/send?phone={phone}&text={encoded}",
            wait_until="domcontentloaded",
        )

        print("Waiting for WhatsApp Web...")
        print("Scan the QR code if prompted.")

        message_box = page.locator("footer [contenteditable='true']").last
        message_box.wait_for(timeout=120000)

        if wait_before_send:
            print(f"Sending in {wait_before_send} seconds...")
            time.sleep(wait_before_send)

        message_box.press("Enter")

        time.sleep(2)
        browser.close()

if __name__ == "__main__":
    send_whatsapp_message(
        phone=contact,
        message=message,
        headless=False,
    )