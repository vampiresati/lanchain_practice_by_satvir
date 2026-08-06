"""
Read a command from a WhatsApp group, answer it with the Jira agent, and post
the resulting ticket numbers to another WhatsApp group.

Flow:
    1. Open the command group (default: ai_commands) and read the last message.
    2. Feed that message to the Jira agent as its prompt.
    3. Pull the ticket keys out of the agent run.
    4. Post the ticket keys to the target group (default: ai_tickets_to_test).

Both WhatsApp steps share a single browser session, so the login/QR wait only
happens once.

Usage:
    python jira_whatsapp_bot.py                 # read, ask Jira, send
    python jira_whatsapp_bot.py --dry-run       # do everything except send
    python jira_whatsapp_bot.py --prompt "..."  # skip the read, use this prompt
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv

import langchain_whats_app as wa
from langchain_jira_agent import agent

load_dotenv()

COMMAND_GROUP = os.getenv("WHATSAPP_COMMAND_GROUP", "ai_commands")
TARGET_GROUP = os.getenv("WHATSAPP_TARGET_GROUP", "ai_tickets_to_test")

# Jira keys look like TRAN-4081: project key, hyphen, number.
TICKET_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{1,20}-\d+\b")


# ============================================================
# Jira agent
# ============================================================

def _keys_from_tool_messages(messages) -> list[str]:
    """
    Pull ticket keys out of the agent's tool results.

    This is preferred over reading the agent's prose: the tool output is the
    raw Jira response, so it cannot contain a key the model invented.
    """

    keys: list[str] = []

    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        content = message.content
        if not isinstance(content, str):
            continue

        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            continue

        if not isinstance(payload, dict) or not payload.get("success"):
            continue

        for key in payload.get("ticket_numbers", []) or []:
            if isinstance(key, str):
                keys.append(key)

        ticket = payload.get("ticket")
        if isinstance(ticket, dict) and isinstance(ticket.get("key"), str):
            keys.append(ticket["key"])

    return keys


def _dedupe(keys) -> list[str]:
    """Drop duplicates while keeping the original order."""

    seen = set()
    unique = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def ask_jira_agent(prompt: str) -> tuple[str, list[str], bool]:
    """
    Run the Jira agent on `prompt`.

    Returns (final answer text, ticket keys, keys_came_from_tools).
    """

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result["messages"]

    final = messages[-1].content
    answer = final if isinstance(final, str) else json.dumps(final, indent=2)

    keys = _dedupe(_keys_from_tool_messages(messages))
    if keys:
        return answer, keys, True

    # Nothing usable in the tool output — fall back to scraping the prose.
    # Flagged as unverified because a model can fabricate a key here.
    return answer, _dedupe(TICKET_KEY_PATTERN.findall(answer)), False


def format_report(prompt: str, keys: list[str], verified: bool) -> str:
    """
    Build the WhatsApp message to post to the target group.

    Deliberately plain: the composer auto-formats text, so a line starting with
    "- " or "* " is turned into a bullet list, and blank lines come out doubled.
    Keeping every line short and unprefixed avoids both.
    """

    lines = [f"Request: {prompt}"]

    if keys:
        lines.append(f"Tickets ({len(keys)}): {', '.join(keys)}")
        if not verified:
            lines.append("(Parsed from the agent's answer, not from Jira directly.)")
    else:
        lines.append("No matching tickets found.")

    return "\n".join(lines)


# ============================================================
# WhatsApp
# ============================================================

def read_group_command(page, group_name: str) -> str:
    """Open `group_name` and return its most recent message."""

    wa._open_chat_by_name(page, group_name)
    return wa._extract_last_visible_message(page)


def send_group_message(page, group_name: str, message: str) -> None:
    """
    Open `group_name` and send `message`.

    Newlines are typed as Shift+Enter. A bare Enter would send the message, so
    typing a multi-line string directly would post one message per line.
    Keep `message` free of "- "/"* " line prefixes — see format_report().
    """

    wa._open_chat_by_name(page, group_name)
    composer = wa._wait_for_composer(page)

    lines = message.split("\n")
    wa._type_text(composer, lines[0])
    for line in lines[1:]:
        composer.press("Shift+Enter")
        if line:
            composer.press_sequentially(line, delay=15)

    composer.press("Enter")
    wa._wait_until_sent(page, composer)


# ============================================================
# Entry point
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-group", default=COMMAND_GROUP,
                        help=f"group to read the prompt from (default: {COMMAND_GROUP})")
    parser.add_argument("--target-group", default=TARGET_GROUP,
                        help=f"group to post ticket numbers to (default: {TARGET_GROUP})")
    parser.add_argument("--prompt", default=None,
                        help="use this prompt instead of reading the command group")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report instead of sending it")
    parser.add_argument("--headless", action="store_true",
                        help="run the browser headlessly (WhatsApp Web often rejects this)")
    args = parser.parse_args()

    playwright, browser, page = wa._launch_whatsapp(headless=args.headless)

    try:
        if args.prompt:
            prompt = args.prompt
            print(f"Prompt (from --prompt): {prompt}")
        else:
            print(f"Reading last message from '{args.command_group}'...")
            prompt = read_group_command(page, args.command_group)
            print(f"Prompt: {prompt}")

        if not prompt.strip():
            print("The command message is empty — nothing to ask Jira.", file=sys.stderr)
            return 1

        print("Asking the Jira agent...")
        answer, keys, verified = ask_jira_agent(prompt)
        print(f"Agent answer: {answer}")
        print(f"Ticket keys: {keys or '(none)'}  (from tools: {verified})")

        report = format_report(prompt, keys, verified)

        if args.dry_run:
            print("\n--- dry run, not sending ---")
            print(f"to: {args.target_group}")
            print(report)
            return 0

        print(f"\nSending to '{args.target_group}'...")
        send_group_message(page, args.target_group, report)
        print("Sent.")
        return 0

    finally:
        wa._close_whatsapp(playwright, browser)


if __name__ == "__main__":
    raise SystemExit(main())
