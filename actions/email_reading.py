"""Read-only, replaceable mailbox interface backed by local Outlook on macOS."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Protocol


@dataclass(frozen=True)
class EmailRequest:
    limit: int = 5
    include_body: bool = True

    def __post_init__(self):
        if type(self.limit) is not int or not 1 <= self.limit <= 10:
            raise ValueError("Request between one and ten unread emails.")
        if type(self.include_body) is not bool:
            raise ValueError("include_body must be a boolean.")


@dataclass(frozen=True)
class EmailMessage:
    id: int | str
    subject: str
    sender: str
    received_at: str
    body: str = ""
    body_truncated: bool = False


@dataclass(frozen=True)
class EmailResult:
    messages: tuple[EmailMessage, ...] = ()
    unread_count: int = 0
    error: str = ""
    scope: str = "default_inbox"
    mailbox_label: str = "Outlook's default Inbox"


class EmailReader(Protocol):
    """Provider contract for unread snapshots; no mailbox mutation interface."""

    def read_unread(self, request: EmailRequest) -> EmailResult:
        ...


_ERRORS = {
    "mailbox_unavailable": "Outlook's scripting interface doesn't expose your mailbox. A read-only provider connection is needed; this does not mean your Inbox is empty.",
    "not_running": "Open Outlook and sign in first. I won't open messages or change their unread status.",
    "automation": "Allow Airis or its launcher to access Outlook in System Settings > Privacy & Security > Automation, then try again.",
    "unsupported": "Outlook couldn't provide unread email data. Check that your Outlook version supports its macOS scripting interface.",
    "mailbox_changed": "Your Inbox changed while I was reading it. Please try again.",
    "too_many": "This Inbox has more than 10,000 unread emails, beyond the current reader's scan limit.",
}


class OutlookMailReader:
    """Only fetch unread messages; never launch, select, send, or modify mail.

    macOS Automation grants broader permission than this interface exposes.
    A fixed getter-only script enforces Airis's read-only capability.
    """

    def __init__(self, *, runner=subprocess.run):
        self._runner = runner

    def read_unread(self, request: EmailRequest) -> EmailResult:
        if not isinstance(request, EmailRequest):
            raise TypeError("The mailbox reader requires an EmailRequest.")
        if sys.platform != "darwin":
            return EmailResult(error="Local Outlook email reading is currently available only on macOS.")
        try:
            result = self._runner(
                ["/usr/bin/osascript", "-l", "JavaScript", str(Path(__file__).with_name("outlook_mail.js")),
                 "unread", str(request.limit), "body" if request.include_body else "headers"],
                check=True, capture_output=True, text=True, timeout=25)
            data = json.loads(result.stdout)
            if not isinstance(data, dict) or type(data.get("ok")) is not bool:
                raise ValueError("Invalid mailbox result.")
            if not data["ok"]:
                return EmailResult(error=_ERRORS.get(data.get("error"), _ERRORS["unsupported"]))
            count, rows = data.get("unread_count"), data.get("messages")
            if (type(count) is not int or not 0 <= count <= 10000 or data.get("scope") != "default_inbox"
                    or not isinstance(rows, list) or len(rows) != min(count, request.limit)):
                raise ValueError("Invalid unread email collection.")
            messages = []
            dates = []
            ids = set()
            for row in rows:
                if (not isinstance(row, dict) or type(row.get("id")) is not int
                        or row["id"] in ids or row.get("unread") is not True
                        or type(row.get("body_truncated")) is not bool):
                    raise ValueError("Invalid unread email metadata.")
                for field, maximum in (("subject", 300), ("sender", 300), ("received_at", 40), ("body", 2000)):
                    if not isinstance(row.get(field), str) or len(row[field]) > maximum:
                        raise ValueError("Invalid email text.")
                date = datetime.fromisoformat(row["received_at"].replace("Z", "+00:00"))
                if date.tzinfo is None or (not request.include_body and (row["body"] or row["body_truncated"])):
                    raise ValueError("Invalid email scope.")
                ids.add(row["id"])
                dates.append(date)
                messages.append(EmailMessage(**{key: row[key] for key in
                    ("id", "subject", "sender", "received_at", "body", "body_truncated")}))
            if dates != sorted(dates, reverse=True):
                raise ValueError("Emails were not sorted newest first.")
            return EmailResult(tuple(messages), count)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, OverflowError):
            return EmailResult(error="I couldn't read Outlook emails. Check Outlook's scripting support and Airis's Outlook permission under System Settings > Privacy & Security > Automation.")
