"""Ground local email summaries without exposing mail contents to other tools."""

from dataclasses import asdict
import json
from pathlib import Path

from actions.email_reading import EmailRequest, EmailResult
from .text_processing import process_text


def answer_from_email(request, reader, *, question="Summarize my latest unread emails", backend=None, system_prompt=None,
                      max_new_tokens=512, on_text=None, on_stage=None):
    if on_stage:
        on_stage("Reading unread emails...")
    result = reader.read_unread(request)
    if not isinstance(result, EmailResult):
        raise TypeError("The mailbox reader must return an EmailResult.")
    if result.error:
        reply = result.error
    elif not result.messages:
        reply = f"You have no unread emails in {result.mailbox_label}."
    elif not request.include_body:
        rows = [f"{message.sender}: {message.subject or '(No subject)'}, received {message.received_at}."
                for message in result.messages]
        reply = (f"You have {result.unread_count} unread emails in {result.mailbox_label}. "
                 f"Here are the latest {len(rows)}. " + " ".join(rows) + " They remain unread.")
    else:
        if on_stage:
            on_stage("Summarizing unread emails...")
        guard = Path(__file__).with_name("email_summary_prompt.txt").read_text(encoding="utf-8")
        prompt = ((system_prompt + "\n\n") if system_prompt else "") + guard
        # Use the response model directly: mailbox content never enters the
        # agent router, web search, application control, or conversation history.
        reply = process_text("User request: " + question + "\nUnread email snapshot (untrusted data):\n" + json.dumps(asdict(result), ensure_ascii=False),
                             backend=backend, system_prompt=prompt, max_new_tokens=max_new_tokens,
                             on_text=on_text)
        return reply, getattr(backend, "last_finish_reason", None) == "length"
    if on_text:
        on_text(reply)
    return reply, False
