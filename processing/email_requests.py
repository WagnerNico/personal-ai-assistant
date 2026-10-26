"""Recognize explicit mailbox requests without giving the model write access."""

import re

from actions.email_reading import EmailRequest


READ_ONLY_REPLY = "I can only read emails. I can't send, reply, delete, or change their read status."
_MAIL = r"(?:e-?mails?|messages|inbox|postfach|nachrichten|gmail)"


def email_request(text):
    """Return a bounded unread request, a read-only refusal, or no mailbox intent."""
    command = text.strip().rstrip(".!?")
    command = re.sub(r"(?:,?\s+(?:please|for me|bitte))+$", "", command, flags=re.I)
    command = re.sub(r"^(?:airis,?\s+)?(?:(?:please|can you|could you|would you|bitte|kannst du|könntest du)\s+)*",
                     "", command, flags=re.I)
    if not re.search(r"\b" + _MAIL + r"\b", command, re.I):
        return None
    if re.match(r"(?:send|reply|delete|remove|mark|archive|forward|sende|antworte|lösche|markiere|archiviere)\b", command, re.I):
        return READ_ONLY_REPLY
    direct = re.match(r"(?:read(?:\s+out)?|summari[sz]e|check|list|show|tell me(?:\s+about)?|"
                      r"give me|inform me(?:\s+about)?|update me(?:\s+on)?|lese|lies|prüfe|zeige|fass(?:e)?)\b", command, re.I)
    question = re.match(r"(?:what(?:'s| is| are)|which|do i have|are there|have i (?:got|received)|"
                        r"was sind|welche|habe ich|gibt es)\b", command, re.I)
    if not direct and not question:
        if not re.fullmatch(r"(?:meine[nr]?\s+)?(?:ungelesen\w*\s+)?(?:e-?mails?|nachrichten)\s+(?:vorlesen|zusammenfassen|anzeigen)", command, re.I):
            return None
    # Discussion of email capabilities and programming examples is ordinary chat.
    if re.search(r"\b(?:example|code|function|tutorial|how to|can airis|beispiel|funktion)\b", command, re.I):
        return None
    if re.search(r"[;\n]|\b(?:and|then|und|danach)\s+(?:send|reply|delete|mark|archive|forward|open|search|sende|lösche|markiere)\b", command, re.I):
        return READ_ONLY_REPLY
    if not (re.search(r"\b(?:my|meine[nr]?|unread|ungelesen\w*|inbox|postfach|outlook|new|latest|recent|neuest\w*)\b", command, re.I)
            or re.fullmatch(r"(?:check|read|list|show)\s+(?:the\s+)?e-?mails?", command, re.I)):
        return None
    if re.search(r"\b(?:sent|deleted|drafts|gelesene|gesendete|entwürfe|all (?:my )?emails|from|since|before|after|containing|von|seit)\b", command, re.I):
        return "I currently read only unread emails in the connected account's Inbox."
    numbers = re.findall(r"\b\d+\b", command)
    if len(numbers) > 1 or (numbers and not 1 <= int(numbers[0]) <= 10):
        return "Please request between one and ten unread emails at a time."
    limit = int(numbers[0]) if numbers else 1 if re.search(r"\b(?:last|latest)\s+(?:unread\s+)?email\b", command, re.I) else 5
    return EmailRequest(limit, include_body=not bool(re.match(r"(?:list|show|which|do i have|are there|have i)", command, re.I)))
