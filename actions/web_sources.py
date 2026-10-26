"""Extract and enforce source preferences independently of a knowledge domain."""

from dataclasses import dataclass
import re
from urllib.parse import urlsplit


_TARGET = r"(?:https?://[^\s<>\"']+|(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/[^\s,]*)?)"
_DOMAIN_SOURCE = re.compile(
    rf"\b(?:use|using|from|via|according to|on|auf|von|laut|nutze|verwende|nimm)\s+"
    rf"(?:the\s+|die\s+|der\s+)?(?P<target>{_TARGET})"
    r"(?:\s+(?:as\s+(?:the\s+)?source|als\s+quelle))?", re.I)
_NAMED_SOURCE = re.compile(
    r"\b(?:use|using|nutze|verwende|nimm)\s+(?P<target>[\w .-]{1,80}?)"
    r"\s+(?:as\s+(?:the\s+)?source|als\s+quelle)\b", re.I)
_ALIASES = {"wikipedia": "wikipedia.org", "bbc": "bbc.com", "reuters": "reuters.com",
            "nasa": "nasa.gov", "yahoo finance": "finance.yahoo.com"}
_KNOWN_SOURCE = re.compile(r"\b(?:from|according to|use|using|via|von|laut|nutze|verwende)\s+"
                           r"(?P<target>" + "|".join(re.escape(name) for name in _ALIASES) + r")\b", re.I)


@dataclass(frozen=True)
class SourcePreference:
    target: str
    query: str
    explicit: bool = False


def source_preference(text):
    """Keep the requested website separate from the subject being searched."""
    match = _DOMAIN_SOURCE.search(text) or _NAMED_SOURCE.search(text) or _KNOWN_SOURCE.search(text)
    explicit = match is not None
    if match is None:
        # Dotted code names such as urllib.parse are not source preferences.
        match = re.search(r"(?P<target>https?://[^\s<>\"']+)", text, re.I)
        if match is None:
            match = re.fullmatch(rf"(?P<target>{_TARGET})[.!?]?", text.strip(), re.I)
    if match is None:
        return None
    target = match["target"].rstrip(".,);!?")
    query = (text[:match.start()] + " " + text[match.end():]).strip(" ,;:")
    return SourcePreference(_ALIASES.get(target.casefold(), target), " ".join(query.split()), explicit)


def source_domain(target):
    """Validate a source without DNS; the network reader validates public IPs."""
    from .web import WebError, public_url
    if not isinstance(target, str):
        raise WebError("Please provide the source's website address.")
    target = _ALIASES.get(target.casefold(), target)
    if "." not in target and "://" not in target:
        code = "source_identity" if re.fullmatch(r"[\w -]{1,80}", target) else ""
        raise WebError(f"Please provide the website address for {target}.", code=code)
    _, host, _, _ = public_url(target if "://" in target else "https://" + target, resolve=False)
    return host.casefold().removeprefix("www.")


def matches_source(url, domain):
    """Accept the requested host and its subdomains, never lookalike suffixes."""
    host = (urlsplit(url).hostname or "").casefold().rstrip(".").removeprefix("www.")
    return host == domain or host.endswith("." + domain)
