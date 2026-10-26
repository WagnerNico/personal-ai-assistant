"""Extract bounded tables, structured facts and links without running scripts."""

import json
from html.parser import HTMLParser
from urllib.parse import urljoin


class PageDetails(HTMLParser):
    """Preserve data that a prose-only HTML extraction would otherwise lose."""

    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.facts = []
        self.links = []
        self.hidden = []
        self.script = None
        self.script_size = 0
        self.anchor = None
        self.cell = None
        self.row = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and len(self.facts) < 80:
            key, value = attrs.get("property") or attrs.get("name"), attrs.get("content")
            if key and value:
                self.facts.append(f"{key[:100]}: {value[:400]}")
        if tag == "script":
            if attrs.get("type", "").casefold() in {"application/ld+json", "application/json"}:
                self.script, self.script_size = [], 0
            self.hidden.append(tag)
        elif tag in {"style", "noscript", "nav", "footer", "form"}:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == "a" and attrs.get("href") and len(self.links) < 24:
            self.anchor = (urljoin(self.url, attrs["href"]), [])
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.script is not None:
            self.script_size += len(data)
            if self.script_size <= 65536:
                self.script.append(data)
        if self.hidden:
            return
        if self.anchor is not None:
            self.anchor[1].append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            if self.script_size <= 65536:
                try:
                    self._flatten(json.loads("".join(self.script)))
                except (ValueError, RecursionError):
                    pass
            self.script = None
        if tag in self.hidden:
            self.hidden = self.hidden[:len(self.hidden) - 1 - self.hidden[::-1].index(tag)]
        if tag == "a" and self.anchor is not None:
            url, parts = self.anchor
            self.links.append((" ".join(" ".join(parts).split())[:160], url))
            self.anchor = None
        elif tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split())[:250])
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row and len(self.facts) < 80:
                self.facts.append(" | ".join(self.row)[:1200])
            self.row = None

    def _flatten(self, value, path="data", depth=0):
        if depth > 8 or len(self.facts) >= 80:
            return
        if isinstance(value, dict):
            for key, item in list(value.items())[:80]:
                self._flatten(item, f"{path}.{str(key)[:80]}", depth + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value[:12]):
                self._flatten(item, f"{path}[{index}]", depth + 1)
        elif isinstance(value, (str, int, float, bool)):
            self.facts.append(f"{path}: {str(value)[:400]}")

    def metadata(self):
        # Whole rows/fields are retained; never cut a price halfway through.
        result = []
        remaining = 6000
        for fact in self.facts:
            if len(fact) + 1 <= remaining:
                result.append(fact)
                remaining -= len(fact) + 1
        return "\n".join(result)


def relevant_excerpt(text, query, *, limit=3000):
    """Keep context around matching facts when a page exceeds the model budget."""
    if len(text) <= limit:
        return text
    import re
    words = set(re.findall(r"\w{3,}", query.casefold())) - {
        "the", "what", "current", "latest", "give", "please", "source", "using", "search", "price",
        "der", "die", "das", "ist", "was", "aktuelle", "preis", "quelle"}
    windows = [(index, text[index:index + 900]) for index in range(0, len(text), 750)]
    ranked = sorted(windows, key=lambda item: sum(word in item[1].casefold() for word in words), reverse=True)
    selected = sorted(ranked[:3], key=lambda item: item[0])
    return "\n[...]\n".join(part for _, part in selected)[:limit]
