"""Opt-in live checks of general public search and structured page reading."""

import json
import os
from pathlib import Path
import unittest

from actions.web import WebAccess
from assistant import Assistant


class EvidenceBackend:
    def respond(self, messages, *, max_new_tokens):
        self.messages = messages
        if '"action":"answer|search|read"' in messages[0]["content"]:
            return '{"action":"answer","query":"","url":""}'
        return "The requested facts are in the retrieved page. [1]"


@unittest.skipUnless(os.environ.get("AIRIS_RUN_WEB_TESTS") == "1",
                     "Set AIRIS_RUN_WEB_TESTS=1 to access public websites.")
class WebRuntimeTests(unittest.TestCase):
    def test_live_general_source_preference(self):
        backend = EvidenceBackend()
        with Assistant(backend=backend, app_control=False) as assistant:
            reply = assistant.respond("Explain photosynthesis, use en.wikipedia.org as source")
        self.assertTrue(reply.sources, reply.text)
        self.assertTrue(all(s.url.startswith("https://en.wikipedia.org/") for s in reply.sources))
        self.assertTrue(any(s.page_read for s in reply.sources))
        self.assertIn("specific source", backend.messages[0]["content"])

    def test_live_general_reader_preserves_a_public_price_table(self):
        evidence = WebAccess().gather(url="https://www.finanzen.net/boersenplaetze/amazon", source="finanzen.net")
        self.assertTrue(evidence.sources, evidence.error)
        source = evidence.sources[0]
        self.assertTrue(source.page_read)
        self.assertIn("amazon", source.title.casefold())
        self.assertIn("Nasdaq", source.metadata)
        self.assertRegex(source.metadata, r"\d+[,.]\d+\s+USD")
        self.assertRegex(source.metadata, r"\d{2}[:.]\d{2}")
        self.assertTrue(source.links)

    def test_live_dated_news(self):
        web = WebAccess()
        for query in ("science", "world politics"):
            with self.subTest(query=query):
                news = web.gather(query=query, kind="news")
                self.assertTrue(news.sources, news.error)
                self.assertTrue(all(source.published_at and source.snippet for source in news.sources))

    def test_live_search_and_public_page_reading(self):
        web = WebAccess()
        page = web.gather(url="https://docs.python.org/3/library/urllib.parse.html")
        self.assertEqual(page.error, "")
        self.assertTrue(page.sources[0].page_read)
        self.assertIn("urllib.parse", page.sources[0].title)
        backend = EvidenceBackend()
        with Assistant(backend=backend, web_access=web, app_control=False) as assistant:
            reply = assistant.respond("Search online for Python urllib.parse official documentation")
        self.assertTrue(reply.sources, reply.text)
        self.assertIn("BEGIN UNTRUSTED WEB EVIDENCE", backend.messages[0]["content"])
        report = dict(direct_page_title=page.sources[0].title,
                      search_sources=[dict(title=s.title, url=s.url, page_read=s.page_read) for s in reply.sources])
        Path("logs/web_runtime_check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
