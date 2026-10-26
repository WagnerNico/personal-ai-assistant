"""Verify adaptive research, generic facts and private bounded tool decisions."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from actions.web import PublicPageReader, WebAccess, WebEvidence, WebError, WebLink, WebSource
from actions.web_content import PageDetails
from actions.web_http import browser_fetch
from assistant import Assistant
from processing.web_research import research_web


def decision(action="answer", query="", url=""):
    return json.dumps(dict(action=action, query=query, url=url))


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.backend, self.web = Mock(), Mock()
        self.overview = WebSource("Overview", "https://example.org/overview", text="Company background.", page_read=True,
                                  links=(WebLink("Details", "https://example.org/details"),))
        self.details = WebSource("Details", "https://example.org/details", text="Observed value: 123.45 USD on 2026-10-10.", page_read=True)

    def test_missing_fact_triggers_read_of_a_discovered_detail_page(self):
        self.web.gather.side_effect = [WebEvidence((self.overview,)), WebEvidence((self.details,))]
        self.backend.respond.side_effect = [decision("read", url=self.details.url), decision()]
        result = research_web("Current value?", {"query": "current value", "source": "example.org"},
                              backend=self.backend, web_access=self.web)
        self.assertIn(self.details, result.sources)
        self.web.gather.assert_called_with(url=self.details.url, source="example.org", on_stage=None)

    def test_failed_lookup_is_refined_without_changing_source(self):
        self.web.gather.side_effect = [WebEvidence(error="No usable results."), WebEvidence((self.details,))]
        self.backend.respond.side_effect = [decision("search", query="short precise topic"), decision()]
        result = research_web("Find facts", {"query": "long unclear topic", "source": "example.org"},
                              backend=self.backend, web_access=self.web)
        self.assertEqual(result.sources, (self.details,))
        self.web.gather.assert_called_with(query="short precise topic", kind="search", source="example.org", on_stage=None)

    def test_sufficient_evidence_stops_without_extra_network_work(self):
        self.web.gather.return_value = WebEvidence((self.details,))
        self.backend.respond.return_value = decision()
        result = research_web("Find facts", {"query": "facts"}, backend=self.backend, web_access=self.web)
        self.assertEqual(result.sources, (self.details,))
        self.web.gather.assert_called_once()

    def test_invalid_actions_unseen_urls_and_repeated_queries_do_not_execute(self):
        for raw in (decision("open_app", query="Spotify"), decision("read", url="http://127.0.0.1/"),
                    decision("read", url="https://unseen.org/"), decision("read", url=self.overview.url),
                    decision("search", query="FACTS"), decision("search", query="facts site:evil.org"), "bad JSON"):
            with self.subTest(raw=raw):
                self.web.reset_mock()
                self.web.gather.return_value = WebEvidence((self.overview,))
                self.backend.respond.return_value = raw
                research_web("Find facts", {"query": "facts"}, backend=self.backend, web_access=self.web)
                self.web.gather.assert_called_once()

    def test_steps_and_evidence_are_bounded(self):
        self.web.gather.return_value = WebEvidence((self.overview,))
        self.backend.respond.side_effect = [decision("search", query="first refinement"), decision("search", query="second refinement")]
        research_web("Find facts", {"query": "facts"}, backend=self.backend, web_access=self.web)
        self.assertEqual(self.web.gather.call_count, 3)
        self.assertEqual(self.backend.respond.call_count, 2)

    def test_time_budget_prevents_further_model_or_network_work(self):
        self.web.gather.return_value = WebEvidence((self.overview,))
        with patch("processing.web_research.time.monotonic", side_effect=[0, 41]):
            research_web("Find facts", {"query": "facts"}, backend=self.backend, web_access=self.web)
        self.backend.respond.assert_not_called()
        self.web.gather.assert_called_once()

    def test_new_page_body_upgrades_a_snippet_and_preserves_publication(self):
        snippet = WebSource("Details", self.details.url, snippet="Summary", published_at="2026-10-10")
        self.web.gather.side_effect = [WebEvidence((snippet,)), WebEvidence((self.details,))]
        self.backend.respond.side_effect = [decision("read", url=self.details.url), decision()]
        result = research_web("News details", {"query": "news"}, backend=self.backend, web_access=self.web)
        self.assertEqual(len(result.sources), 1)
        self.assertTrue(result.sources[0].page_read)
        self.assertEqual(result.sources[0].published_at, "2026-10-10")

    def test_only_final_answer_is_streamed_or_committed_to_history(self):
        self.web.gather.side_effect = [WebEvidence((self.overview,)), WebEvidence((self.details,))]
        self.backend.respond.side_effect = [decision("read", url=self.details.url), decision(), "The observation was 123.45 USD. [2]"]
        with Assistant(backend=self.backend, web_access=self.web, app_control=False) as assistant:
            chunks = []
            result = assistant.respond("Search for the observed value", on_text=chunks.append)
            self.assertEqual("".join(chunks), result.text)
            self.assertNotIn('"action"', str(assistant.history))
            self.assertIn("123.45 USD", result.text)
            final_prompt = self.backend.respond.call_args.args[0][0]["content"]
            self.assertIn("123.45 USD", final_prompt)
            self.assertNotIn('"links"', final_prompt)


class GenericContentTests(unittest.TestCase):
    def test_tables_metadata_and_json_are_data_without_script_execution(self):
        html = ('<head><meta property="og:description" content="Observed measurements">'
                '<script type="application/ld+json">{"name":"Example","value":123.45,"unit":"USD"}</script>'
                '<script>openSpotify();</script></head><body><table>'
                '<tr><th>Place</th><th>Value</th><th>Time</th></tr>'
                '<tr><td>Example</td><td>123.45 USD</td><td>12:34</td></tr></table>'
                '<a href="/details">More data</a><nav><a href="/menu">Menu</a></nav></body>')
        details = PageDetails("https://example.org/")
        details.feed(html)
        self.assertIn("Place | Value | Time", details.metadata())
        self.assertIn("123.45 USD | 12:34", details.metadata())
        self.assertIn("data.value: 123.45", details.metadata())
        self.assertNotIn("openSpotify", details.metadata())
        self.assertEqual(details.links, [("More data", "https://example.org/details")])

    def test_reader_returns_discoverable_public_links_and_filters_private_links(self):
        html = b'<title>Example</title><p>Facts.</p><a href="/detail">Details</a><a href="http://127.0.0.1">Private</a>'
        reader = PublicPageReader()
        reader.fetch = Mock(return_value=("https://example.org/", html, "text/html"))
        source = reader.read("https://example.org/")
        self.assertEqual(source.links, (WebLink("Details", "https://example.org/detail"),))

    def test_json_page_is_available_to_the_same_reader(self):
        reader = PublicPageReader()
        reader.fetch = Mock(return_value=("https://example.org/data", b'{"value":12.5,"unit":"C"}', "application/json"))
        source = reader.read("https://example.org/data")
        self.assertTrue(source.page_read)
        self.assertIn("12.5", source.text)


class BrowserReaderTests(unittest.TestCase):
    def setUp(self):
        self.factory = Mock()
        self.session = self.factory.return_value.__enter__ = Mock(return_value=Mock())
        self.factory.return_value.__exit__ = Mock(return_value=False)
        self.client = self.session.return_value
        self.response = Mock(status_code=200, headers={"content-type": "text/html"})
        self.options = SimpleNamespace(RESOLVE="resolve", PROXY="proxy", TIMEOUT_MS="timeout")
        self.module = patch.dict("sys.modules", curl_cffi=SimpleNamespace(CurlOpt=self.options, requests=SimpleNamespace(Session=self.factory)))
        self.module.start()
        self.addCleanup(self.module.stop)
        self.dns = patch("actions.web.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))])
        self.dns.start()
        self.addCleanup(self.dns.stop)
        self.client.get.side_effect = self.get

    def get(self, url, **kwargs):
        kwargs["content_callback"](b"<p>A fact.</p>")
        return self.response

    def fetch(self, url="https://example.org/page", **kwargs):
        return browser_fetch(url, timeout=6, max_bytes=1000, content_types=("text/html",), **kwargs)

    def test_browser_get_preserves_ip_pin_tls_and_cookie_boundaries(self):
        self.assertIn(b"A fact", self.fetch()[1])
        args = self.factory.call_args.kwargs
        self.assertEqual(args["curl_options"]["resolve"], ["example.org:443:8.8.8.8"])
        self.assertEqual(args["curl_options"]["proxy"], "")
        self.assertTrue(args["verify"])
        self.assertFalse(args["trust_env"])
        self.assertTrue(args["discard_cookies"])
        self.assertFalse(self.client.get.call_args.kwargs["allow_redirects"])
        self.response.close.assert_called_once()

    def test_private_and_foreign_source_redirects_never_receive_a_get(self):
        for destination in ("http://127.0.0.1/", "https://other.org/"):
            with self.subTest(destination=destination):
                self.client.get.reset_mock()
                self.response.status_code = 302
                self.response.headers = {"location": destination}
                with self.assertRaises(WebError):
                    self.fetch(source_domain="example.org")
                self.client.get.assert_called_once()

    def test_size_limit_is_enforced_while_receiving_body(self):
        def huge(url, **kwargs):
            self.assertEqual(kwargs["content_callback"](b"x" * 1001), 0)
            raise RuntimeError("Callback aborted transfer")
        self.client.get.side_effect = huge
        with self.assertRaisesRegex(WebError, "too large"):
            self.fetch()

    def test_403_uses_the_general_fallback_without_provider_special_cases(self):
        reader = PublicPageReader()
        reader.fetch = Mock(side_effect=WebError("Blocked", code="http_403"))
        with patch("actions.web_http.browser_fetch", return_value=("https://example.org/page", b"<p>A fact.</p>", "text/html")) as fetch:
            source = reader.read("https://example.org/page")
        self.assertTrue(source.page_read)
        fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
