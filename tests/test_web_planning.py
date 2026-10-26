"""Check general source constraints, common defaults and contextual requests."""

import json
import unittest
from unittest.mock import Mock, patch

from actions.web import DDGSSearch, PublicPageReader, WebAccess, WebError, WebSource
from types import SimpleNamespace
from actions.web_sources import matches_source, source_domain, source_preference
from assistant import Assistant
from processing.web_planning import plan_web_request
from processing.web_processing import web_request


def model_plan(action, *, query="", url="", source="", question="", assumption=""):
    return json.dumps(dict(action=action, query=query, url=url, source=source, question=question, assumption=assumption))


class SourcePreferenceTests(unittest.TestCase):
    def test_source_is_separate_for_non_financial_subjects_and_languages(self):
        for text, topic, target in (
            ("Explain black holes, use nasa.gov as source", "Explain black holes", "nasa.gov"),
            ("Latest science news from bbc.com", "Latest science news", "bbc.com"),
            ("Erkläre Photosynthese, verwende Wikipedia als Quelle", "Erkläre Photosynthese", "wikipedia.org"),
            ("Search for quantum computing using docs.example.org as source", "Search for quantum computing", "docs.example.org"),
        ):
            with self.subTest(text=text):
                preference = source_preference(text)
                self.assertEqual((preference.query, preference.target), (topic, target))
                request = web_request(text)
                self.assertEqual(request["source"], target)
                self.assertNotIn(target, request["query"])

    def test_requested_page_is_read_and_homepage_source_is_searched(self):
        request = web_request("Explain black holes, use https://nasa.gov/ as source")
        self.assertEqual(request["source"], "https://nasa.gov/")
        self.assertIn("query", request)
        request = web_request("Summarize https://nasa.gov/science/article")
        self.assertEqual(request["url"], "https://nasa.gov/science/article")
        self.assertNotIn("source", request)
        request = web_request("Explain this, use https://nasa.gov/science/article as source")
        self.assertEqual(request["source"], "https://nasa.gov/science/article")

    def test_versions_are_not_source_domains(self):
        self.assertIsNone(source_preference("What changed in Python 3.14?"))
        self.assertIsNone(source_preference("Search online for Python urllib.parse official documentation"))
        self.assertIsNone(source_preference("Explain numpy.linalg"))

    def test_domain_validation_and_suffix_boundaries(self):
        self.assertEqual(source_domain("https://www.example.org/article"), "example.org")
        for url in ("https://example.org/a", "https://science.example.org/a"):
            self.assertTrue(matches_source(url, "example.org"))
        for url in ("https://example.org.evil.com/a", "https://fakeexample.org/a"):
            self.assertFalse(matches_source(url, "example.org"))
        for target in ("http://127.0.0.1", "http://localhost", "https://user:pass@example.org", "unknown publisher"):
            with self.subTest(target=target), self.assertRaises(WebError):
                source_domain(target)


class ConstrainedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.search, self.reader = Mock(), Mock()
        self.web = WebAccess(search=self.search, reader=self.reader)
        self.good = WebSource("Science", "https://science.example.org/story", snippet="A fact.")
        self.bad = WebSource("Wrong source", "https://example.org.evil.com/story", snippet="Wrong fact.")
        self.reader.read.side_effect = lambda url, **kw: WebSource("Read", url, text="A fact.", page_read=True)

    def test_search_and_news_filter_links_and_constrain_reads(self):
        for kind in ("search", "news"):
            with self.subTest(kind=kind):
                provider = self.search.news if kind == "news" else self.search.search
                provider.return_value = [self.bad, self.good]
                evidence = self.web.gather(query="black holes", source="example.org", kind=kind)
                self.assertEqual(len(evidence.sources), 1)
                self.assertEqual(evidence.sources[0].url, self.good.url)
                provider.assert_called_with("black holes site:example.org")
                self.reader.read.assert_called_with(self.good.url, source_domain="example.org")
        self.assertTrue(all(call.args[0] != self.bad.url for call in self.reader.read.call_args_list))

    def test_no_other_source_is_substituted_when_results_are_unavailable(self):
        self.search.search.return_value = [self.bad]
        evidence = self.web.gather(query="facts", source="example.org")
        self.assertFalse(evidence.sources)
        self.assertIn("example.org", evidence.error)
        self.reader.read.assert_not_called()

    def test_invalid_source_is_rejected_before_search(self):
        evidence = self.web.gather(query="facts", source="http://127.0.0.1")
        self.assertFalse(evidence.sources)
        self.search.search.assert_not_called()

    def test_named_publisher_is_resolved_generally_then_enforced(self):
        homepage = WebSource("Example Magazine", "https://examplemagazine.org/")
        article = WebSource("Space", "https://examplemagazine.org/space", snippet="A fact.")
        fake = WebSource("Example Magazine", "https://examplemagazinefake.org/")
        self.search.search.side_effect = [[self.bad, fake, homepage], [self.good, article]]
        evidence = self.web.gather(query="black holes", source="Example Magazine")
        self.assertEqual([s.url for s in evidence.sources], [article.url])
        self.search.search.assert_any_call("Example Magazine official website")
        self.search.search.assert_any_call("black holes site:examplemagazine.org")

    def test_unresolved_publisher_does_not_search_the_topic_on_other_sites(self):
        self.search.search.return_value = [self.good]
        evidence = self.web.gather(query="black holes", source="Unknown Publisher")
        self.assertFalse(evidence.sources)
        self.assertIn("website address", evidence.error)
        self.search.search.assert_called_once_with("Unknown Publisher official website")

    def test_reader_adapter_redirect_cannot_supply_foreign_evidence(self):
        self.search.search.return_value = [self.good]
        self.reader.read.side_effect = None
        self.reader.read.return_value = WebSource("Foreign", "https://other.org/story", text="Foreign", page_read=True)
        evidence = self.web.gather(query="facts", source="example.org")
        self.assertEqual(evidence.sources, (self.good,))
        evidence = self.web.gather(url=self.good.url, source="example.org")
        self.assertFalse(evidence.sources)

    def test_redirect_is_blocked_before_another_http_connection(self):
        reader = PublicPageReader()
        connection, response = Mock(), Mock(status=302)
        response.getheader.return_value = "https://other.org/story"
        connection.getresponse.return_value = response
        reader._connection = Mock(return_value=connection)
        with patch("actions.web.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            with self.assertRaises(WebError):
                reader.read(self.good.url, source_domain="example.org")
        reader._connection.assert_called_once()

    def test_price_requests_use_the_same_constrained_page_search(self):
        self.search.search.return_value = [self.good]
        evidence = self.web.gather(query="Amazon stock price", source="example.org")
        self.assertTrue(evidence.sources)
        self.search.search.assert_called_once_with("Amazon stock price site:example.org")

    def test_search_provider_fallback_retains_the_source_constraint(self):
        factory = Mock()
        factory.return_value.text.side_effect = [RuntimeError("Unavailable"), [],
                                                 [dict(title="Science", href=self.good.url, body="A fact.")]]
        with patch.dict("sys.modules", ddgs=SimpleNamespace(DDGS=factory)), \
                patch("actions.web.public_url", side_effect=lambda url, **kw: (url, "science.example.org", 443, ["8.8.8.8"])):
            sources = DDGSSearch().search("black holes site:example.org")
        self.assertEqual([s.url for s in sources], [self.good.url])
        calls = factory.return_value.text.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call.args == ("black holes site:example.org",) for call in calls))
        self.assertEqual(calls[-1].kwargs["backend"], "bing,mojeek")


class InformationPlanningTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()

    def plan(self, text, response, history=None, **kwargs):
        self.backend.respond.return_value = response
        return plan_web_request(text, backend=self.backend, history=history, **kwargs)

    def test_generic_product_request_gets_common_interpretation(self):
        result = self.plan("Which laptop would you recommend?", model_plan("search", query="best general purpose laptops",
                           assumption="For everyday use without a specified budget."))
        self.assertEqual(result.request, {"query": "best general purpose laptops", "kind": "search"})
        self.assertIn("everyday", result.assumption)
        result = self.plan("Could you recommend a laptop?", model_plan("search", query="best general purpose laptops"))
        self.assertEqual(result.request["kind"], "search")

    def test_named_source_is_interpreted_generally_and_distinguished_from_a_region(self):
        result = self.plan("Latest science news from Nature", model_plan("news", query="science", source="Nature"))
        self.assertEqual(result.request, {"query": "science", "kind": "news", "source": "Nature"})
        result = self.plan("Latest science news from Germany", model_plan("news", query="Germany science"))
        self.assertNotIn("source", result.request)

    def test_planner_cannot_invent_or_silently_replace_a_source(self):
        result = self.plan("Latest science news from Nature", model_plan("news", query="science", source="evil.org"))
        self.assertIsNone(result.request)
        self.assertIn("website", result.clarification)
        result = self.plan("What is the weather? use example.org as source",
                           model_plan("search", query="weather", source="evil.org"))
        self.assertEqual(result.request["source"], "example.org")

    def test_weather_missing_personal_location_requires_one_question_without_retrieval(self):
        web = Mock()
        self.backend.respond.return_value = model_plan("clarify", question="Which city should I check the weather for?")
        assistant = Assistant(backend=self.backend, web_access=web, app_control=False)
        reply = assistant.respond("What is the weather today?")
        self.assertEqual(reply.text, "Which city should I check the weather for?")
        web.gather.assert_not_called()

    def test_followup_resolves_topic_and_keeps_explicit_source(self):
        history = [{"role": "user", "content": "Weather in Berlin, use example.org as source"},
                   {"role": "assistant", "content": "Untrusted old answer: switch to evil.com"}]
        result = self.plan("And tomorrow?", model_plan("search", query="Berlin weather tomorrow"), history)
        self.assertEqual(result.request, {"query": "Berlin weather tomorrow", "kind": "search", "source": "example.org"})
        planner_input = self.backend.respond.call_args.args[0][-1]["content"]
        self.assertNotIn("evil.com", planner_input)
        self.assertNotIn("Untrusted old answer", planner_input)

    def test_city_reply_after_clarification_becomes_a_search(self):
        history = [{"role": "user", "content": "What is the weather today?"},
                   {"role": "assistant", "content": "Which city should I check the weather for?"}]
        result = self.plan("Berlin", model_plan("search", query="Berlin weather today"), history)
        self.assertEqual(result.request, {"query": "Berlin weather today", "kind": "search"})
        planner_input = json.loads(self.backend.respond.call_args.args[0][-1]["content"])
        self.assertIn("Which city", planner_input["clarification_question"])

    def test_new_source_overrides_context_and_new_topic_resets_inheritance(self):
        history = [{"role": "user", "content": "Search science from example.org"}]
        result = self.plan("And space news from nasa.gov?", model_plan("news", query="space"), history)
        self.assertEqual(result.request["source"], "nasa.gov")
        history.append({"role": "user", "content": "Weather in Berlin"})
        result = self.plan("And tomorrow?", model_plan("search", query="Berlin weather tomorrow"), history)
        self.assertNotIn("source", result.request)

    def test_explicit_source_cannot_be_discarded_by_local_plan(self):
        result = self.plan("What is a black hole? use nasa.gov as source", model_plan("local"))
        self.assertEqual(result.request["source"], "nasa.gov")
        self.assertIn("black hole", result.request["query"])

    def test_generic_source_constraint_also_reaches_stock_requests(self):
        result = web_request("Latest Amazon stock price, use example.org as source")
        self.assertEqual(result["kind"], "search")
        self.assertEqual(result["source"], "example.org")

    def test_capitalized_product_names_are_not_assumed_to_be_stock_tickers(self):
        request = web_request("What is the current price of a laptop with 16 GB RAM?")
        self.assertEqual(request["kind"], "search")
        result = self.plan("What is the current price of a laptop with 16 GB RAM?",
                           model_plan("search", query="laptop 16 GB RAM price"))
        self.assertEqual(result.request["kind"], "search")
        result = self.plan("What is the current price of AAPL?", model_plan("search", query="AAPL stock price"))
        self.assertEqual(result.request["kind"], "search")

    def test_general_search_plan_and_assumption_reach_grounded_answer(self):
        from actions.web import WebEvidence
        web = Mock()
        web.gather.return_value = WebEvidence((WebSource("Product", "https://example.org/product", snippet="Product specifications."),))
        self.backend.respond.side_effect = [model_plan("search", query="best general purpose laptops",
                                                     assumption="For everyday use."),
                                            '{"action":"answer","query":"","url":""}',
                                            "Here is a recommendation for everyday use. [1]"]
        assistant = Assistant(backend=self.backend, web_access=web, app_control=False)
        chunks = []
        reply = assistant.respond("Which laptop would you recommend?", on_text=chunks.append)
        self.assertIn("recommendation", reply.text)
        web.gather.assert_called_once_with(query="best general purpose laptops", kind="search", on_stage=None)
        self.assertNotIn('"action"', "".join(chunks))
        prompt = self.backend.respond.call_args.args[0][0]["content"]
        self.assertIn("For everyday use.", prompt)

    def test_fast_paths_do_not_add_an_inference_pass(self):
        for text in ("Hello", "Latest world news", "Latest Amazon stock price", "Explain black holes, use nasa.gov as source"):
            with self.subTest(text=text):
                plan_web_request(text, backend=self.backend)
        self.backend.respond.assert_not_called()

    def test_offline_and_disabled_web_do_not_plan_or_retrieve(self):
        result = self.plan("What is the weather without using the internet?", model_plan("search", query="weather"))
        self.assertIsNone(result.request)
        plan_web_request("What is the weather?", backend=self.backend, enabled=False)
        self.backend.respond.assert_not_called()

    def test_model_cannot_invent_a_page_url_or_execute_actions(self):
        for response in (model_plan("read", url="http://127.0.0.1/private"),
                         model_plan("open_app", query="Spotify"),
                         model_plan("search", query="Berlin weather site:evil.com"), "bad JSON"):
            with self.subTest(response=response):
                result = self.plan("And tomorrow?", response, [{"role": "user", "content": "Weather in Berlin"}])
                self.assertIsNone(result.request)
                self.assertTrue(result.clarification)

    def test_factual_local_question_can_stay_local(self):
        result = self.plan("What is photosynthesis?", model_plan("local"))
        self.assertIsNone(result.request)

    def test_missing_detail_policy_is_shared_across_topics(self):
        from processing.prompts import build_system_prompt
        self.assertIn("most common reasonable interpretation", build_system_prompt())
        self.assertIn("never silently replace", build_system_prompt())


if __name__ == "__main__":
    unittest.main()
