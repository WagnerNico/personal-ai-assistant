"""Opt-in request interpretation checks using cached 9B weights without audio."""

import fcntl
import json
import os
from pathlib import Path
import time
import unittest


@unittest.skipUnless(os.environ.get("AIRIS_RUN_WEB_MODEL_TESTS") == "1",
                     "Set AIRIS_RUN_WEB_MODEL_TESTS=1 after quitting Airis.")
class WebPlanningModelRuntimeTests(unittest.TestCase):
    def test_general_requests_with_the_actual_model(self):
        for path in Path("logs/runtime").glob("airis-*.lock"):
            if path.name.startswith("airis-test-"):
                continue
            with path.open("r") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.fail("Quit Airis before loading a second model for this check.")
        from processing.mlx_model import load_mlx_backend
        from processing.web_planning import plan_web_request
        backend = load_mlx_backend()
        backend.temperature = 0.0
        report = []
        cases = [
            ("What is the weather today?", [], "clarify", ()),
            ("What is the weather in Berlin today?", [], "search", ("berlin",)),
            ("And tomorrow?", [{"role": "user", "content": "Weather in Berlin, use example.org as source"}], "search", ("berlin",)),
            ("Berlin", [{"role": "user", "content": "What is the weather today?"},
                        {"role": "assistant", "content": "Which city should I check the weather for?"}], "search", ("berlin",)),
            ("Which laptop would you recommend?", [], "search", ("laptop",)),
            ("Latest science news from Nature", [], "news", ("science",)),
            ("Latest science news from Germany", [], "news", ("germany",)),
            ("What is photosynthesis?", [], "local", ()),
        ]
        for text, history, expected, terms in cases:
            with self.subTest(text=text):
                started = time.monotonic()
                print(f"Planning: {text}", flush=True)
                result = plan_web_request(text, backend=backend, history=history)
                actual = "clarify" if result.clarification else result.request.get("kind", "search") if result.request else "local"
                report.append(dict(text=text, request=result.request, clarification=result.clarification,
                                   assumption=result.assumption, elapsed_seconds=round(time.monotonic() - started, 2)))
                print(json.dumps(report[-1]), flush=True)
                self.assertEqual(actual, expected)
                for term in terms:
                    self.assertIn(term, result.request["query"].casefold())
                if text == "And tomorrow?":
                    self.assertEqual(result.request["source"], "example.org")
                if text.endswith("from Nature"):
                    self.assertEqual(result.request["source"].casefold(), "nature")
                if text.endswith("from Germany"):
                    self.assertNotIn("source", result.request)
        Path("logs/web_planning_model_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
