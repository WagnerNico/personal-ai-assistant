"""Opt-in web answer checks with the cached 9B model and no audio."""

import json
import os
from pathlib import Path
import re
import subprocess
import time
import unittest


@unittest.skipUnless(os.environ.get("AIRIS_RUN_WEB_MODEL_TESTS") == "1",
                     "Set AIRIS_RUN_WEB_MODEL_TESTS=1 after quitting Airis.")
class WebModelRuntimeTests(unittest.TestCase):
    def test_general_research_with_the_full_model(self):
        processes = subprocess.check_output(["ps", "-A", "-o", "command="], text=True)
        if any(" -m ui.desktop" in line or " -m output.mlx_speech_worker" in line
               for line in processes.splitlines()):
            self.fail("Quit Airis before loading the full model for this check.")
        from assistant import Assistant
        from processing.mlx_model import load_mlx_backend
        print("Loading the cached Qwen3.5 9B model for silent general research checks...", flush=True)
        backend = load_mlx_backend()
        backend.temperature = 0.0
        report = []
        with Assistant(backend=backend, app_control=False) as assistant:
            for query in ("What is the latest stock price of Amazon, use finanzen.net as source",
                          "Explain photosynthesis, use en.wikipedia.org as source",
                          "Give me the latest science news"):
                started = time.monotonic()
                print(f"Checking: {query}", flush=True)
                reply = assistant.respond(query, mode="text", max_new_tokens=512,
                                          on_stage=lambda value: print(value, flush=True))
                report.append(dict(query=query, text=reply.text, elapsed_seconds=round(time.monotonic() - started, 2),
                                   sources=[dict(title=s.title, url=s.url, published_at=s.published_at,
                                                 publisher=s.publisher, page_read=s.page_read) for s in reply.sources]))
                print(reply.text, flush=True)
                self.assertTrue(reply.sources, reply.text)
                self.assertNotIn("Would you like me", reply.text)
                self.assertNotIn("Sources retrieved", reply.text)
                if "price" in query:
                    self.assertRegex(reply.text, r"\d+[.,]\d+")
                    self.assertRegex(reply.text, r"(?i)(USD|EUR|dollars?|euros?|\$|€)")
                    self.assertTrue(all("finanzen.net" in s.url for s in reply.sources))
                    self.assertNotRegex(reply.text, r"(?i)(couldn't retrieve|cannot provide|unable to provide)")
                elif "news" in query:
                    self.assertTrue(any(s.published_at for s in reply.sources))
                    self.assertRegex(reply.text, r"\[\d\]")
                else:
                    self.assertIn("light", reply.text.casefold())
                    self.assertTrue(all("en.wikipedia.org" in s.url for s in reply.sources))
        destination = Path(__file__).resolve().parents[1] / "logs/web_model_check.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
