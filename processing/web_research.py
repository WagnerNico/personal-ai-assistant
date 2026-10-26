"""Gather evidence through bounded, private search/read/coverage decisions."""

from dataclasses import replace
import json
from pathlib import Path
import re
import time

from actions.web import WebEvidence
from actions.web_content import relevant_excerpt
from .text_processing import process_text


MAX_FOLLOWUP_STEPS = 2
MAX_SOURCES = 8
RESEARCH_SECONDS = 40


def research_web(text, request, *, web_access, backend, on_stage=None):
    """Use the same tools for every topic; never expose private plans to speech."""
    started = time.monotonic()
    evidence = web_access.gather(**request, on_stage=on_stage)
    if backend is not None and not callable(getattr(backend, "respond", None)):
        return evidence
    sources = list(evidence.sources)[:MAX_SOURCES]
    queries = [request["query"]] if "query" in request else []
    read_urls = {source.url for source in sources if source.page_read}
    attempted_urls = {request["url"]} if "url" in request else set()
    goal = request.get("query", text)
    for step in range(MAX_FOLLOWUP_STEPS):
        if time.monotonic() - started >= RESEARCH_SECONDS:
            break
        if on_stage:
            on_stage("Checking whether the sources answer your question...")
        allowed = {source.url for source in sources if len(source.url) <= 500}
        allowed.update(link.url for source in sources for link in source.links if len(link.url) <= 500)
        records = []
        for source in sources:
            records.append(dict(title=source.title[:200], url=source.url if source.url in allowed else "",
                                snippet=source.snippet[:500], text=relevant_excerpt(source.text, goal, limit=1600),
                                metadata=relevant_excerpt(source.metadata, goal, limit=1800),
                                page_read=source.page_read, published_at=source.published_at,
                                links=[dict(title=link.title, url=link.url) for link in source.links if link.url in allowed]))
        data = dict(request=text, search_goal=goal, source_constraint=request.get("source", ""),
                    evidence=records, retrieval_error=evidence.error,
                    allowed_urls=sorted(allowed - read_urls - attempted_urls),
                    read_urls=sorted(read_urls | attempted_urls), previous_queries=queries,
                    remaining_steps=MAX_FOLLOWUP_STEPS - step)
        try:
            raw = process_text(json.dumps(data, ensure_ascii=True), backend=backend,
                               system_prompt=Path(__file__).with_name("web_research_prompt.txt").read_text(encoding="utf-8"),
                               max_new_tokens=192)
            plan = json.loads(raw)
            if (not isinstance(plan, dict) or set(plan) != {"action", "query", "url"}
                    or any(not isinstance(value, str) for value in plan.values())):
                break
            if plan["action"] == "answer":
                break
            next_request = {"source": request["source"]} if request.get("source") else {}
            if plan["action"] == "search":
                query = plan["query"].strip()
                normalized = " ".join(query.casefold().split())
                if (plan["url"] or not query or len(query) > 500 or re.search(r"\bsite:", query, re.I)
                        or normalized in {" ".join(previous.casefold().split()) for previous in queries}):
                    break
                next_request.update(query=query, kind=request.get("kind", "search"))
                queries.append(query)
                if on_stage:
                    on_stage("Searching for the missing information...")
            elif plan["action"] == "read":
                url = plan["url"]
                if plan["query"] or url not in allowed or url in read_urls or url in attempted_urls:
                    break
                next_request["url"] = url
                attempted_urls.add(url)
                if on_stage:
                    on_stage("Reading a relevant detail page...")
            else:
                break
            if time.monotonic() - started >= RESEARCH_SECONDS:
                break
            evidence = web_access.gather(**next_request, on_stage=on_stage)
            for source in evidence.sources:
                if source.page_read:
                    read_urls.add(source.url)
                index = next((i for i, previous in enumerate(sources) if previous.url == source.url), None)
                if index is not None:
                    previous = sources[index]
                    if source.page_read or not previous.page_read:
                        sources[index] = replace(source, published_at=source.published_at or previous.published_at,
                                                 publisher=source.publisher or previous.publisher,
                                                 snippet=source.snippet or previous.snippet)
                else:
                    if len(sources) == MAX_SOURCES:
                        weak = next((i for i, previous in enumerate(sources) if not previous.page_read), 0)
                        sources.pop(weak)
                    sources.append(source)
        except (ValueError, TypeError, RuntimeError):
            break
    return WebEvidence(tuple(sources), error=evidence.error, error_code=evidence.error_code)
