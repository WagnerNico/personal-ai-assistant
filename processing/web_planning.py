"""Resolve conversational information requests before bounded web retrieval."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

from actions.web_sources import source_preference
from .text_processing import process_text
from .web_processing import web_request


_QUESTION = re.compile(r"^(?:(?:please|can you|could you|would you|bitte|kannst du|könntest du)\s+)*"
                       r"(?:what|who|where|when|which|how|i need|i would like|ich brauche|ich möchte|"
                       r"recommend|find|tell me|give me|show me|was|wer|wo|wann|welche\w*|"
                       r"wie viel|empfiehl\w*|sag mir|gib mir|zeige mir)\b", re.I)
_PERSONAL = re.compile(r"^(?:what is (?:my|your) name|who (?:am i|are you)|how are you|"
                       r"wie (?:heiße ich|heißt du)|wer (?:bin ich|bist du))\b", re.I)
_FOLLOWUP = re.compile(r"^(?:and\b|what about\b|how about\b|und\b|was ist mit\b|"
                       r"wie sieht es mit\b|from that source\b|use the same source\b)", re.I)
_WEATHER = re.compile(r"\b(?:weather|wetter)\b", re.I)
_SOURCE_RELATION = re.compile(r"\b(?:from|according to|laut|von)\s+\S+", re.I)
_OFFLINE = re.compile(r"\b(?:don't|do not|without|no|nicht|ohne)\s+(?:(?:use|using|the|das|die)\s+)*"
                      r"(?:web|internet|online|search|browsing|suche)\b", re.I)


@dataclass(frozen=True)
class WebPlan:
    request: dict | None = None
    clarification: str = ""
    assumption: str = ""


def plan_web_request(text, *, backend, history=None, enabled=True, on_stage=None):
    """Use fast paths for clear requests and the local model for missing context.

    Plans never reach speech or external providers. Only a standalone query is
    sent out; source constraints come from user messages, not model output.
    """
    request = web_request(text)
    if _OFFLINE.search(text) or not enabled:
        return WebPlan(request)
    users = [message["content"][:1500] for message in (history or [])
             if message.get("role") == "user"][-4:]
    followup = bool(users and _FOLLOWUP.match(text.strip()))
    last = (history or [{}])[-1]
    clarification = last.get("content", "") if last.get("role") == "assistant" else ""
    # Include only our bounded question, never a source-bearing old answer.
    clarification = clarification if clarification.endswith("?") and len(clarification) <= 250 else ""
    context_reply = bool(clarification and users and request is None and not _QUESTION.match(text.strip()))
    preference = source_preference(text)
    relation = list(_SOURCE_RELATION.finditer(text))
    source_interpretation = bool(preference is None and request is not None and relation)
    needs_plan = (followup or context_reply or bool(_WEATHER.search(text)) or source_interpretation or
                  (preference is not None and not preference.query) or
                  (request is None and _QUESTION.match(text.strip()) and not _PERSONAL.search(text)))
    if not needs_plan:
        return WebPlan(request)
    # Stream-only integrations can keep using the deterministic router.
    if backend is not None and not callable(getattr(backend, "respond", None)):
        return WebPlan(request)
    if on_stage:
        on_stage("Understanding the information request...")
    try:
        raw = process_text(json.dumps({"request": text, "recent_user_messages": users,
                                      "clarification_question": clarification if context_reply else "",
                                      "current_time": datetime.now().astimezone().isoformat(timespec="seconds")}, ensure_ascii=True),
                           backend=backend,
                           system_prompt=Path(__file__).with_name("web_request_prompt.txt").read_text(encoding="utf-8"),
                           max_new_tokens=256)
        plan = json.loads(raw)
        keys = {"action", "query", "url", "source", "question", "assumption"}
        if not isinstance(plan, dict) or set(plan) != keys or any(not isinstance(v, str) for v in plan.values()):
            raise ValueError("Invalid information request plan.")
        action = plan["action"]
        if action not in {"local", "search", "news", "read", "clarify"}:
            raise ValueError("Unsupported information request action.")
        if len(plan["assumption"]) > 250 or len(plan["question"]) > 250 or len(plan["source"]) > 2048:
            raise ValueError("Information request text is too long.")
        if action == "local":
            if any(plan[key] for key in ("query", "url", "source", "question", "assumption")):
                raise ValueError("A local request cannot include a web operation.")
            # A planner cannot discard an explicit web request or source.
            return WebPlan(request)
        if action == "clarify":
            if not plan["question"].strip() or plan["query"] or plan["url"] or plan["assumption"] or plan["source"]:
                raise ValueError("Invalid clarification plan.")
            return WebPlan(clarification=plan["question"].strip())
        if plan["question"]:
            raise ValueError("Retrieval cannot include a clarification.")
        if action == "read":
            supplied = [url.rstrip(".,);!?") for url in re.findall(r"https?://[^\s<>\"']+", text, re.I)]
            if plan["query"] or not plan["url"] or plan["url"] not in supplied:
                raise ValueError("The page URL was not supplied by the user.")
            selected = {"url": plan["url"]}
        else:
            if plan["url"] or not plan["query"].strip() or len(plan["query"]) > 500 or re.search(r"\bsite:", plan["query"], re.I):
                raise ValueError("Invalid standalone search query.")
            selected = {"query": plan["query"].strip(), "kind": action}
        if preference is None and (followup or context_reply):
            # Follow-up chains inherit a source, but a new topic resets it.
            for previous in reversed(users):
                preference = source_preference(previous)
                if preference is not None or not _FOLLOWUP.match(previous.strip()):
                    break
        if preference is not None:
            selected["source"] = preference.target
        elif plan["source"]:
            source = plan["source"].strip()
            allowed = [text] + (users if followup or context_reply else [])
            if not source or not any(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", user, re.I) for user in allowed):
                raise ValueError("The selected source was not requested by the user.")
            selected["source"] = source
        return WebPlan(selected, assumption=plan["assumption"].strip())
    except (ValueError, TypeError, RuntimeError):
        if source_interpretation:
            return WebPlan(clarification="Which website should I use as the source?")
        if followup or context_reply or _WEATHER.search(text):
            # Do not send an unresolved pronoun or guessed personal place online.
            return WebPlan(clarification="Which city should I check the weather for?" if _WEATHER.search(text)
                           else "What topic should I look up?")
        return WebPlan(request)
