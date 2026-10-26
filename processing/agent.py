"""Route explicit app requests through bounded tools before ordinary chat."""

import json
from pathlib import Path
import re

from actions.app_control import AppControl, AppRequest, AppResult, normalize_name
from .text_processing import process_text, validate_text_request
from .web_processing import answer_from_web, web_request
from .web_planning import plan_web_request
from .spotify_requests import spotify_request
from actions.spotify import SpotifyRequest
from actions.email_reading import EmailRequest
from .email_requests import email_request
from .email_processing import answer_from_email


_COMMAND_PREFIX = (
    r"^(?:airis,?\s+)?(?:(?:please|can you|could you|would you|i want you to|i want to|"
    r"i would like you to|i would like to|i'd like you to|i'd like to|"
    r"bitte|kannst du|könntest du)\s+)*(?:please\s+)?")
_APP_COMMAND = re.compile(
    _COMMAND_PREFIX +
    r"(?P<verb>open|launch|start|bring up|öffne|oeffne|starte|"
    r"switch(?:\s+(?:over|back))?\s+to|go(?:\s+back)?\s+to|focus(?:\s+on)?|"
    r"wechsle\s+zu|wechsel\s+zu|gehe\s+zu)\s+(?P<app>.+)$", re.I | re.S)
_STANDALONE_FULL_SCREEN = re.compile(
    _COMMAND_PREFIX +
    r"(?:make|put|set|show|switch|mache|mach)\s+(?P<app>.+?)\s+"
    r"(?:(?:to|in|into|im)\s+)?(?:full[ -]?screen|vollbild(?:modus)?)$", re.I | re.S)
_FULL_SCREEN = re.compile(
    r"(?:\s*,?\s+(?:and\s+)?(?:make|put|switch|show)(?:\s+it)?(?:\s+to|\s+in)?\s+"
    r"(?:full[ -]?screen|fullscreen)|\s+in\s+(?:full[ -]?screen|fullscreen)|"
    r"\s+im\s+vollbild(?:modus)?|\s+und\s+mach(?:e)?\s+(?:es\s+)?vollbild)\s*$", re.I)
_EXTRA_ACTION = re.compile(r"\b(?:and|then|und|danach)\b|[;\n&]", re.I)
_PROMPT = Path(__file__).with_name("app_action_prompt.txt")


def _mentioned(text, alias):
    return f" {alias} " in f" {normalize_name(text)} "


def _app_request(text, control, backend):
    """Use exact commands first; use the model only to interpret varied wording."""
    command = text.strip().rstrip(".!?")
    command = re.sub(r"(?:,?\s+(?:please|for me|bitte))+$", "", command, flags=re.I)
    match = _APP_COMMAND.fullmatch(command)
    standalone = _STANDALONE_FULL_SCREEN.fullmatch(command) if match is None else None
    match = match or standalone
    if match is None:
        return None
    action = ("full_screen_app" if standalone else
              "open_app" if match["verb"].casefold() in
              ("open", "launch", "start", "bring up", "öffne", "oeffne", "starte") else "switch_app")
    body = re.sub(r"(?:,?\s+(?:please|for me|bitte))+$", "", match["app"], flags=re.I)
    full_screen = bool(standalone or _FULL_SCREEN.search(body))
    body = _FULL_SCREEN.sub("", body).strip()
    # Compound actions stay outside this deliberately small tool interface.
    if _EXTRA_ACTION.search(body):
        return AppResult("I can open, switch to, or make one app full screen. Please request those actions only.")
    body = re.sub(r"^(?:the|my)\s+", "", body, flags=re.I)
    body = re.sub(r"^(?:app|application)\s+", "", body, flags=re.I)
    body = re.sub(r"\s+(?:app|application)$", "", body, flags=re.I)
    name = body.strip()
    if (not name or len(name) > 160 or any(char in name for char in '/\\:$`<>\"\'')
            or any(ord(char) < 32 for char in name)):
        return AppResult("Please name an installed app, such as Outlook or Spotify.")
    apps = control.apps()
    normalized = normalize_name(name)
    if any(normalized in app.aliases for app in apps):
        return AppRequest(name, full_screen, action)
    # The planner sees only apps explicitly named in the current request.
    candidates = [app for app in apps if any(_mentioned(name, alias) for alias in app.aliases)]
    if not candidates:
        return AppRequest(name, full_screen, action)  # The tool reports an unavailable app.
    prompt = _PROMPT.read_text(encoding="utf-8")
    prompt += "\nInstalled app names: " + json.dumps([app.name for app in candidates])
    prompt += "\nRequested action: " + action
    try:
        raw = process_text(text, backend=backend, system_prompt=prompt, max_new_tokens=128)
        plan = json.loads(raw)
        if (not isinstance(plan, dict) or set(plan) != {"action", "app", "full_screen"}
                or plan["action"] != action or not isinstance(plan["app"], str)
                or type(plan["full_screen"]) is not bool):
            raise ValueError("Invalid app tool selection.")
        selected = normalize_name(plan["app"])
        if not any(selected in app.aliases for app in candidates):
            raise ValueError("The selected app was not named in the request.")
        if plan["full_screen"] and not re.search(r"full[ -]?screen|vollbild", text, re.I):
            raise ValueError("Full screen was not requested.")
        if full_screen and not plan["full_screen"]:
            raise ValueError("The plan omitted the requested full screen.")
        return AppRequest(plan["app"], plan["full_screen"], action)
    except (ValueError, RuntimeError, TypeError):
        return AppResult("I couldn't identify that app request. Try 'open Outlook', 'switch to Spotify', or 'make Spotify full screen'.")


def process_agent_text(text, *, app_control: AppControl | None, backend=None,
                       history=None, system_prompt=None, max_new_tokens=512,
                       on_text=None, on_stage=None, web_access=None, web_enabled=True,
                       mode="text", on_sources=None, spotify_control=None, on_music_result=None,
                       email_reader=None, email_enabled=True, web_history=None):
    """Keep chat streaming unchanged and return receipts only after tool results.

    Tool plans are private: they never enter TTS or conversation history. A
    future agent can call AppControl.execute(AppRequest(...)) directly while
    retaining the same platform and installed-app checks.
    """
    validate_text_request(text, max_new_tokens)
    mail_request = email_request(text)
    if mail_request is not None:
        if isinstance(mail_request, str):
            reply = mail_request
        elif not email_enabled or email_reader is None:
            reply = "Email reading is turned off. Enable it in Settings to read unread emails."
        else:
            return answer_from_email(mail_request, email_reader, question=text, backend=backend,
                                     system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                                     on_text=on_text, on_stage=on_stage)
        if on_text:
            on_text(reply)
        return reply, False
    retrieval_history = history if web_history is None else web_history
    if spotify_control is not None:
        music_request = spotify_request(text, prefix=_COMMAND_PREFIX)
        if music_request is not None:
            result = (spotify_control.execute(music_request, web_access=web_access,
                                               web_enabled=web_enabled, on_stage=on_stage)
                      if isinstance(music_request, SpotifyRequest) else music_request)
            if result.applied and on_music_result is not None:
                on_music_result(result)
            if on_text:
                on_text(result.text)
            return result.text, False
    request = web_request(text) if web_access is not None else None
    # Explicit URLs are read as pages before the app-command grammar sees them.
    if request is not None and "url" in request:
        if not web_enabled:
            reply = "Web access is turned off. Enable web access in Settings to read pages."
            if on_text:
                on_text(reply)
            return reply, False
        return answer_from_web(text, request, web_access=web_access, backend=backend,
                               history=retrieval_history, system_prompt=system_prompt,
                               max_new_tokens=max_new_tokens, mode=mode, on_text=on_text,
                               on_stage=on_stage, on_sources=on_sources)
    if app_control is not None:
        app_request = _app_request(text, app_control, backend)
        if app_request is not None:
            result = (app_control.execute(app_request, on_stage=on_stage)
                      if isinstance(app_request, AppRequest) else app_request)
            if on_text is not None:
                on_text(result.text)
            return result.text, False
    plan = (plan_web_request(text, backend=backend, history=retrieval_history, enabled=web_enabled, on_stage=on_stage)
            if web_access is not None else None)
    if plan is not None and plan.clarification:
        if on_text:
            on_text(plan.clarification)
        return plan.clarification, False
    request = plan.request if plan is not None else request
    if request is not None:
        if not web_enabled:
            reply = "Web access is turned off. Enable web access in Settings to search online."
            if on_text:
                on_text(reply)
            return reply, False
        return answer_from_web(text, request, web_access=web_access, backend=backend,
                               history=retrieval_history, system_prompt=system_prompt,
                               max_new_tokens=max_new_tokens, mode=mode, on_text=on_text,
                               on_stage=on_stage, on_sources=on_sources,
                               assumption=plan.assumption if plan else "")
    response = process_text(text, backend=backend, history=history,
                            system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                            on_text=on_text)
    return response, getattr(backend, "last_finish_reason", None) == "length"
