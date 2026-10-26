"""Gmail Inbox reads with OAuth read-only scope and no message write methods."""

import base64
from datetime import datetime, timezone
from functools import partial
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
from threading import RLock

from .email_reading import EmailMessage, EmailRequest, EmailResult


READ_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_API = "https://gmail.googleapis.com/gmail/v1/users/me/"
_CONNECT = "Connect Gmail in Settings > Email (read only), then ask again."


class GmailConnectionError(RuntimeError):
    """A user-facing connection failure without credentials or server payloads."""


class MacOSMailTokenStore:
    """Use the native Keychain backend explicitly; never fall back to files."""

    def _backend(self):
        import sys
        if sys.platform != "darwin":
            raise GmailConnectionError("Gmail token storage currently requires macOS Keychain.")
        from keyring.backends.macOS import Keyring
        return Keyring()

    def load(self):
        return self._backend().get_password("Airis Gmail read-only", "account")

    def save(self, value):
        self._backend().set_password("Airis Gmail read-only", "account", value)

    def clear(self):
        from keyring.errors import PasswordDeleteError
        try:
            self._backend().delete_password("Airis Gmail read-only", "account")
        except PasswordDeleteError:
            pass


class _MailText(HTMLParser):
    """Extract visible text from HTML without rendering or loading resources."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        elif tag in ("br", "p", "div", "li") and not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _body(payload, snippet):
    plain, html = [], []
    pending = [payload]
    visited = 0
    while pending:
        part = pending.pop()
        visited += 1
        if not isinstance(part, dict) or visited > 1000:
            raise ValueError("Invalid message payload.")
        # Skip attachment parts, including attached text files.
        if part.get("filename"):
            continue
        pending.extend(reversed(part.get("parts", [])))
        body = part.get("body", {})
        data = body.get("data", "")
        if body.get("attachmentId") or not data:
            continue
        if part.get("mimeType") not in ("text/plain", "text/html"):
            continue
        if not isinstance(data, str) or len(data) > 300000:
            continue
        raw = base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_", validate=True)
        text = raw.decode("utf-8", errors="replace")
        if part["mimeType"] == "text/plain":
            plain.append(text)
        else:
            parser = _MailText()
            parser.feed(text)
            html.append(" ".join("".join(parser.parts).split()))
    content = "\n".join(plain or html)
    snippet_only = not content
    content = content or snippet
    return content[:2000], len(content) > 2000 or snippet_only


class GmailMailReader:
    """One connected Gmail account; sign-in is explicit and reads are GET-only.

    Token refresh uses Google's OAuth token endpoint, never a mailbox mutation.
    No API credentials, bodies, or tokens are written into the repository.
    """

    def __init__(self, *, store=None, session_factory=None):
        self._store = store if store is not None else MacOSMailTokenStore()
        self._session_factory = session_factory
        self._lock = RLock()

    @staticmethod
    def _check_scopes(credentials):
        scopes = credentials.granted_scopes if credentials.granted_scopes is not None else credentials.scopes
        if set(scopes or ()) != {READ_SCOPE}:
            raise GmailConnectionError("Gmail authorization must grant only read-only Gmail access. Reconnect using a dedicated Airis OAuth client.")

    def connect(self, client_file):
        """Open Google's sign-in page after the user chooses a desktop client file."""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            config = json.loads(Path(client_file).read_text(encoding="utf-8"))
            installed = config.get("installed", {})
            if (not isinstance(installed, dict) or not installed.get("client_id") or
                    installed.get("auth_uri") != "https://accounts.google.com/o/oauth2/auth" or
                    installed.get("token_uri") != "https://oauth2.googleapis.com/token"):
                raise GmailConnectionError("Choose the JSON file for a Google OAuth Desktop app client.")
            flow = InstalledAppFlow.from_client_config(config, scopes=[READ_SCOPE], autogenerate_code_verifier=True)
            flow.fetch_token = partial(flow.fetch_token, timeout=20)
            # The library's callback handler logs the full URL, including the
            # authorization code, at INFO. Keep it out of desktop.log.
            logging.getLogger("google_auth_oauthlib.flow").setLevel(logging.WARNING)
            credentials = flow.run_local_server(host="localhost", port=0, timeout_seconds=180,
                authorization_prompt_message="", success_message="Google sign-in received. Return to Airis to check the connection.",
                access_type="offline", prompt="consent", include_granted_scopes="false")
            self._check_scopes(credentials)
            if not credentials.refresh_token:
                raise GmailConnectionError("Google did not grant persistent access. Reconnect Gmail and approve read-only access.")
            with self._lock:
                self._store.save(credentials.to_json())
        except GmailConnectionError:
            raise
        except Exception:
            raise GmailConnectionError("Gmail connection failed or timed out. Check your OAuth Desktop client, Gmail API, test-user setup, and macOS Keychain access, then retry.") from None

    def disconnect(self):
        with self._lock:
            try:
                self._store.clear()
            except Exception:
                raise GmailConnectionError("Could not remove Gmail credentials from macOS Keychain.") from None

    def _credentials(self):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        raw = self._store.load()
        if not raw:
            raise GmailConnectionError(_CONNECT)
        config = json.loads(raw)
        if config.get("token_uri") != "https://oauth2.googleapis.com/token":
            raise GmailConnectionError("Invalid Gmail authorization. " + _CONNECT)
        # Validate cached scopes before supplying requested scopes to the library.
        if set(config.get("scopes", ())) != {READ_SCOPE}:
            raise GmailConnectionError("Invalid Gmail permission. " + _CONNECT)
        credentials = Credentials.from_authorized_user_info(config, scopes=[READ_SCOPE])
        self._check_scopes(credentials)
        if not credentials.valid:
            if not credentials.refresh_token:
                raise GmailConnectionError(_CONNECT)
            credentials.refresh(partial(Request(), timeout=20))
            self._check_scopes(credentials)
            self._store.save(credentials.to_json())
        return credentials

    @staticmethod
    def _get(session, credentials, path, params=None):
        # Paths are internal constants or validated Gmail message identifiers.
        with session.get(_API + path, params=params,
                         headers={"Authorization": "Bearer " + credentials.token},
                         timeout=(5, 15), allow_redirects=False, stream=True) as response:
            if response.status_code == 401:
                raise GmailConnectionError("Gmail authorization expired. " + _CONNECT)
            if response.status_code == 403:
                raise GmailConnectionError("Gmail access was denied. Enable the Gmail API and check your Google Workspace administrator's app policy.")
            if response.status_code != 200:
                raise GmailConnectionError("Gmail couldn't complete the read request. Please try again later.")
            parts, size = [], 0
            for part in response.iter_content(chunk_size=65536):
                size += len(part)
                if size > 2 * 1024 * 1024:
                    raise GmailConnectionError("This Gmail message exceeds the reader's size limit.")
                parts.append(part)
            data = json.loads(b"".join(parts))
            if not isinstance(data, dict):
                raise ValueError("Invalid Gmail response.")
            return data

    def read_unread(self, request):
        if not isinstance(request, EmailRequest):
            raise TypeError("The Gmail reader requires an EmailRequest.")
        try:
            import requests
            with self._lock:
                credentials = self._credentials()
                factory = self._session_factory or requests.Session
                with factory() as session:
                    inbox = self._get(session, credentials, "labels/INBOX")
                    count = inbox.get("messagesUnread")
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid Inbox count.")
                    if count == 0:
                        return EmailResult(scope="gmail_inbox", mailbox_label="your Gmail Inbox")
                    listed = self._get(session, credentials, "messages",
                                       {"labelIds": ["INBOX", "UNREAD"], "maxResults": request.limit})
                    rows = listed.get("messages", [])
                    if not isinstance(rows, list) or len(rows) != min(count, request.limit):
                        raise GmailConnectionError("Your Gmail Inbox changed while I was reading it. Please try again.")
                    messages = []
                    for row in rows:
                        message_id = row.get("id")
                        if not isinstance(message_id, str) or not re.fullmatch(r"[a-fA-F0-9]{1,40}", message_id):
                            raise ValueError("Invalid Gmail message identifier.")
                        query = {"format": "full" if request.include_body else "metadata"}
                        if not request.include_body:
                            query["metadataHeaders"] = ["From", "Subject"]
                        data = self._get(session, credentials, "messages/" + message_id, query)
                        if data.get("id") != message_id or not {"INBOX", "UNREAD"} <= set(data.get("labelIds", ())):
                            raise GmailConnectionError("Your Gmail Inbox changed while I was reading it. Please try again.")
                        payload = data.get("payload", {})
                        headers = {header["name"].lower(): header["value"] for header in payload.get("headers", [])}
                        sender, subject = headers.get("from", "Unknown sender"), headers.get("subject", "")
                        received = datetime.fromtimestamp(int(data["internalDate"]) / 1000, timezone.utc).isoformat()
                        body, truncated = _body(payload, data.get("snippet", "")) if request.include_body else ("", False)
                        messages.append(EmailMessage(message_id, subject[:300], sender[:300], received, body, truncated))
                    messages.sort(key=lambda item: item.received_at, reverse=True)
                    if len({message.id for message in messages}) != len(messages):
                        raise ValueError("Duplicate Gmail messages.")
                    return EmailResult(tuple(messages), count, scope="gmail_inbox", mailbox_label="your Gmail Inbox")
        except GmailConnectionError as error:
            return EmailResult(error=str(error), scope="gmail_inbox", mailbox_label="your Gmail Inbox")
        except ImportError:
            return EmailResult(error="Install actions/requirements-email.txt in the Airis environment to connect Gmail.")
        except Exception:
            # Server exceptions can include request headers, tokens or mail data.
            return EmailResult(error="I couldn't read Gmail. Check your connection and Keychain access, or reconnect Gmail in Settings.",
                               scope="gmail_inbox", mailbox_label="your Gmail Inbox")
