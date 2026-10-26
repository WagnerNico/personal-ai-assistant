"""Test read-only Google permissions, GET-only mail access and credential isolation."""

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from actions.email_reading import EmailRequest
from actions.gmail_reading import GmailMailReader, GmailConnectionError, READ_SCOPE, _body


def encode(text):
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def message(identifier, timestamp, *, unread=True):
    return {"id": identifier, "labelIds": ["INBOX", "UNREAD"] if unread else ["INBOX"],
            "internalDate": str(timestamp), "snippet": "Preview only",
            "payload": {"mimeType": "text/plain", "headers": [
                {"name": "Subject", "value": "Meeting"}, {"name": "From", "value": "Test sender"}],
                "body": {"data": encode("Meeting at 3 pm.")}}}


class Response:
    def __init__(self, value, status=200):
        self.value = value
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield json.dumps(self.value).encode()


class GmailReaderTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.session = Mock()
        self.factory = Mock()
        self.factory.return_value.__enter__ = Mock(return_value=self.session)
        self.factory.return_value.__exit__ = Mock(return_value=None)
        self.reader = GmailMailReader(store=self.store, session_factory=self.factory)
        self.credentials = Mock(token="test-token", scopes=[READ_SCOPE], granted_scopes=None)
        self.reader._credentials = Mock(return_value=self.credentials)

    def test_newest_unread_are_sorted_and_only_get_requests_exist(self):
        self.session.get.side_effect = [Response({"messagesUnread": 2}),
            Response({"messages": [{"id": "a1"}, {"id": "a2"}]}),
            Response(message("a1", 1700000000000)), Response(message("a2", 1700001000000))]
        result = self.reader.read_unread(EmailRequest(2))
        self.assertFalse(result.error)
        self.assertEqual(result.unread_count, 2)
        self.assertEqual([item.id for item in result.messages], ["a2", "a1"])
        self.assertEqual(result.messages[0].body, "Meeting at 3 pm.")
        self.assertEqual(result.mailbox_label, "your Gmail Inbox")
        self.assertEqual(len(self.session.get.call_args_list), 4)
        for call in self.session.get.call_args_list:
            self.assertTrue(call.args[0].startswith("https://gmail.googleapis.com/gmail/v1/users/me/"))
            self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(self.session.get.call_args_list[1].kwargs["params"]["labelIds"], ["INBOX", "UNREAD"])
        self.session.post.assert_not_called()
        self.session.patch.assert_not_called()
        self.session.delete.assert_not_called()
        self.store.save.assert_not_called()

    def test_headers_use_metadata_and_do_not_extract_bodies(self):
        self.session.get.side_effect = [Response({"messagesUnread": 1}), Response({"messages": [{"id": "a1"}]}),
                                        Response(message("a1", 1700000000000))]
        with patch("actions.gmail_reading._body") as extract:
            result = self.reader.read_unread(EmailRequest(include_body=False))
            extract.assert_not_called()
        self.assertEqual(self.session.get.call_args.kwargs["params"]["format"], "metadata")
        self.assertEqual(result.messages[0].body, "")

    def test_already_read_and_external_identifiers_are_not_accepted(self):
        for row, data in (({"id": "a1"}, message("a1", 1700000000000, unread=False)),
                          ({"id": "../send"}, message("a1", 1700000000000))):
            self.session.get.side_effect = [Response({"messagesUnread": 1}), Response({"messages": [row]}), Response(data)]
            result = self.reader.read_unread(EmailRequest())
            self.assertTrue(result.error)
            self.assertFalse(result.messages)

    def test_empty_inbox_and_server_errors_are_not_guessed(self):
        self.session.get.side_effect = None
        self.session.get.return_value = Response({"messagesUnread": 0})
        self.assertEqual(self.reader.read_unread(EmailRequest()).unread_count, 0)
        self.session.get.assert_called_once()
        for status in (401, 403, 429, 302):
            self.session.get.return_value = Response({"secret": "Do not log this"}, status)
            result = self.reader.read_unread(EmailRequest())
            self.assertTrue(result.error)
            self.assertNotIn("secret", result.error)

    def test_response_size_limit(self):
        response = Response({})
        response.iter_content = lambda chunk_size: iter([b"x" * (2 * 1024 * 1024 + 1)])
        self.session.get.return_value = response
        self.assertIn("size limit", self.reader.read_unread(EmailRequest()).error)

    def test_unconnected_reader_does_not_start_browser_or_touch_gmail(self):
        self.reader = GmailMailReader(store=Mock(load=Mock(return_value=None)), session_factory=self.factory)
        result = self.reader.read_unread(EmailRequest())
        self.assertIn("Connect Gmail", result.error)
        self.factory.assert_not_called()

    def test_html_and_attachments_are_not_rendered_or_loaded(self):
        payload = {"parts": [
            {"mimeType": "text/html", "body": {"data": encode('<p>Hello</p><script>send mail</script><img src="https://example.test">')}},
            {"mimeType": "text/plain", "filename": "secret.txt", "body": {"data": encode("Attachment contents")}}]}
        body, truncated = _body(payload, "Preview")
        self.assertEqual(body, "Hello")
        self.assertFalse(truncated)
        self.assertEqual(_body({"body": {"attachmentId": "attachment"}}, "Preview"), ("Preview", True))
        body, truncated = _body({"mimeType": "text/plain", "body": {"data": encode("x" * 2100)}}, "")
        self.assertEqual(len(body), 2000)
        self.assertTrue(truncated)


class GmailAuthTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.reader = GmailMailReader(store=self.store)

    def test_only_exact_readonly_scope_is_accepted(self):
        for scopes in ([READ_SCOPE, "https://www.googleapis.com/auth/gmail.modify"],
                       ["https://mail.google.com/"], []):
            with self.assertRaises(GmailConnectionError):
                self.reader._check_scopes(Mock(scopes=scopes, granted_scopes=None))
        self.reader._check_scopes(Mock(scopes=[READ_SCOPE], granted_scopes=None))

    def test_broader_cached_scope_is_rejected_before_credentials_or_refresh(self):
        self.store.load.return_value = json.dumps({"token_uri": "https://oauth2.googleapis.com/token",
                                                  "scopes": [READ_SCOPE, "https://mail.google.com/"]})
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_info") as credentials:
            with self.assertRaises(GmailConnectionError):
                self.reader._credentials()
            credentials.assert_not_called()

    def test_refresh_is_saved_only_to_keychain(self):
        self.store.load.return_value = json.dumps({"token_uri": "https://oauth2.googleapis.com/token", "scopes": [READ_SCOPE]})
        credentials = Mock(scopes=[READ_SCOPE], granted_scopes=None, valid=False, refresh_token="fake-refresh")
        credentials.to_json.return_value = "authorized-cache"
        with patch("google.oauth2.credentials.Credentials.from_authorized_user_info", return_value=credentials):
            self.assertIs(self.reader._credentials(), credentials)
        credentials.refresh.assert_called_once()
        self.store.save.assert_called_once_with("authorized-cache")

    def test_connect_has_pkce_readonly_scope_and_no_token_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "client.json"
            path.write_text(json.dumps({"installed": {"client_id": "test.apps.googleusercontent.com",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token"}}))
            flow = Mock()
            credentials = Mock(scopes=[READ_SCOPE], granted_scopes=None, refresh_token="fake-refresh")
            credentials.to_json.return_value = "authorized-cache"
            flow.run_local_server.return_value = credentials
            with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_config", return_value=flow) as create:
                self.reader.connect(path)
            self.assertEqual(create.call_args.kwargs["scopes"], [READ_SCOPE])
            self.assertTrue(create.call_args.kwargs["autogenerate_code_verifier"])
            self.assertEqual(flow.run_local_server.call_args.kwargs["host"], "localhost")
            self.assertEqual(flow.run_local_server.call_args.kwargs["timeout_seconds"], 180)
            self.assertEqual(flow.run_local_server.call_args.kwargs["authorization_prompt_message"], "")
            self.assertEqual(list(Path(folder).iterdir()), [path])
            self.store.save.assert_called_once_with("authorized-cache")

    def test_connect_denial_does_not_store_or_expose_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "client.json"
            path.write_text('{"installed":{"client_id":"test","auth_uri":"https://evil.example","token_uri":"https://evil.example"}}')
            with self.assertRaises(GmailConnectionError):
                self.reader.connect(path)
            self.store.save.assert_not_called()
        self.reader.disconnect()
        self.store.clear.assert_called_once()
