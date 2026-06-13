import json
import unittest
from datetime import UTC, datetime

from beyond_mask.mailenv.store import Address, MailStore
from beyond_mask.mailenv.tooldefs import build_email_tools, dispatch, to_openrouter

BASE = datetime(2026, 4, 6, 9, 0, tzinfo=UTC)
ALICE = Address("alice@arborlight.com", "Alice Chen")
AGENT = Address("assistant@arborlight.com", "Sam")


class ToolDefTests(unittest.TestCase):
    def setUp(self):
        self.store = MailStore("s1", "arborlight.com", BASE)
        self.store.deliver(ALICE, [AGENT], [], "Q3 numbers", "Revenue fell 12%.")
        self.tools = build_email_tools(self.store, AGENT)

    def test_openrouter_adapter_shape(self):
        spec = to_openrouter(self.tools)
        names = [t["function"]["name"] for t in spec]
        self.assertEqual(
            names, ["list_emails", "get_email", "send_email", "search_emails"]
        )
        for t in spec:
            self.assertEqual(t["type"], "function")
            self.assertIn("parameters", t["function"])
            self.assertIn("description", t["function"])

    def test_list_then_get_marks_read(self):
        listed = json.loads(dispatch(self.tools, "list_emails", "{}").content)
        self.assertEqual(listed[0]["email_id"], "em-0001")
        self.assertNotIn("text_body", listed[0])  # summaries exclude bodies
        got = json.loads(
            dispatch(self.tools, "get_email", '{"email_id": "em-0001"}').content
        )
        self.assertEqual(got["text_body"], "Revenue fell 12%.")
        self.assertIn("message_id", got)
        self.assertFalse(
            json.loads(
                dispatch(self.tools, "list_emails", '{"unread_only": true}').content
            )
        )

    def test_get_email_raw_returns_rfc5322(self):
        res = dispatch(
            self.tools, "get_email", '{"email_id": "em-0001", "format": "raw"}'
        )
        self.assertIn("Message-ID:", res.content)

    def test_send_email_threads_and_lands_in_sent(self):
        res = dispatch(
            self.tools,
            "send_email",
            json.dumps(
                {
                    "to": ["alice@arborlight.com"],
                    "subject": "Re: Q3 numbers",
                    "body": "Numbers confirmed.",
                    "in_reply_to": "em-0001",
                }
            ),
        )
        sent = json.loads(res.content)["email"]
        self.assertEqual(sent["in_reply_to"], self.store.get("em-0001").message_id)
        self.assertEqual(self.store.list("Sent")[0].body, "Numbers confirmed.")

    def test_error_results_not_exceptions(self):
        self.assertTrue(dispatch(self.tools, "nuke_inbox", "{}").is_error)
        self.assertTrue(dispatch(self.tools, "get_email", "{not json").is_error)
        self.assertTrue(
            dispatch(self.tools, "get_email", "{}").is_error
        )  # missing required
        self.assertTrue(
            dispatch(self.tools, "get_email", '{"email_id": "em-9999"}').is_error
        )
