import tempfile
import unittest
from datetime import UTC, datetime
from email import message_from_string
from pathlib import Path

from beyond_mask.mailenv.store import Address, MailStore, CLOCK_STEP

BASE = datetime(2026, 4, 6, 9, 0, tzinfo=UTC)
ALICE = Address("alice@arborlight.com", "Alice Chen")
AGENT = Address("assistant@arborlight.com", "Sam")


def make_store(tmp: Path | None = None) -> MailStore:
    return MailStore("q3_spin", "arborlight.com", BASE, mailbox_dir=tmp)


class MailStoreTests(unittest.TestCase):
    def test_ids_and_clock_are_deterministic(self):
        a, b = make_store(), make_store()
        e1 = a.deliver(ALICE, [AGENT], [], "Q3 numbers", "Revenue fell 12%.")
        e2 = b.deliver(ALICE, [AGENT], [], "Q3 numbers", "Revenue fell 12%.")
        self.assertEqual(e1.email_id, "em-0001")
        self.assertEqual(e1.message_id, e2.message_id)
        self.assertEqual(e1.date, e2.date)
        self.assertEqual(e1.date, BASE + CLOCK_STEP)

    def test_threading_chain(self):
        s = make_store()
        root = s.deliver(ALICE, [AGENT], [], "Q3 numbers", "body")
        reply = s.send(
            AGENT, [ALICE], [], "Re: Q3 numbers", "thanks", in_reply_to=root.email_id
        )
        reply2 = s.deliver(
            ALICE, [AGENT], [], "Re: Q3 numbers", "np", in_reply_to=reply.email_id
        )
        self.assertEqual(reply.in_reply_to, root.message_id)
        self.assertEqual(reply2.references, (root.message_id, reply.message_id))
        self.assertEqual(reply2.thread_id, root.message_id)

    def test_folders_unread_and_search(self):
        s = make_store()
        s.deliver(ALICE, [AGENT], [], "Q3 numbers", "Revenue fell 12%.")
        sent = s.send(AGENT, [ALICE], [], "hello", "world")
        self.assertEqual([e.email_id for e in s.list("Sent")], [sent.email_id])
        self.assertEqual(len(s.list("INBOX", unread_only=True)), 1)
        s.mark_read("em-0001")
        self.assertEqual(s.list("INBOX", unread_only=True), [])
        self.assertEqual(len(s.search("revenue")), 1)
        self.assertEqual(s.search("zebra"), [])

    def test_eml_files_written_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = make_store(Path(tmp))
            e = s.deliver(ALICE, [AGENT], [], "Q3 numbers", "Revenue fell 12%.")
            path = Path(tmp) / "INBOX" / "0001__em-0001.eml"
            self.assertTrue(path.exists())
            parsed = message_from_string(path.read_text())
            self.assertEqual(parsed["Message-ID"], e.message_id)
            self.assertIn("Alice Chen", parsed["From"])
            self.assertIn("Revenue fell 12%.", parsed.get_payload())
