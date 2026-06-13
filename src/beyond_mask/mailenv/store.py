from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime, formataddr
from pathlib import Path

CLOCK_STEP = timedelta(minutes=7)
FOLDERS = ("INBOX", "Sent")


@dataclass(frozen=True)
class Address:
    address: str
    name: str = ""

    def as_header(self) -> str:
        return formataddr((self.name or None, self.address))

    def as_dict(self) -> dict:
        return {"address": self.address, "name": self.name}


@dataclass(frozen=True)
class StoredEmail:
    email_id: str
    message_id: str
    folder: str
    sender: Address
    to: tuple[Address, ...]
    cc: tuple[Address, ...]
    subject: str
    date: datetime
    body: str
    in_reply_to: str | None
    references: tuple[str, ...]
    unread: bool = True

    @property
    def thread_id(self) -> str:
        return self.references[0] if self.references else self.message_id


def to_rfc5322(e: StoredEmail) -> str:
    msg = EmailMessage()
    msg["From"] = e.sender.as_header()
    msg["To"] = ", ".join(a.as_header() for a in e.to)
    if e.cc:
        msg["Cc"] = ", ".join(a.as_header() for a in e.cc)
    msg["Subject"] = e.subject
    msg["Date"] = format_datetime(e.date)
    msg["Message-ID"] = e.message_id
    if e.in_reply_to:
        msg["In-Reply-To"] = e.in_reply_to
    if e.references:
        msg["References"] = " ".join(e.references)
    msg.set_content(e.body)
    return msg.as_string()


class MailStore:
    def __init__(
        self,
        scenario_id: str,
        domain: str,
        base_time: datetime,
        mailbox_dir: Path | None = None,
    ) -> None:
        self._scenario_id = scenario_id
        self._domain = domain
        self._clock = base_time
        self._mailbox_dir = mailbox_dir
        self._emails: list[StoredEmail] = []
        self._seq = 0

    def _next(self, sender, to, cc, subject, body, in_reply_to, folder) -> StoredEmail:
        self._seq += 1
        self._clock += CLOCK_STEP
        digest = hashlib.sha256(
            f"{self._scenario_id}|{self._seq}".encode()
        ).hexdigest()[:16]
        parent = (
            self.get_by_message_id(self._resolve_message_id(in_reply_to))
            if in_reply_to
            else None
        )
        email = StoredEmail(
            email_id=f"em-{self._seq:04d}",
            message_id=f"<{digest}@{self._domain}>",
            folder=folder,
            sender=sender,
            to=tuple(to),
            cc=tuple(cc),
            subject=subject,
            date=self._clock,
            body=body,
            in_reply_to=parent.message_id if parent else None,
            references=(*parent.references, parent.message_id) if parent else (),
            unread=folder == "INBOX",
        )
        self._emails.append(email)
        self._write_eml(email)
        return email

    def _resolve_message_id(self, ref: str) -> str:
        for e in self._emails:  # accept either email_id or message_id
            if ref in (e.email_id, e.message_id):
                return e.message_id
        raise KeyError(f"unknown email reference: {ref}")

    def deliver(
        self, sender, to, cc, subject, body, *, in_reply_to=None
    ) -> StoredEmail:
        return self._next(sender, to, cc, subject, body, in_reply_to, "INBOX")

    def send(self, sender, to, cc, subject, body, *, in_reply_to=None) -> StoredEmail:
        return self._next(sender, to, cc, subject, body, in_reply_to, "Sent")

    def get(self, email_id: str) -> StoredEmail | None:
        return next((e for e in self._emails if e.email_id == email_id), None)

    def get_by_message_id(self, message_id: str) -> StoredEmail | None:
        return next((e for e in self._emails if e.message_id == message_id), None)

    def list(
        self, folder: str = "INBOX", limit: int = 20, unread_only: bool = False
    ) -> list[StoredEmail]:
        out = [
            e
            for e in self._emails
            if e.folder == folder and (e.unread or not unread_only)
        ]
        return out[:limit]

    def search(self, query: str, folder: str = "INBOX") -> list[StoredEmail]:
        q = query.lower()
        return [
            e
            for e in self._emails
            if e.folder == folder and (q in e.subject.lower() or q in e.body.lower())
        ]

    def mark_read(self, email_id: str) -> None:
        for i, e in enumerate(self._emails):
            if e.email_id == email_id:
                self._emails[i] = replace(e, unread=False)

    def _write_eml(self, e: StoredEmail) -> None:
        if self._mailbox_dir is None:
            return
        folder = self._mailbox_dir / e.folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self._seq:04d}__{e.email_id}.eml").write_text(to_rfc5322(e))
