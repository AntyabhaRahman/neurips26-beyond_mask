from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from beyond_mask.mailenv.store import Address, MailStore, StoredEmail, to_rfc5322


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], ToolResult]


def to_openrouter(defs: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": d.name,
                "description": d.description,
                "parameters": d.input_schema,
            },
        }
        for d in defs
    ]


def dispatch(defs: list[ToolDef], name: str, arguments: str) -> ToolResult:
    tool = next((d for d in defs if d.name == name), None)
    if tool is None:
        return ToolResult(f"Unknown tool: {name}", is_error=True)
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolResult(f"Invalid JSON arguments: {exc}", is_error=True)
    missing = [k for k in tool.input_schema.get("required", []) if k not in args]
    if missing:
        return ToolResult(f"Missing required arguments: {missing}", is_error=True)
    try:
        return tool.handler(args)
    except (KeyError, TypeError, ValueError) as exc:
        return ToolResult(f"Tool error: {exc}", is_error=True)


def email_summary(e: StoredEmail) -> dict:
    return {
        "email_id": e.email_id,
        "message_id": e.message_id,
        "thread_id": e.thread_id,
        "from": e.sender.as_dict(),
        "to": [a.address for a in e.to],
        "cc": [a.address for a in e.cc],
        "subject": e.subject,
        "date": e.date.isoformat(),
        "unread": e.unread,
    }


def email_json(e: StoredEmail) -> dict:
    return {
        **email_summary(e),
        "text_body": e.body,
        "in_reply_to": e.in_reply_to,
        "references": list(e.references),
    }


def _parse_recipients(raw: object, field_name: str, *, required: bool) -> list[Address]:
    if raw is None and not required:
        return []
    if (
        not isinstance(raw, list)
        or (required and not raw)
        or not all(isinstance(a, str) and "@" in a for a in raw)
    ):
        raise ValueError(f"{field_name} must be a non-empty list of email addresses")
    return [Address(a) for a in raw]


def build_email_tools(store: MailStore, agent: Address) -> list[ToolDef]:
    def list_emails(args: dict) -> ToolResult:
        emails = store.list(
            args.get("folder", "INBOX"),
            int(args.get("limit", 20)),
            bool(args.get("unread_only", False)),
        )
        return ToolResult(json.dumps([email_summary(e) for e in emails]))

    def get_email(args: dict) -> ToolResult:
        e = store.get(args["email_id"])
        if e is None:
            return ToolResult(f"No email with id {args['email_id']}", is_error=True)
        store.mark_read(e.email_id)
        if args.get("format", "json") == "raw":
            return ToolResult(to_rfc5322(e))
        return ToolResult(json.dumps(email_json(e)))

    def send_email(args: dict) -> ToolResult:
        to = _parse_recipients(args["to"], "to", required=True)
        cc = _parse_recipients(args.get("cc"), "cc", required=False)
        sent = store.send(
            agent,
            to,
            cc,
            args["subject"],
            args["body"],
            in_reply_to=args.get("in_reply_to"),
        )
        return ToolResult(json.dumps({"status": "sent", "email": email_json(sent)}))

    def search_emails(args: dict) -> ToolResult:
        hits = store.search(args["query"], args.get("folder", "INBOX"))
        return ToolResult(json.dumps([email_summary(e) for e in hits]))

    obj = {"type": "object", "additionalProperties": False}
    return [
        ToolDef(
            "list_emails",
            "List emails in a folder (summaries without bodies).",
            {
                **obj,
                "properties": {
                    "folder": {"type": "string", "enum": ["INBOX", "Sent"]},
                    "limit": {"type": "integer"},
                    "unread_only": {"type": "boolean"},
                },
                "required": [],
            },
            list_emails,
        ),
        ToolDef(
            "get_email",
            "Fetch one email in full by email_id; marks it read. "
            "format='raw' returns the raw RFC 5322 message.",
            {
                **obj,
                "properties": {
                    "email_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["json", "raw"]},
                },
                "required": ["email_id"],
            },
            get_email,
        ),
        ToolDef(
            "send_email",
            "Send an email from your account. Pass in_reply_to=<email_id> "
            "to reply within a thread (threading headers are set for you).",
            {
                **obj,
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "cc": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "in_reply_to": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            send_email,
        ),
        ToolDef(
            "search_emails",
            "Search a folder by keyword over subject and body.",
            {
                **obj,
                "properties": {
                    "query": {"type": "string"},
                    "folder": {"type": "string", "enum": ["INBOX", "Sent"]},
                },
                "required": ["query"],
            },
            search_emails,
        ),
    ]
