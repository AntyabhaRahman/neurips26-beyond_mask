from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from email.utils import parseaddr
from pathlib import Path

import yaml

from beyond_mask.mailenv.store import Address

TOP_LEVEL = {
    "id",
    "agent",
    "base_time",
    "initial_inbox",
    "script",
    "ground_truth",
    "belief_elicitation",
    "end",
    "pressure_kind",
    "variants",
}


class ScenarioError(ValueError):
    pass


def _addr(raw: str) -> Address:
    name, addr = parseaddr(raw)
    if not addr or "@" not in addr:
        raise ScenarioError(f"invalid address: {raw!r}")
    return Address(addr, name)


@dataclass(frozen=True)
class ScenarioEmail:
    id: str | None
    sender: Address
    subject: str
    body: str
    in_reply_to: str | None = None


@dataclass(frozen=True)
class Trigger:
    on_start: bool = False
    at_turn: int | None = None
    after_agent_email_to: str | None = None


@dataclass(frozen=True)
class ScriptStep:
    trigger: Trigger
    email: ScenarioEmail

    @property
    def email_body(self) -> str:
        return self.email.body


@dataclass(frozen=True)
class Proposition:
    id: str
    proposition: str
    evidence_emails: tuple[str, ...] = ()
    world_knowledge: bool = False


@dataclass(frozen=True)
class EndCondition:
    agent_emails: str
    max_turns: int = 12


@dataclass(frozen=True)
class AgentSpec:
    address: Address
    system_prompt: str


@dataclass(frozen=True)
class Scenario:
    id: str
    agent: AgentSpec
    base_time: datetime
    initial_inbox: tuple[ScenarioEmail, ...]
    script: tuple[ScriptStep, ...]
    ground_truth: tuple[Proposition, ...]
    belief_elicitation: dict[str, tuple[str, ...]]
    end: EndCondition
    pressure_kind: str
    variant: str | None = None
    variant_names: tuple[str, ...] = ()

    @property
    def domain(self) -> str:
        return self.agent.address.address.split("@", 1)[1]

    @property
    def slug(self) -> str:
        return f"{self.id}@{self.variant}" if self.variant else self.id


def _parse_email(raw: dict) -> ScenarioEmail:
    return ScenarioEmail(
        id=raw.get("id"),
        sender=_addr(raw["from"]),
        subject=raw.get("subject", ""),
        body=raw["body"],
        in_reply_to=raw.get("in_reply_to"),
    )


def _apply_variant(raw: dict, name: str) -> dict:
    """Apply a named variant to a raw scenario dict.

    NOTE: drops apply first, so set list indices refer to the post-drop script.
    """
    variants = raw.get("variants") or {}
    if name not in variants:
        raise ScenarioError(f"unknown variant {name!r} (have: {sorted(variants)})")
    merged = copy.deepcopy(raw)
    spec = variants[name]

    # Issue 2: wrap drop errors as ScenarioError
    for idx in sorted(spec.get("drop_script_steps", []), reverse=True):
        try:
            del merged["script"][idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise ScenarioError(
                f"variant {name!r}: drop index {idx!r} does not resolve: {exc}"
            ) from exc

    for dotted, value in (spec.get("set") or {}).items():
        node = merged
        parts = dotted.split(".")
        try:
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node[part]
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            raise ScenarioError(
                f"variant {name!r}: path {dotted!r} does not resolve: {exc}"
            ) from exc

        last = parts[-1]
        try:
            if isinstance(node, list):
                node[int(last)] = value
            else:
                # Issue 3: require the key to already exist to catch typos
                if last not in node:
                    raise ScenarioError(
                        f"variant {name!r}: path {dotted!r} targets unknown key {last!r}"
                    )
                node[last] = value
        except (IndexError, ValueError, TypeError) as exc:
            raise ScenarioError(
                f"variant {name!r}: path {dotted!r} does not resolve: {exc}"
            ) from exc

    return merged


def parse_scenario(raw: dict, variant: str | None = None) -> Scenario:
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise ScenarioError(f"unknown fields: {sorted(unknown)}")
    variant_names = tuple(sorted(raw.get("variants") or {}))
    if variant:
        raw = _apply_variant(raw, variant)
    try:
        inbox = tuple(_parse_email(e) for e in raw["initial_inbox"])

        # Issue 4: build ordered list of declared email ids for in_reply_to validation.
        # Order: initial_inbox first (in order), then script emails (in list order,
        # i.e. assumed firing order).
        declared_ids: list[str] = [e.id for e in inbox if e.id is not None]

        steps_raw = raw.get("script", [])
        steps = []
        for s in steps_raw:
            email = _parse_email(s["email"])
            # Validate in_reply_to before appending this email's id
            ref = email.in_reply_to
            if ref is not None and ref not in declared_ids:
                raise ScenarioError(
                    f"email replies to unknown or later id {ref!r} "
                    f"(script steps are assumed listed in firing order)"
                )
            if email.id is not None:
                declared_ids.append(email.id)
            steps.append(
                ScriptStep(
                    # Issue 1: key presence alone is not enough; use .get() with False default
                    Trigger(
                        on_start=bool(s["trigger"].get("on_start", False)),
                        at_turn=s["trigger"].get("at_turn"),
                        after_agent_email_to=s["trigger"].get("after_agent_email_to"),
                    ),
                    email,
                )
            )
        steps = tuple(steps)

        end = EndCondition(
            raw["end"]["agent_emails"], int(raw["end"].get("max_turns", 12))
        )

        # Issue 5: at_turn must be strictly less than max_turns
        for i, step in enumerate(steps):
            t = step.trigger
            if t.at_turn is not None and not (0 < t.at_turn < end.max_turns):
                raise ScenarioError(
                    f"script step {i}: at_turn={t.at_turn} must satisfy "
                    f"0 < at_turn < max_turns ({end.max_turns})"
                )

        props = []
        known_email_ids = {e.id for e in inbox if e.id}
        for p in raw["ground_truth"]:
            world = p.get("evidence") == "world_knowledge"
            if not world:
                evid = p["evidence"]["emails"]
                # Issue 6: evidence emails must be a list, not a string
                if not isinstance(evid, list):
                    raise ScenarioError(
                        f"proposition {p['id']}: evidence emails must be a list"
                    )
                emails = tuple(evid)
                bad = set(emails) - known_email_ids
                if bad:
                    raise ScenarioError(
                        f"proposition {p['id']}: unknown evidence emails {sorted(bad)}"
                    )
            else:
                emails = ()
            props.append(Proposition(p["id"], p["proposition"], emails, world))
        beliefs = {k: tuple(v) for k, v in raw["belief_elicitation"].items()}
        bad_beliefs = set(beliefs) - {p.id for p in props}
        if bad_beliefs:
            raise ScenarioError(
                f"belief prompts for unknown propositions: {sorted(bad_beliefs)}"
            )
        base_time = (
            datetime.fromisoformat(raw["base_time"].replace("Z", "+00:00"))
            if isinstance(raw["base_time"], str)
            else raw["base_time"]
        )
        return Scenario(
            id=raw["id"],
            agent=AgentSpec(
                _addr(f"{raw['agent'].get('name', '')} <{raw['agent']['address']}>"),
                raw["agent"]["system_prompt"],
            ),
            base_time=base_time,
            initial_inbox=inbox,
            script=steps,
            ground_truth=tuple(props),
            belief_elicitation=beliefs,
            end=end,
            pressure_kind=raw.get("pressure_kind", "unspecified"),
            variant=variant,
            variant_names=variant_names,
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ScenarioError(f"invalid scenario: {exc!r}") from exc


def load_scenario(path: Path, variant: str | None = None) -> Scenario:
    return parse_scenario(yaml.safe_load(path.read_text()), variant=variant)
