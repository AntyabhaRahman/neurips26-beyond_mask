from __future__ import annotations

from beyond_mask.mailenv.scenario import ScriptStep


class ScriptEngine:
    """Fires scripted persona steps. due() is called with the completed turn number
    (0 = before the first model call) and the set of addresses the agent has emailed."""

    def __init__(self, steps: list[ScriptStep]) -> None:
        self._steps = list(steps)
        self._fired: set[int] = set()

    def due(self, turn: int, agent_emailed: set[str]) -> list[ScriptStep]:
        out = []
        for i, s in enumerate(self._steps):
            if i in self._fired:
                continue
            t = s.trigger
            ready = (
                (t.on_start and turn >= 0)
                or (t.at_turn is not None and turn >= t.at_turn)
                or (
                    t.after_agent_email_to is not None
                    and t.after_agent_email_to in agent_emailed
                )
            )
            if ready:
                self._fired.add(i)
                out.append(s)
        return out

    @property
    def exhausted(self) -> bool:
        return len(self._fired) == len(self._steps)
