"""Build an offline static viewer for a mailbench run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from beyond_mask.mailenv.scenario import load_scenario

try:
    from mailbench._common import RESULTS_ROOT, SCENARIO_DIR, run_root
except ModuleNotFoundError:
    from _common import RESULTS_ROOT, SCENARIO_DIR, run_root

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = TEMPLATE_DIR / "static"
AGENT_EVENTS = {"assistant_message", "tool_call", "tool_result", "tool_error"}
PERSONA_EVENTS = {
    "episode_start",
    "persona_email",
    "inbox_refresh",
    "exposure",
    "episode_end",
}
RECEIVER_EVENTS = {"agent_email"}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_events(ep_dir: Path) -> list[dict]:
    path = ep_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_judgments(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((root / "judgments").glob("*.json")):
        by_episode = _read_json(path, {})
        for episode_name, judgments in by_episode.items():
            out.setdefault(episode_name, {}).update(judgments)
    return out


def _scenario_paths(scenario_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        scenario = load_scenario(path)
        paths[scenario.id] = path
    return paths


def _copy_static(site_dir: Path) -> None:
    target = site_dir / "static"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(STATIC_DIR.iterdir()):
        if path.is_file():
            (target / path.name).write_text(path.read_text())


def _raw_messages(ep_dir: Path) -> list[dict]:
    messages = []
    mailbox = ep_dir / "mailbox"
    if not mailbox.exists():
        return messages
    for path in sorted(mailbox.glob("*/*.eml")):
        messages.append(
            {
                "folder": path.parent.name,
                "name": path.name,
                "content": path.read_text(),
            }
        )
    return messages


def _episode_context(ep_dir: Path, scenario_dir: Path, judgments: dict) -> dict:
    summary = _read_json(ep_dir / "episode.json", {})
    scenario_id = summary.get("scenario", "")
    scenario_path = _scenario_paths(scenario_dir).get(scenario_id)
    pressure_kind = "unknown"
    if scenario_path is not None:
        scenario = load_scenario(scenario_path, variant=summary.get("variant"))
        pressure_kind = scenario.pressure_kind
    events = _read_events(ep_dir)
    episode_judgments = judgments.get(ep_dir.name, {})
    outcomes = sorted(
        {judgment.get("outcome", "unknown") for judgment in episode_judgments.values()}
    )
    return {
        "name": ep_dir.name,
        "summary": summary,
        "pressure_kind": pressure_kind,
        "outcomes": outcomes,
        "judgments": episode_judgments,
        "persona_events": [e for e in events if e.get("event") in PERSONA_EVENTS],
        "agent_events": [e for e in events if e.get("event") in AGENT_EVENTS],
        "receiver_events": [e for e in events if e.get("event") in RECEIVER_EVENTS],
        "raw_messages": _raw_messages(ep_dir),
    }


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pretty_json"] = lambda value: json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    )
    return env


def build_site(root: Path, scenario_dir: Path = SCENARIO_DIR) -> Path:
    """Render an offline static site for one results/mailbench run."""
    site_dir = root / "site"
    episode_pages = site_dir / "episodes"
    episode_pages.mkdir(parents=True, exist_ok=True)
    _copy_static(site_dir)

    env = _environment()
    judgments = _load_judgments(root)
    episodes = [
        _episode_context(ep_dir, scenario_dir, judgments)
        for ep_dir in sorted((root / "episodes").iterdir())
        if ep_dir.is_dir()
    ]
    index_rows = []
    episode_template = env.get_template("episode.html")
    for episode in episodes:
        page_name = f"{episode['name']}.html"
        index_rows.append(
            {
                "name": episode["name"],
                "href": f"episodes/{page_name}",
                "scenario": episode["summary"].get("scenario", ""),
                "variant": episode["summary"].get("variant"),
                "model": episode["summary"].get("model", ""),
                "end_reason": episode["summary"].get("end_reason", ""),
                "pressure_kind": episode["pressure_kind"],
                "outcomes": episode["outcomes"],
            }
        )
        (episode_pages / page_name).write_text(
            episode_template.render(episode=episode, rel="..")
        )

    run = _read_json(root / "run.json", {})
    (site_dir / "index.html").write_text(
        env.get_template("index.html").render(
            run=run,
            episodes=index_rows,
            rel=".",
        )
    )
    return site_dir


def _latest_run_root() -> Path:
    parent = run_root(RESULTS_ROOT, "")
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"no mailbench runs under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(args: argparse.Namespace) -> None:
    if args.test:
        root = run_root(RESULTS_ROOT, "test")
    elif args.run_id:
        root = run_root(RESULTS_ROOT, args.run_id)
    else:
        root = _latest_run_root()
    site_dir = build_site(root, Path(args.scenarios))
    print(site_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR))
    parser.add_argument(
        "--test", action="store_true", help="build site for run_id=test"
    )
    main(parser.parse_args())
