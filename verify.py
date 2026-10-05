#!/usr/bin/env python3
"""Report the model each subagent actually ran on, from its own transcript.

The hook rewriting `model` proves only that the hook ran. This proves Claude
Code honoured it. Subagent transcripts live beside the session transcript:

    ~/.claude/projects/<slug>/<session-id>/subagents/agent-*.jsonl
                                                     agent-*.meta.json

    ./verify.py                 # most recent session in the current project
    ./verify.py --session <id>
    ./verify.py --project ~/some/other/repo
    ./verify.py --all           # every session in the project, newest first
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECTS = pathlib.Path.home() / ".claude" / "projects"


def slug(path: pathlib.Path) -> str:
    """Claude Code's project directory name: the abs path, non-alnum to dashes."""
    return "".join(c if c.isalnum() else "-" for c in str(path.resolve()))


def sessions(project_dir: pathlib.Path) -> list[pathlib.Path]:
    """Session dirs that actually contain subagents, newest first."""
    out = [d for d in project_dir.iterdir() if d.is_dir() and (d / "subagents").is_dir()]
    return sorted(out, key=lambda d: (d / "subagents").stat().st_mtime, reverse=True)


def models_in(jsonl: pathlib.Path) -> list[str]:
    seen = []
    for line in jsonl.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "assistant":
            continue
        m = e.get("message", {}).get("model")
        if m and m not in seen:
            seen.append(m)
    return seen


def report(session: pathlib.Path) -> int:
    subs = sorted((session / "subagents").glob("agent-*.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not subs:
        return 0

    print(f"\nsession {session.name}")
    for j in subs:
        meta_path = j.with_suffix(".meta.json")
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                pass
        desc = (meta.get("description") or meta.get("name") or j.stem)[:52]
        models = models_in(j) or ["(no assistant turn)"]
        # More than one model in a single subagent means a mid-run switch, which
        # the hook cannot do -- worth seeing rather than collapsing.
        print(f"  {desc:<54} {', '.join(m.replace('claude-', '') for m in models)}")
    return len(subs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=".", help="repo path (default: cwd)")
    p.add_argument("--session", help="session id, instead of the most recent")
    p.add_argument("--all", action="store_true", help="every session, newest first")
    a = p.parse_args()

    d = PROJECTS / slug(pathlib.Path(a.project))
    if not d.is_dir():
        sys.exit(f"no Claude Code history for {pathlib.Path(a.project).resolve()}\n"
                 f"  (looked in {d})")

    if a.session:
        picked = [d / a.session]
        if not picked[0].is_dir():
            sys.exit(f"no such session: {picked[0]}")
    else:
        found = sessions(d)
        if not found:
            sys.exit(f"no sessions with subagents under {d}\n"
                     f"  dispatch an agent first, then re-run this.")
        picked = found if a.all else found[:1]

    total = sum(report(s) for s in picked)
    print(f"\n{total} subagent(s) across {len(picked)} session(s)")


if __name__ == "__main__":
    main()
