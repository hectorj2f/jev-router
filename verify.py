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
import re
import sys

PROJECTS = pathlib.Path.home() / ".claude" / "projects"

# A dispatch that died before its first real turn records `<synthetic>` as the
# model and carries the failure in the message text. That case matters more
# here than any other -- a router that picks a model the endpoint cannot serve
# kills the agent, and the symptom is a transcript with no model in it.
SYNTHETIC = "<synthetic>"
FAILED_MODEL = re.compile(r"selected model \(([^)]+)\)")


def slug(path: pathlib.Path) -> str:
    """Claude Code's project directory name: the abs path, non-alnum to dashes."""
    return "".join(c if c.isalnum() else "-" for c in str(path.resolve()))


def sessions(project_dir: pathlib.Path) -> list[pathlib.Path]:
    """Session dirs that actually contain subagents, newest first."""
    out = [d for d in project_dir.iterdir() if d.is_dir() and (d / "subagents").is_dir()]
    return sorted(out, key=lambda d: (d / "subagents").stat().st_mtime, reverse=True)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def models_in(jsonl: pathlib.Path) -> tuple[list[str], str | None]:
    """Models this agent ran on, and why it died if it did."""
    seen, failure = [], None
    for line in jsonl.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "assistant":
            continue
        if e.get("isApiErrorMessage"):
            body = text_of(e.get("message", {}).get("content"))
            hit = FAILED_MODEL.search(body)
            failure = f"{hit.group(1)} unavailable" if hit else body[:70].strip()
            continue
        m = e.get("message", {}).get("model")
        if m and m != SYNTHETIC and m not in seen:
            seen.append(m)
    return seen, failure


def report(session: pathlib.Path) -> tuple[int, int]:
    subs = sorted((session / "subagents").glob("agent-*.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not subs:
        return 0, 0

    print(f"\nsession {session.name}")
    failed, last_failed = 0, False
    for j in subs:
        meta_path = j.with_suffix(".meta.json")
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                pass
        desc = (meta.get("description") or meta.get("name") or j.stem)[:52]
        models, failure = models_in(j)
        last_failed = bool(failure)
        if failure:
            failed += 1
            print(f"  {desc:<54} FAILED  {failure}")
            continue
        # More than one model in a single subagent means a mid-run switch, which
        # the hook cannot do -- worth seeing rather than collapsing.
        shown = ", ".join(m.replace("claude-", "") for m in models)
        print(f"  {desc:<54} {shown or '(no assistant turn)'}")
    return len(subs), failed, last_failed


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

    counts = [report(s) for s in picked]
    total = sum(n for n, _, _ in counts)
    failed = sum(f for _, f, _ in counts)
    print(f"\n{total} subagent(s) across {len(picked)} session(s)"
          + (f", {failed} failed" if failed else ""))
    # Only chase a failure that is still the latest word. Transcripts are
    # permanent, so a session that hit an unavailable model once and was then
    # fixed would otherwise print this advice forever.
    if failed and any(last for _, _, last in counts):
        print("The most recent dispatch failed on an unavailable model, so it was routed\n"
              "to a tier this endpoint does not serve. Run ./probe.py from a Claude Code\n"
              "tool context to record what it does.")
    elif failed:
        print("Those failures predate the newest dispatch, which succeeded -- most likely\n"
              "already fixed. Transcripts are permanent, so they stay listed.")


if __name__ == "__main__":
    main()
