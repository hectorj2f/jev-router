#!/usr/bin/env python3
"""Replay the router's decisions for a session.

    ./explain.py                  # this session's decisions, as a table
    ./explain.py 3                # everything behind decision 3
    ./explain.py --last           # everything behind the most recent one
    ./explain.py --session <id>   # a session other than the most recent
    ./explain.py --sessions       # what has been recorded, and when
    ./explain.py --json

The table is the default view because a dispatch router decides many times in
a session, and a wrong decision is visible as the row that looks unlike its
neighbours -- one shard on opus at confidence 0.2, or six in a row held. One
decision shown alone has nothing to look unlike.

Nothing here renders anything: `route.table` and `route.detail` already draw a
decision, and reusing them is what keeps the replay honest. A second renderer
would be free to drift from the one that runs during a fan-out, and the whole
point of reading this back is that it shows what actually happened.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decisions  # noqa: E402
import route  # noqa: E402

# A dispatch prompt runs to tens of kilobytes. Enough to recognise which task
# this was, not the whole thing again.
TASK_SHOWN = 800


def ago(ts: float) -> str:
    s = max(0, int(time.time() - ts))
    for limit, div, unit in ((60, 1, "s"), (3600, 60, "m"), (86400, 3600, "h")):
        if s < limit:
            return f"{s // div}{unit} ago"
    return f"{s // 86400}d ago"


def clipped(rec: dict) -> dict:
    """The recorded task, cut to what a terminal can hold.

    The original length travels with the record, so a task that was short and
    one that was cut twice -- here and at the log's own limit -- stay apart.
    """
    task = rec.get("task") or ""
    full = rec.get("task_len", len(task))
    cut = task[:TASK_SHOWN].rstrip() if len(task) > TASK_SHOWN else task
    if len(cut) < full:
        cut += f" ... [{full} chars in all]"
    return {**rec, "task": cut}


def list_sessions(known: list[dict], as_json: bool) -> None:
    if as_json:
        json.dump(known, sys.stdout, indent=2)
        print()
        return
    hdr = f"{'session':<40} {'decisions':>9}  last"
    print(hdr)
    print("-" * len(hdr))
    for s in known:
        print(f"{s['session'][:40]:<40} {s['count']:>9}  {ago(s['at'])}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="replay what the router decided, and why")
    p.add_argument("index", nargs="?", type=int,
                   help="show everything behind one decision, by its number")
    p.add_argument("--session", default="",
                   help="a session other than the most recent")
    p.add_argument("--sessions", action="store_true",
                   help="list every session with a log")
    p.add_argument("--last", action="store_true",
                   help="everything behind the most recent decision")
    p.add_argument("--json", action="store_true", help="the records themselves")
    args = p.parse_args()

    known = decisions.sessions()
    if not known:
        print("Nothing recorded yet. Either front end writes one entry per\n"
              "subagent dispatch, so this fills up on its own; by hand, pass\n"
              "./route.py a --session to write under.")
        return

    if args.sessions:
        list_sessions(known, args.json)
        return

    # Newest by write time, because the shell has no way to know which session
    # it was run from. The mod passes --session and never relies on this.
    session = args.session or known[0]["session"]
    recs = decisions.read(session)
    if not recs:
        print(f"no decisions recorded for session {session}")
        return

    if args.json:
        json.dump(recs, sys.stdout, indent=2)
        print()
        return

    idx = len(recs) if args.last else args.index
    if idx is not None:
        if not 1 <= idx <= len(recs):
            sys.exit(f"this session has {len(recs)} decisions, numbered 1 to "
                     f"{len(recs)}")
        rec = recs[idx - 1]
        print(f"\n  #{idx}  {rec['label']}  --  {ago(rec['at'])}, "
              f"dispatched from {rec['default']}")
        route.detail(clipped(rec), rec["default"])
        return

    span = (ago(recs[0]["at"]) if len(recs) == 1
            else f"{ago(recs[0]['at'])} to {ago(recs[-1]['at'])}")
    print(f"session {session}: {len(recs)} decisions, {span}\n")
    route.table([{**r, "id": f"{i}  {r['label']}"}
                 for i, r in enumerate(recs, 1)], column="decision")
    print("\n./explain.py <n> for everything behind one of them")


if __name__ == "__main__":
    main()
