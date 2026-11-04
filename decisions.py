#!/usr/bin/env python3
"""The decision log: what the router decided, and what it was told.

A routing decision is invisible by construction. It happens before the agent
starts, the only trace it leaves is a model id on a transcript nobody reads,
and by the time the work comes back wrong the reasoning is gone. That makes
the one question worth asking about a router -- was it right? -- the one
question it cannot currently answer about itself.

Every front end appends here, so an ordinary session turns into a measurement
instead of an anecdote. explain.py reads it back; verify.py already knows what
each subagent actually ran on, and this is the other half of that join.

JSONL, append-only, because the writers are concurrent: route.py fans out
across twelve threads and Claude Code can start several dispatches at once,
each its own hook process. Read-modify-write on a shared JSON array loses
records under exactly the fan-out this repo exists to route.

One append write() is atomic against other appenders here because XNU holds
the vnode lock across the whole call, so record size is not the constraint --
measured clean at 48 KB a line, 16 processes, which a pipe's PIPE_BUF would
have torn long before. That is a kernel and filesystem property rather than a
POSIX guarantee: a log kept on NFS or SMB would need real locking.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy  # noqa: E402

# Overridable so selftest.py can exercise the log without appending to the one
# you are about to read back. A hook runs in its own process, so the seam has
# to be an environment variable rather than an argument.
LOG_DIR = pathlib.Path(os.environ.get("JEV_DECISIONS_DIR")
                       or policy.CACHE_DIR / "decisions")
RETENTION = 7 * 24 * 3600

# Enough to read the dispatch back and see what was scored, not enough to
# replay it byte for byte. A dispatch prompt runs to the 60k state limit and
# fifty of those is a log nobody opens twice. The original length is recorded
# alongside, so a short task and a cut one stay distinguishable.
TASK_LIMIT = 4_000

# The session id arrives from outside and lands in a filename.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

_pruned = False


def path_for(session: str) -> pathlib.Path:
    name = _UNSAFE.sub("-", str(session)).strip("-.")[:120] or "unknown"
    return LOG_DIR / f"{name}.jsonl"


def record(session: str, label: str, default: str, decision: dict,
           task: str = "") -> None:
    """Append one decision, including the ones that changed nothing.

    Holds are the whole point of logging this. Both front ends return early
    when the decision matches the tier already in use, so the asymmetry doing
    its job has always looked identical to the router not running.

    Never raises. It is called from a path that fails open, and a log that can
    take down a dispatch is worse than no log.
    """
    if not session:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"at": time.time(), "label": label, "default": default,
                 "task": task[:TASK_LIMIT], "task_len": len(task), **decision}
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        p = path_for(session)
        # Claim the file before appending to it. _prune() in a sibling hook
        # process decides by mtime and open(..., "a") does not touch it, so
        # resuming a session whose log has gone stale leaves a window where
        # the log is unlinked between this open and this write -- the record
        # then lands on a detached inode and is gone without an error.
        p.touch(exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
        if not p.exists():
            # Lost it anyway: the prune had already stat'd before the touch.
            # The retry races nothing, since _prune() runs once per process.
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        _prune()
    except Exception:
        pass


def read(session: str, limit: int | None = None) -> list[dict]:
    """One session's decisions, oldest first."""
    try:
        text = path_for(session).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line from a killed writer costs one record, not the log
    return out[-limit:] if limit else out


def sessions() -> list[dict]:
    """Every session with a log, most recently written first.

    The hook front end has no way to tell you its session id, so `explain.py`
    with no argument has to be able to find the newest one by itself.
    """
    out = []
    for p in LOG_DIR.glob("*.jsonl"):
        try:
            out.append({"session": p.stem, "at": p.stat().st_mtime,
                        "count": sum(1 for _ in p.open(encoding="utf-8",
                                                       errors="replace"))})
        except OSError:
            continue
    return sorted(out, key=lambda s: s["at"], reverse=True)


def _prune() -> None:
    """Drop logs past the retention window, once per process.

    Per write would mean a directory scan per shard of a fan-out, to delete
    something that has been expired for days either way.
    """
    global _pruned
    if _pruned:
        return
    _pruned = True

    cutoff = time.time() - RETENTION
    for p in LOG_DIR.glob("*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass
