#!/usr/bin/env python3
"""Report the model each Codex subagent actually ran on, against the decision.

verify.py's sibling, and the same question: a hook rewriting `model` on a
`spawn_agent` call proves only that the hook ran. Codex opens a whole rollout
per spawned agent rather than a subdirectory under the parent's, so the
evidence is a separate file:

    ~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-<started>-<thread-id>.jsonl

Each one is JSONL, one `{timestamp, ordinal, type, payload}` record per line.
`turn_context.model` is the model in force for a turn and is plaintext, which
matters because the spawn message and the assistant's reasoning next to it are
Fernet-encrypted and read back as `gAAAAAB...`.

    ./verify_codex.py                 # newest session with Codex decisions
    ./verify_codex.py --session <id>
    ./verify_codex.py --all           # every such session, newest first
    ./verify_codex.py --codex-home <path>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decisions  # noqa: E402
from providers import codex  # noqa: E402

CODEX_HOME = pathlib.Path(os.environ.get("CODEX_HOME")
                          or pathlib.Path.home() / ".codex")

# All three rungs share the family prefix, so printing it on every row says
# nothing. verify.py drops "claude-" for the same reason.
FAMILY = "gpt-"

# A spawn lands within seconds of the hook that routed it, but the two clocks
# are the decision log's wall time and the rollout's own `timestamp`, so give
# the pairing a window rather than demanding an ordering.
SKEW = 120.0

# How far either side of a session's decisions to look when nothing joined and
# the report has to say what it did find instead.
NEARBY = 3600.0

# A spawn nothing decided for is worth listing, but one parent thread here had
# 337 of them against a decision log that keeps seven days -- unbounded, those
# rows bury the join they are meant to annotate.
UNROUTED_SHOWN = 10


def rung(model: str) -> str:
    """A slug as a row prints it: the rung if it is one, the slug either way."""
    tier = codex.tier_of(model)
    short = model[len(FAMILY):] if model.startswith(FAMILY) else model
    return f"{short} ({tier})" if tier else short


def first_meta(path: pathlib.Path) -> dict | None:
    """The rollout's own session_meta, reading no further than it.

    A thread forked from its parent replays the parent's session_meta as well,
    second; the first is always the file's own -- its `id` matched the thread
    id in the filename in all 442 rollouts measured here. Stopping there is
    also what keeps indexing off the 1.8G of transcript behind these files:
    0.3s to index the lot, against 5.4s to read them through.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"session_meta"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") == "session_meta":
                    return e.get("payload") or {}
    except OSError:
        return None
    return None


def spawn_of(meta: dict) -> dict | None:
    """What spawned this thread, or None if a person did.

    `source` is a bare string ("cli", "vscode") for a thread someone started
    and an object for a spawned one. The object nests under the spawn kind,
    and `thread_spawn` is the only kind in 442 rollouts -- so take whichever
    single key is present instead of naming it, because a new kind should
    still attribute rather than read as an unknown shape. 0.160.0 began
    mirroring the same fields at the top of the payload; that is the fallback,
    not the primary, since the nested form is the one every version writes.
    """
    src = meta.get("source")
    sub = src.get("subagent") if isinstance(src, dict) else None
    if isinstance(sub, dict):
        for body in sub.values():
            if isinstance(body, dict) and body.get("agent_path"):
                return body
    if meta.get("thread_source") == "subagent" and meta.get("agent_path"):
        return {"agent_path": meta["agent_path"],
                "parent_thread_id": meta.get("parent_thread_id"),
                "agent_nickname": meta.get("agent_nickname")}
    return None


def started(meta: dict, path: pathlib.Path) -> float:
    ts = meta.get("timestamp") or ""
    try:
        return datetime.datetime.fromisoformat(
            ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    try:
        # The name carries a local-time stamp and the payload a UTC one, so
        # mtime is the safer of the two fallbacks.
        return path.stat().st_mtime
    except OSError:
        return 0.0


def index(home: pathlib.Path) -> tuple[dict[str, list[dict]], list[dict]]:
    """Every spawned thread on disk by the task_name it was spawned under.

    task_name is the join key because it is the one thing the router sees at
    dispatch that survives into the rollout: the last segment of `agent_path`
    is the `task_name` of the `spawn_agent` call that produced the thread, in
    all 438 spawned threads here. The same measurement is what says the model
    read below is worth reading at all -- of those, the 116 whose spawn named
    a model landed on exactly the model it named, none disagreeing.
    """
    by_task: dict[str, list[dict]] = {}
    broken: list[dict] = []

    for path in sorted((home / "sessions").rglob("rollout-*.jsonl")):
        try:
            if path.stat().st_size == 0:
                continue  # an aborted start, not a format this cannot read
        except OSError:
            continue
        meta = first_meta(path)
        if meta is None:
            broken.append({"path": path, "why": "no session_meta record",
                           "version": None})
            continue
        spawn = spawn_of(meta)
        if spawn is None:
            if meta.get("thread_source") in (None, "subagent"):
                broken.append({"path": path, "why": "no subagent attribution",
                               "version": meta.get("cli_version")})
            continue  # a thread a person started: nothing was routed into it
        task = (spawn.get("agent_path") or "").rsplit("/", 1)[-1]
        if not task:
            broken.append({"path": path, "why": "empty agent_path",
                           "version": meta.get("cli_version")})
            continue
        by_task.setdefault(task, []).append({
            "path": path, "task": task, "at": started(meta, path),
            "parent": spawn.get("parent_thread_id") or meta.get("parent_thread_id"),
            "nickname": spawn.get("agent_nickname"),
            "version": meta.get("cli_version"), "cwd": meta.get("cwd"),
        })
    return by_task, broken


def models_in(path: pathlib.Path) -> tuple[list[str], int]:
    """Models this thread ran turns on, in order, and how many turns it took.

    Last wins at the call site: a full-history fork replays the parent's
    turn_contexts ahead of its own, so the first model in the file can be the
    parent's rather than this agent's. Consecutive repeats collapse because a
    turn_context is written per turn and the interesting event is the change.
    """
    seen: list[str] = []
    turns = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"turn_context"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") != "turn_context":
                    continue
                p = e.get("payload") or {}
                turns += 1
                settings = (p.get("collaboration_mode") or {}).get("settings") or {}
                m = p.get("model") or settings.get("model")
                if m and (not seen or seen[-1] != m):
                    seen.append(m)
    except OSError:
        pass
    return seen, turns


def decided(rec: dict) -> str | None:
    """The rung the router picked, however the front end spelled it.

    `decide()` returns a tier name, but the slug is what goes on the wire and a
    front end is free to log that instead; neither spelling should decide
    whether a decision is a Codex one.
    """
    m = rec.get("model") or ""
    return m if m in codex.TIERS else codex.tier_of(m)


def pick(cands: list[dict] | None, at: float, used: set) -> dict | None:
    """The spawn a decision produced, when a task_name has been reused."""
    free = [c for c in (cands or []) if id(c) not in used]
    if not free:
        return None
    after = [c for c in free if c["at"] >= at - SKEW]
    return min(after or free, key=lambda c: abs(c["at"] - at))


def rows_for(recs: list[dict], by_task: dict[str, list[dict]]) -> list[dict]:
    """One row per Codex decision, plus one per spawn nothing decided for."""
    rows, used, parents = [], set(), set()

    for rec in recs:
        want = decided(rec)
        if want is None:
            continue  # another provider's decision in the same log
        label = (rec.get("label") or "").strip()
        roll = pick(by_task.get(label), rec.get("at", 0.0), used)
        if roll is None:
            rows.append({"label": label, "want": want, "verdict": "NO TURN",
                         "detail": "no spawned thread under this task_name"})
            continue

        used.add(id(roll))
        parents.add(roll["parent"])
        models, turns = models_in(roll["path"])
        if not models:
            why = (f"no model on any of {turns} turn_context record(s)"
                   if turns else "no turn_context record at all")
            rows.append({"label": label, "want": want, "verdict": "UNREADABLE",
                         "detail": f"{why}, {roll['path'].name}"})
            continue

        got = models[-1]
        switched = (f"  [switched: {' -> '.join(rung(m) for m in models)}]"
                    if len(models) > 1 else "")
        tier = codex.tier_of(got)
        if tier is None:
            verdict, detail = "OFF LADDER", f"ran {rung(got)}, not a rung"
        elif tier == want:
            verdict, detail = "ok", rung(got)
        else:
            verdict, detail = "OVERRIDDEN", f"ran {rung(got)}"
        rows.append({"label": label, "want": want, "verdict": verdict,
                     "detail": detail + switched})

    # A spawn under the same parent that no decision claimed. The hook declines
    # full-history forks because Codex refuses a model override on them, so
    # some of these are the router correctly standing aside -- which is still
    # worth seeing, because it is also what a hook that never fired looks like.
    if not recs:
        return rows
    lo = min(r.get("at", 0.0) for r in recs) - SKEW
    hi = max(r.get("at", 0.0) for r in recs) + NEARBY
    loose = sorted((c for cands in by_task.values() for c in cands
                    if id(c) not in used and c["parent"] in parents
                    and lo <= c["at"] <= hi), key=lambda c: c["at"])
    for c in loose[:UNROUTED_SHOWN]:
        models, _ = models_in(c["path"])
        rows.append({"label": c["task"], "want": "--", "verdict": "UNROUTED",
                     "detail": f"ran {rung(models[-1])}" if models
                               else "no model recorded"})
    if len(loose) > UNROUTED_SHOWN:
        rows.append({"label": f"... {len(loose) - UNROUTED_SHOWN} more",
                     "want": "--", "verdict": "+ more",
                     "detail": "unrouted spawns on the same thread, not listed"})
    return rows


def nearby(by_task: dict[str, list[dict]], lo: float, hi: float) -> list[str]:
    return sorted({c["task"] for cands in by_task.values() for c in cands
                   if lo - NEARBY <= c["at"] <= hi + NEARBY})


def report(session: str, by_task: dict[str, list[dict]]) -> list[dict]:
    recs = decisions.read(session)
    rows = rows_for(recs, by_task)
    print(f"\nsession {session}")
    for r in rows:
        print(f"  {r['label'][:44]:<46} {r['want']:<6} {r['verdict']:<11} "
              f"{r['detail']}")

    if rows and all(r["verdict"] == "NO TURN" for r in rows):
        # Never let a total miss read as a clean sheet. The likeliest cause is
        # that the front end logged something other than the spawn's task_name
        # as the label, which is the only thing this join has to go on.
        lo = min(r.get("at", 0.0) for r in recs)
        hi = max(r.get("at", 0.0) for r in recs)
        found = nearby(by_task, lo, hi)
        print(f"\n  Nothing here joined. The key is the spawn's task_name, "
              f"matched against\n  the decision's label. Labels recorded: "
              f"{', '.join(sorted({r['label'] for r in rows}))[:120]}")
        print(f"  task_names spawned around the same time: "
              f"{', '.join(found)[:200] or '(none)'}")
    return rows


def tally(rows: list[dict], verdict: str) -> int:
    return sum(1 for r in rows if r["verdict"] == verdict)


def main() -> None:
    p = argparse.ArgumentParser(
        description="what Codex actually ran each routed subagent on")
    p.add_argument("--session", help="session id, instead of the most recent")
    p.add_argument("--all", action="store_true",
                   help="every session with Codex decisions, newest first")
    p.add_argument("--codex-home", default=str(CODEX_HOME),
                   help=f"Codex state directory (default: {CODEX_HOME})")
    a = p.parse_args()

    home = pathlib.Path(a.codex_home).expanduser()
    if not (home / "sessions").is_dir():
        sys.exit(f"no Codex rollouts under {home / 'sessions'}\n"
                 f"  Codex writes one per spawned agent; run a session first, "
                 f"or pass --codex-home.")

    by_task, broken = index(home)
    known = [s["session"] for s in decisions.sessions()
             if any(decided(r) for r in decisions.read(s["session"]))]

    if not by_task:
        print(f"No spawned Codex threads found under {home / 'sessions'}.")
        if broken:
            # The dangerous case: rollouts are there and none of them parsed.
            # Reporting zero dispatches would be indistinguishable from a clean
            # run, so refuse instead.
            print(f"{len(broken)} rollout(s) are in a shape this does not read:")
            for b in broken[:10]:
                print(f"  {b['path'].name}  codex-cli "
                      f"{b['version'] or '(unknown version)'}  -- {b['why']}")
            sys.exit("The rollout format changed. Nothing here is verified.")
        if known:
            sys.exit(f"{len(known)} session(s) recorded Codex decisions and no "
                     f"spawn reached disk.\n  Every one of those dispatches is "
                     f"unverified, not confirmed.")
        return

    if not known:
        print("No Codex routing decisions recorded. The Codex front end writes\n"
              "one entry per spawn_agent dispatch, so this fills up on its own\n"
              "once it is installed; there is nothing to join against yet.")
        print(f"\n{len(by_task)} spawned thread(s) are on disk and readable, so "
              f"the rollout side\nof the join is working.")
        return

    if a.session:
        if a.session not in known:
            sys.exit(f"no Codex decisions recorded for session {a.session}\n"
                     f"  sessions with any: {', '.join(known) or '(none)'}")
        picked = [a.session]
    else:
        picked = known if a.all else known[:1]

    rows = [r for s in picked for r in report(s, by_task)]
    over, missing = tally(rows, "OVERRIDDEN"), tally(rows, "NO TURN")
    off, unread = tally(rows, "OFF LADDER"), tally(rows, "UNREADABLE")
    loose = tally(rows, "UNROUTED")
    routed = sum(1 for r in rows if r["want"] != "--")
    parts = [f"{routed} routed dispatch(es) across {len(picked)} session(s)"]
    for n, what in ((over, "overridden"), (missing, "with no turn"),
                    (off, "off the ladder"), (unread, "unreadable"),
                    (loose, "spawns nothing decided for")):
        if n:
            parts.append(f"{n} {what}")
    print("\n" + ", ".join(parts))

    if over:
        print("A turn ran on a model the router did not pick, so something "
              "downstream\nof the hook chose it. `agents.<name>."
              "default_subagent_model` in\n~/.codex/config.toml is the known "
              "candidate, and it bites hardest on a\nheld decision, where the "
              "hook deliberately emits nothing and Codex is\nleft to resolve "
              "the model itself.")
    if missing:
        print("A decision with no turn means the spawn never happened or Codex "
              "refused\nthe override -- it refuses one on a full-history fork, "
              "which the hook is\nsupposed to decline before deciding at all.")
    if off:
        print("A turn off the ladder is not an error: gpt-6.1-terra is a "
              "specialist the\nrouter will not pick. It does mean the spend on "
              "that turn is unbudgeted.")
    if unread:
        print("An unreadable rollout is a format change, not a routing result. "
              "Treat\nthose rows as unknown rather than as passes.")
    if broken:
        vers = sorted({b["version"] or "unknown" for b in broken})
        print(f"\n{len(broken)} rollout(s) skipped as unrecognised "
              f"(codex-cli {', '.join(vers)}):")
        for b in broken[:5]:
            print(f"  {b['path'].name}  -- {b['why']}")

    # Nothing confirmed is a result in its own right, and the one this must
    # never let pass for a clean run.
    if routed and not tally(rows, "ok"):
        sys.exit("Not one dispatch was confirmed to have run on the model it "
                 "was routed to.")


if __name__ == "__main__":
    main()
