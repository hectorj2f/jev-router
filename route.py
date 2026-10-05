#!/usr/bin/env python3
"""Route a fan-out: pick a model and effort per shard, using Jev.

Reads a JSON array of shards on stdin, each {"id": "...", "task": "..."}, and
writes the same array back with "model", "effort", and the raw signals added.
Feed the result to a Workflow script as `args`.

    ./route.py "rename the Deadline field and fix every caller"   # one task, explained
    ./route.py "task a" "task b"               # several, as a table
    ./route.py < shards.json > routed.json
    ./route.py --table < shards.json          # human-readable
    ./route.py --default opus --table < shards.json

One Jev request per shard, fired concurrently: `state` is per-shard, and a
single request shares one state across all its questions, so shards cannot be
batched into one call. At ~$0.00003 each and a 40 req/s ceiling this is free
and fast for any realistic fan-out.

The policy itself lives in policy.py -- this file only maps it over a list.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy  # noqa: E402


def route_one(shard: dict, default: str, cache: bool) -> dict:
    out = dict(shard)
    try:
        decision = policy.route(shard["task"], default=default, cache=cache)
    except SystemExit as e:
        # Fail open: an unrouted shard runs on the default, it does not fail the
        # fan-out. A router that can take the whole run down with it is worse
        # than no router.
        return {**out, "model": default, "effort": policy.EFFORT[default],
                "why": f"jev error: {e}"}
    return {**out, **decision}


def scale(need: float, width: int = 36) -> str:
    """Where `need` landed against the two tier lines.

    Worth drawing rather than printing bare: a decision 0.01 off a threshold and
    one 0.4 clear of it read identically as a number, and only the first is
    worth re-running to see whether Jev's drift flips it.
    """
    lo, hi = -0.25, 1.0
    at = lambda v: max(0, min(width - 1, round((v - lo) / (hi - lo) * (width - 1))))  # noqa: E731
    row = ["."] * width
    row[at(policy.SONNET_LINE)] = "|"
    row[at(policy.OPUS_LINE)] = "|"
    row[at(need)] = "#"
    return f"haiku {''.join(row)} opus"


def detail(r: dict, default: str) -> None:
    """Everything behind one decision -- the point of routing a task by hand."""
    print(f"\n  {r['task']}\n")
    if "why" in r:
        print(f"  {r['model']}, unrouted: {r['why']}\n")
        return

    effort = f"   effort {r['effort']}" if r["effort"] else ""
    print(f"  model        {r['model']}{effort}")
    if r["wanted"] != r["suggested"]:
        print(f"               {r['wanted']} is not served here, so {r['suggested']} is "
              f"the floor; ladder is {', '.join(r['tiers'])}")
    if r["held"]:
        print(f"               held at {default}; {r['suggested']} was suggested but "
              f"a downgrade needs mechanical > 0.75 or confidence >= 0.70")
    elif r["model"] != default:
        print(f"               {default} -> {r['model']}")

    print(f"  need         {r['need']:>5.2f}  {scale(r['need'])}")
    print(f"  confidence   {r['conf']:>5.2f}")

    print("\n  signals")
    for name in ("depth", "breadth"):
        v = r["signals"][name]
        criteria = policy.QUESTIONS[name]["criteria"]
        # Jev returns a weighted score across the levels, not a level, so the
        # nearest criterion is an approximation of a value that sits between two.
        print(f"    {name:<13}{v:>5.2f}/4  ~ {criteria[round(v)][:58]}")
    for name in ("mechanical", "irreversible", "ambiguous"):
        print(f"    {name:<13}{r['signals'][name]:>5.2f}")

    notes = [k for k, v in r["advice"].items() if v]
    if notes:
        says = {"confirm": "writes something hard to undo -- confirm before running",
                "clarify": "under-specified -- ask back rather than guess"}
        print("\n  advice")
        for n in notes:
            print(f"    {says[n]}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="*",
                   help="task text; omit to read a JSON array on stdin")
    p.add_argument("--default", default="sonnet", choices=policy.TIERS,
                   help="the model the fan-out would otherwise use")
    p.add_argument("--table", action="store_true", help="human-readable output")
    p.add_argument("--json", action="store_true", help="JSON, even for positional tasks")
    p.add_argument("--no-cache", action="store_true", help="bypass the disk cache")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    if args.task:
        shards = [{"id": f"task-{i}", "task": t} for i, t in enumerate(args.task)]
    else:
        if sys.stdin.isatty():
            sys.exit("give a task as an argument, or pipe a JSON array in. -h for both.")
        shards = json.load(sys.stdin)
        if not isinstance(shards, list) or not all("task" in s for s in shards):
            sys.exit("stdin must be a JSON array of objects, each with a 'task' key")
        for i, s in enumerate(shards):
            s.setdefault("id", f"shard-{i}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        routed = list(pool.map(
            lambda s: route_one(s, args.default, not args.no_cache), shards))

    # A single task typed by hand wants the reasoning; a pipe wants JSON.
    if args.json or not (args.table or args.task):
        json.dump(routed, sys.stdout, indent=2)
        print()
        return

    if len(routed) == 1 and not args.table:
        detail(routed[0], args.default)
        return

    hdr = f"{'shard':<28} {'model':<7} {'need':>5} {'conf':>5}  advice"
    print(hdr)
    print("-" * len(hdr))
    for r in routed:
        if "why" in r:
            print(f"{r['id'][:28]:<28} {r['model']:<7} {'--':>5} {'--':>5}  {r['why']}")
            continue
        notes = [k for k, v in r["advice"].items() if v]
        if r["held"]:
            notes.append(f"held from {r['suggested']}")
        print(f"{r['id'][:28]:<28} {r['model']:<7} {r['need']:>5.2f} {r['conf']:>5.2f}  "
              f"{', '.join(notes) or '-'}")

    tally = {t: sum(1 for r in routed if r["model"] == t) for t in policy.TIERS}
    print("\n" + ", ".join(f"{tally[t]}x{t}" for t in policy.TIERS))


if __name__ == "__main__":
    main()
