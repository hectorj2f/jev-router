#!/usr/bin/env python3
"""Route a fan-out: pick a model and effort per shard, using Jev.

Reads a JSON array of shards on stdin, each {"id": "...", "task": "..."}, and
writes the same array back with "model", "effort", and the raw signals added.
Feed the result to a Workflow script as `args`.

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--default", default="sonnet", choices=policy.TIERS,
                   help="the model the fan-out would otherwise use")
    p.add_argument("--table", action="store_true", help="human-readable output")
    p.add_argument("--no-cache", action="store_true", help="bypass the disk cache")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    shards = json.load(sys.stdin)
    if not isinstance(shards, list) or not all("task" in s for s in shards):
        sys.exit("stdin must be a JSON array of objects, each with a 'task' key")
    for i, s in enumerate(shards):
        s.setdefault("id", f"shard-{i}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        routed = list(pool.map(
            lambda s: route_one(s, args.default, not args.no_cache), shards))

    if not args.table:
        json.dump(routed, sys.stdout, indent=2)
        print()
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
