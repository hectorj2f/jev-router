#!/usr/bin/env python3
"""Route described tasks (fan-out shards) through the new policy.

The session-prompt run says nothing reaches haiku. This asks whether that is the
policy or the prompts: same policy, but tasks that state what the work involves
rather than pointing at it.
"""

from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import policy  # noqa: E402

shards = json.load(open(sys.argv[1]))
with ThreadPoolExecutor(max_workers=10) as pool:
    answers = list(pool.map(policy.ask, [s["task"] for s in shards]))

hdr = f"{'shard':<16} {'model':<7} {'need':>5} {'mech':>5} {'conf':>5}  advice"
print(hdr)
print("-" * len(hdr))
tally: dict[str, int] = {}
for s, ans in zip(shards, answers):
    d = policy.decide(ans)
    tally[d["model"]] = tally.get(d["model"], 0) + 1
    held = f"  (held from {d['suggested']})" if d["held"] else ""
    advice = " ".join(k for k, v in d["advice"].items() if v) or "-"
    print(f"{s['id']:<16} {d['model']:<7} {d['need']:>5.2f} "
          f"{d['signals']['mechanical']:>5.2f} {d['conf']:>5.2f}  {advice}{held}")

print("\n" + ", ".join(f"{tally.get(t, 0)}x{t}" for t in policy.TIERS))
