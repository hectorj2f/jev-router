#!/usr/bin/env python3
"""Does feeding session context into `state` help the router, or bury the task?

Two opposing hypotheses, both testable:

  CONTAMINATION -- a trivial task surrounded by hard conversation reads as hard,
  because the questions ask about "this task" and Jev has to find it first.

  ENRICHMENT -- a task whose difficulty lives behind a reference ("review this
  PR") reads correctly once the reference is resolved into the state.

If contamination is real, bulk context is the wrong instrument regardless of
cost. If enrichment is real at a few hundred tokens, it is the cheap fix.

All filler here is synthetic. Nothing from a real conversation or repo.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import policy  # noqa: E402

TRIVIAL = "Report the value of the maxRetries constant in internal/worker/retry.go."

# A terse pointer whose real weight is invisible from the text.
OPAQUE = "Review PR #4471 and tell me whether it is safe to merge."

HARD_TALK = """\
user: The epoch rollover is still wrong when a tier lands mid-window. I think
the projection needs to be rebuilt from the change log rather than patched.
assistant: Agreed, but rebuilding on every landing is O(n) in the log. The
alternative is a watermark per tier, which breaks the single-writer invariant
the compactor relies on. We would need to weaken that to a per-tier writer and
prove the compactor still converges.
user: What does that do to the backfill path?
assistant: Backfill would have to take the same per-tier lock, which serialises
it against live landings. On the largest lines that is a multi-hour stall. We
could shard the lock by epoch, but then a rollover straddling two epochs can
interleave and we are back to the original bug.
"""

# A few hundred tokens of resolved facts about the reference, not a transcript.
PR_FACTS = """\

Referenced PR #4471:
  title: Rework tier landing to rebuild the projection from the change log
  files changed: 31, +2847 -1193
  touches: internal/store/projection.go, internal/store/compactor.go,
           internal/worker/backfill.go, 9 test files, 3 migration files
  checks: 2 failing (integration-postgres, e2e-landing)
"""


def row(label: str, state: str) -> None:
    d = policy.route(state, default="opus")
    s = d["signals"]
    print(f"  {label:<34} {d['model']:<7} need {d['need']:>6.2f}  conf {d['conf']:>4.2f}  "
          f"depth {s['depth']:>4}  breadth {s['breadth']:>4}  "
          f"[{len(state):>6} chars]")


def main() -> None:
    print("\nCONTAMINATION -- a trivial task, with increasing unrelated hard context")
    print(f"  task: {TRIVIAL}")
    row("bare", TRIVIAL)
    for mult in (1, 4, 16):
        ctx = HARD_TALK * mult
        row(f"+{mult}x hard conversation", f"{ctx}\nuser: {TRIVIAL}")

    print("\nENRICHMENT -- an opaque pointer, with the reference resolved")
    print(f"  task: {OPAQUE}")
    row("bare", OPAQUE)
    row("+resolved PR facts", OPAQUE + PR_FACTS)
    row("+hard conversation instead", f"{HARD_TALK}\nuser: {OPAQUE}")


if __name__ == "__main__":
    main()
