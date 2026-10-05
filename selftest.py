#!/usr/bin/env python3
"""Exercise hook_agent.py against synthetic PreToolUse events.

Checks the decision logic and every fail-open path without needing Claude Code
or a real dispatch. Does call Jev for the two routing cases, so it needs a key;
answers are cached, so a second run is free and deterministic.

    ./selftest.py          # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

HOOK = pathlib.Path(__file__).resolve().parent / "hook_agent.py"

MECHANICAL = ("Run gofmt on internal/store/release_load.go and commit the "
              "result.")
DEEP = ("Design the epoch-rollover scheme for the store's tiers, weighing the "
        "trade-offs against the existing landing projection.")


def fake_transcript(tmp: pathlib.Path, model: str = "claude-opus-5") -> str:
    """A transcript with one assistant turn, which is all session_tier reads.

    Named after the model: sharing one path meant the off-ladder case silently
    overwrote the opus transcript that later cases read, and two real assertions
    failed against a fable session they were never meant to see.
    """
    p = tmp / f"transcript-{model}.jsonl"
    p.write_text(json.dumps({"type": "assistant", "message": {"model": model}}) + "\n")
    return str(p)


def run(event, env: dict | None = None) -> tuple[int, str]:
    payload = event if isinstance(event, str) else json.dumps(event)
    r = subprocess.run([sys.executable, str(HOOK)], input=payload,
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, **(env or {})})
    return r.returncode, r.stdout.strip()


def routed_model(out: str) -> str | None:
    if not out:
        return None
    return json.loads(out)["hookSpecificOutput"]["updatedInput"].get("model")


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    t = fake_transcript(tmp)
    ti = lambda **kw: {"transcript_path": t, "tool_input": kw}  # noqa: E731

    cases = []

    def check(name, ok, detail=""):
        cases.append((name, ok, detail))

    # Fail-open paths: every one of these must leave the dispatch untouched.
    for name, ev in [
        ("fork is never rerouted", ti(subagent_type="fork", prompt=MECHANICAL)),
        ("explicit model wins", ti(model="opus", prompt=MECHANICAL)),
        ("unreadable transcript", {"transcript_path": "/nonexistent",
                                   "tool_input": {"prompt": MECHANICAL}}),
        ("missing tool_input", {"transcript_path": t}),
        ("empty object", {}),
    ]:
        code, out = run(ev)
        check(name, code == 0 and out == "", f"exit={code} out={out[:60]!r}")

    code, out = run("this is not json")
    check("malformed stdin", code == 0 and out == "", f"exit={code} out={out[:60]!r}")

    # Off-ladder session model: decline rather than guess.
    code, out = run({"transcript_path": fake_transcript(tmp, "claude-fable-5-1"),
                     "tool_input": {"prompt": MECHANICAL}})
    check("off-ladder session model declines", code == 0 and out == "",
          f"exit={code} out={out[:60]!r}")

    # Routing. Asserted as "not opus" / "stays opus" rather than an exact tier,
    # because Jev drifts by a few hundredths between calls and a task sitting on
    # a threshold can legitimately flip.
    code, out = run(ti(subagent_type="general-purpose",
                       description="gofmt a file", prompt=MECHANICAL))
    m = routed_model(out)
    check("mechanical task leaves opus", code == 0 and m is not None and m != "opus",
          f"routed to {m!r}")

    code, out = run(ti(subagent_type="general-purpose",
                       description="design epoch rollover", prompt=DEEP))
    check("deep task stays on opus", code == 0 and out == "",
          f"rewrote to {routed_model(out)!r}" if out else "")

    # A tier the endpoint cannot serve must never be emitted. Routing to a
    # missing model is strictly worse than not routing: the dispatch dies on a
    # 404 rather than running on the model it would otherwise have used.
    ev = ti(subagent_type="general-purpose", description="read one value",
            prompt="Report the value of the maxRetries constant in internal/worker/retry.go.")
    code, out = run(ev, env={"JEV_TIERS": "sonnet,opus"})
    m = routed_model(out)
    check("never routes to an unavailable tier", code == 0 and m in (None, "sonnet"),
          f"routed to {m!r} with haiku off the ladder")

    code, out = run(ev, env={"JEV_TIERS": "haiku,sonnet,opus"})
    check("routes to haiku when it is available", code == 0 and routed_model(out) == "haiku",
          f"routed to {routed_model(out)!r} with haiku on the ladder")

    # The whole tool_input must come back, since updatedInput replaces it.
    code, out = run(ti(subagent_type="general-purpose", description="gofmt a file",
                       prompt=MECHANICAL, extra_key="preserve me"))
    ok = False
    if out:
        upd = json.loads(out)["hookSpecificOutput"]["updatedInput"]
        ok = upd.get("extra_key") == "preserve me" and upd.get("prompt") == MECHANICAL
    check("updatedInput preserves every original key", ok, f"got {out[:80]!r}")

    width = max(len(n) for n, _, _ in cases)
    failed = 0
    for name, ok, detail in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
