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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decisions  # noqa: E402

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


# Where the hook subprocesses write their decisions. Set in main(), once there
# is a temp dir to point at: the one thing a self-test must not do is append a
# hundred synthetic decisions to the log you read with ./explain.py.
LOGDIR = ""


def run(event, env: dict | None = None) -> tuple[int, str]:
    payload = event if isinstance(event, str) else json.dumps(event)
    r = subprocess.run([sys.executable, str(HOOK)], input=payload,
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "JEV_DECISIONS_DIR": LOGDIR,
                            **(env or {})})
    return r.returncode, r.stdout.strip()


def routed_model(out: str) -> str | None:
    if not out:
        return None
    return json.loads(out)["hookSpecificOutput"]["updatedInput"].get("model")


def main() -> int:
    global LOGDIR
    tmp = pathlib.Path(tempfile.mkdtemp())
    LOGDIR = str(tmp / "hooklog")
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
    #
    # JEV_TIERS is pinned for these two so they test the policy rather than the
    # machine: on an endpoint serving only one tier the hook correctly routes
    # nothing, and a green/red signal that flips with your Vertex grants is
    # testing the wrong thing.
    FULL = {"JEV_TIERS": "haiku,sonnet,opus"}

    code, out = run(ti(subagent_type="general-purpose",
                       description="gofmt a file", prompt=MECHANICAL), env=FULL)
    m = routed_model(out)
    check("mechanical task leaves opus", code == 0 and m is not None and m != "opus",
          f"routed to {m!r}")

    code, out = run(ti(subagent_type="general-purpose",
                       description="design epoch rollover", prompt=DEEP), env=FULL)
    check("deep task stays on opus", code == 0 and out == "",
          f"rewrote to {routed_model(out)!r}" if out else "")

    # A one-tier ladder has nothing to route between, and must not emit the one
    # tier it has as though that were a decision.
    code, out = run(ti(subagent_type="general-purpose", description="gofmt a file",
                       prompt=MECHANICAL), env={"JEV_TIERS": "opus"})
    check("single-tier ladder routes nothing", code == 0 and out == "",
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
                       prompt=MECHANICAL, extra_key="preserve me"), env=FULL)
    ok = False
    if out:
        upd = json.loads(out)["hookSpecificOutput"]["updatedInput"]
        ok = upd.get("extra_key") == "preserve me" and upd.get("prompt") == MECHANICAL
    check("updatedInput preserves every original key", ok, f"got {out[:80]!r}")

    # The decision log, pointed at a temp dir. A self-test that appends to the
    # log it then reads back is testing the leftovers of the last run.
    logdir = tmp / "decisions"
    decisions.LOG_DIR = logdir
    plan = {"model": "sonnet", "effort": "medium", "need": 0.41, "conf": 0.82}

    decisions.record("sess-a", "rename Deadline", "opus", plan, DEEP)
    got = decisions.read("sess-a")
    check("a decision survives a round trip",
          len(got) == 1 and got[0]["model"] == "sonnet" and got[0]["task"] == DEEP
          and got[0]["label"] == "rename Deadline" and got[0]["default"] == "opus",
          f"read back {got!r:.80}")

    long_task = "x" * (decisions.TASK_LIMIT + 500)
    decisions.record("sess-a", "big", "opus", plan, long_task)
    got = decisions.read("sess-a")
    check("a long task is cut, and says how long it was",
          len(got) == 2 and len(got[1]["task"]) == decisions.TASK_LIMIT
          and got[1]["task_len"] == len(long_task),
          f"kept {len(got[-1]['task'])} of {len(long_task)}")

    # Appends rather than rewrites, because the writers are concurrent: a
    # fan-out is twelve threads, and Claude Code starts dispatches in parallel.
    check("records accumulate in order",
          [g["label"] for g in got] == ["rename Deadline", "big"],
          f"{[g['label'] for g in got]}")

    known = decisions.sessions()
    check("a session with a log can be found without being named",
          [s["session"] for s in known] == ["sess-a"] and known[0]["count"] == 2,
          f"{known!r:.80}")

    with decisions.path_for("sess-a").open("a", encoding="utf-8") as f:
        f.write('{"at": 1, "label": "killed mid-wri')
    check("a torn line costs one record, not the log",
          len(decisions.read("sess-a")) == 2,
          f"read {len(decisions.read('sess-a'))} records back")

    # The session id arrives from outside and lands in a filename.
    escape = decisions.path_for("../../etc/passwd")
    check("a session id cannot escape the log dir", escape.parent == logdir,
          f"wrote to {escape}")

    decisions.record("", "no session", "opus", plan, DEEP)
    check("no session id, no log", not (logdir / "unknown.jsonl").exists(),
          "an unattributed decision was written anyway")

    # The claim the log exists for: the hook returns early when the decision
    # matches the tier already in use, which is exactly when nothing else can
    # tell you whether the router ran.
    code, out = run({**ti(subagent_type="general-purpose",
                          description="design epoch rollover", prompt=DEEP),
                     "session_id": "sess-hook"}, env=FULL)
    decisions.LOG_DIR = pathlib.Path(LOGDIR)
    logged = decisions.read("sess-hook")
    check("a decision that changed nothing is still logged",
          code == 0 and out == "" and len(logged) == 1
          and logged[0]["model"] == "opus" and logged[0]["default"] == "opus",
          f"exit={code} out={out[:40]!r} logged {logged!r:.60}")

    # jev.py exits rather than raises, and SystemExit is not an Exception, so
    # the record above used to be unreachable on an outage -- which left a dead
    # API looking exactly like a router that never fired.
    code, out = run({**ti(subagent_type="general-purpose",
                          description="rename a variable", prompt=MECHANICAL),
                     "session_id": "sess-outage"},
                    env={"HOME": str(tmp / "keyless"), "TYPESAFE_API_KEY": ""})
    outage = decisions.read("sess-outage")
    check("a jev outage is logged, not swallowed",
          code == 0 and out == "" and len(outage) == 1
          and outage[0]["model"] == "opus"
          and outage[0].get("why", "").startswith("jev error:"),
          f"exit={code} out={out[:40]!r} logged {outage!r:.60}")

    # A sibling hook's _prune() decides by mtime, and open(..., "a") does not
    # touch it, so a resumed session whose log has gone stale can have it
    # unlinked between this process's open and its write. The writer below
    # pauses inside that window on purpose: without the claiming touch, its
    # record and the nine before it vanish with no error raised anywhere.
    stale = decisions.path_for("sess-stale")
    stale.write_text("".join(json.dumps({"n": i}) + "\n" for i in range(9)))
    os.utime(stale, (1, 1))
    here = str(HOOK.parent)
    writer = subprocess.Popen(
        [sys.executable, "-c",
         f"import builtins,sys,time;sys.path.insert(0,{here!r});"
         "real=builtins.open;"
         "builtins.open=lambda *a,**k: ((time.sleep(1.0), real(*a,**k))[1]"
         "  if str(a[0]).endswith('sess-stale.jsonl') else real(*a,**k));"
         "import decisions;"
         "decisions.record('sess-stale','l','opus',{'model':'opus'},'t')"],
        env={**os.environ, "JEV_DECISIONS_DIR": LOGDIR})
    subprocess.run([sys.executable, "-c",
                    f"import sys;sys.path.insert(0,{here!r});"
                    "import decisions;decisions._prune()"],
                   env={**os.environ, "JEV_DECISIONS_DIR": LOGDIR}, timeout=60)
    writer.wait(timeout=60)
    survived = decisions.read("sess-stale")
    check("a stale log survives a concurrent prune", len(survived) == 10,
          f"{len(survived)} records survived, expected 10")

    width = max(len(n) for n, _, _ in cases)
    failed = 0
    for name, ok, detail in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
