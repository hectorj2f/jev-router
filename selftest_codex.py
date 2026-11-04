#!/usr/bin/env python3
"""Exercise hook_codex.py against synthetic Codex PreToolUse events.

Checks the decision logic and every fail-open path without needing Codex or a
real spawn. Unlike selftest.py this calls Jev for nothing: the plan is pinned
per case, so the subject is the hook's reading of a Codex event rather than
Jev's reading of a task, and the run is free, offline and deterministic.

    ./selftest_codex.py    # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ["JEV_PROVIDER"] = "codex"
import decisions  # noqa: E402
from providers import codex  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
HOOK = HERE / "hook_codex.py"

MECHANICAL = ("Run gofmt on internal/store/release_load.go and commit the "
              "result.")
DEEP = ("Design the epoch-rollover scheme for the store's tiers, weighing the "
        "trade-offs against the existing landing projection.")

# The hook runs as its own process, the way Codex runs it, so the stub has to
# be installed from outside it. This imports the modules first and rebinds on
# them, then runs the hook by path: hook_codex.py's own `import policy` finds
# the patched module already in sys.modules.
DRIVER = '''
import json, os, pathlib, runpy, sys
sys.path.insert(0, os.environ["JEV_REPO"])
os.environ["JEV_PROVIDER"] = "codex"
import policy
from providers import codex
plan = json.loads(os.environ["JEV_FAKE_PLAN"])
policy.route = lambda state, default=None, cache=True: plan
codex.EFFORT.update(json.loads(os.environ.get("JEV_FAKE_EFFORT") or "{}"))
runpy.run_path(os.environ["JEV_HOOK"], run_name="__main__")
'''

PLAN = {"model": "sol", "effort": "medium", "need": 0.41, "conf": 0.82,
        "suggested": "sol", "held": False, "wanted": "sol"}

# Where the hook subprocesses write their decisions. Set in main(), once there
# is a temp dir to point at: the one thing a self-test must not do is append a
# hundred synthetic decisions to the log you read with ./explain.py.
LOGDIR = ""
DRIVER_PATH = ""


def run(event, plan: dict | None = None, effort: dict | None = None,
        env: dict | None = None) -> tuple[int, str]:
    payload = event if isinstance(event, str) else json.dumps(event)
    r = subprocess.run([sys.executable, DRIVER_PATH], input=payload,
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ,
                            "JEV_REPO": str(HERE),
                            "JEV_HOOK": str(HOOK),
                            "JEV_DECISIONS_DIR": LOGDIR,
                            "JEV_FAKE_PLAN": json.dumps(plan or PLAN),
                            "JEV_FAKE_EFFORT": json.dumps(effort or {}),
                            **(env or {})})
    return r.returncode, r.stdout.strip()


def updated(out: str) -> dict | None:
    if not out:
        return None
    return json.loads(out)["hookSpecificOutput"].get("updatedInput")


def event(**ti) -> dict:
    """A PreToolUse event with every field Codex marks required.

    Spelled out rather than trimmed to what the hook reads: the point of a
    synthetic event is that it is the one the engine would have sent.
    """
    ti.setdefault("fork_turns", "none")
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "sess-codex",
        "turn_id": "turn-1",
        "cwd": str(HERE),
        "model": "gpt-6.1-sol",
        "permission_mode": "default",
        "transcript_path": None,
        # The wire spelling, not the bare one. V2 registers the tool in a
        # namespace and only the default and V1 namespaces are mapped back, so
        # what a hook is handed is the two run together -- as all 443 real
        # spawns on the machine this was written on were. The fixture said
        # `spawn_agent` and so this suite passed, in full, against a hook that
        # could never once have fired.
        "tool_name": "collaborationspawn_agent",
        "tool_use_id": "call-1",
        "tool_input": ti,
    }


def spawn(**kw) -> dict:
    """The arguments a V2 spawn_agent call actually carries."""
    args = {"message": MECHANICAL, "task_name": "gofmt_release_load"}
    args.update(kw)
    return args


def main() -> int:
    global LOGDIR, DRIVER_PATH
    tmp = pathlib.Path(tempfile.mkdtemp())
    LOGDIR = str(tmp / "hooklog")
    DRIVER_PATH = str(tmp / "drive.py")
    pathlib.Path(DRIVER_PATH).write_text(DRIVER)

    cases = []

    def check(name, ok, detail=""):
        cases.append((name, ok, detail))

    # Fail-open paths: every one of these must leave the spawn untouched.
    for name, ev in [
        ("another tool is not ours", {**event(**spawn()), "tool_name": "shell"}),
        ("a tool merely containing the name is not ours",
         {**event(**spawn()), "tool_name": "spawn_agent_group"}),
        ("no tool_name at all", {k: v for k, v in event(**spawn()).items()
                                 if k != "tool_name"}),
        ("default fork_turns is a full-history fork",
         event(**{k: v for k, v in spawn().items() if k != "fork_turns"})),
        ("fork_turns=all is a full-history fork",
         event(**spawn(fork_turns="all"))),
        ("a legacy turn count is a full-history fork",
         event(**spawn(fork_turns="12"))),
        ("blank fork_turns reads as the default",
         event(**spawn(fork_turns="  "))),
        ("v1 fork_context=true is a full-history fork",
         event(**spawn(fork_context=True, fork_turns=None))),
        ("explicit model wins", event(**spawn(model="gpt-6-astra"))),
        ("no message, nothing to score", event(**spawn(message=""))),
        ("missing tool_input", {k: v for k, v in event(**spawn()).items()
                                if k != "tool_input"}),
        ("empty object", {}),
    ]:
        code, out = run(ev)
        check(name, code == 0 and out == "", f"exit={code} out={out[:60]!r}")

    code, out = run("this is not json")
    check("malformed stdin", code == 0 and out == "", f"exit={code} out={out[:60]!r}")

    # Both spellings reach the policy: the namespaced one every V2 dispatch
    # actually arrives as, and the bare one V1 and the default namespace keep.
    # A renamed namespace is the same shape as the first.
    for name, tool in [("namespaced", "collaborationspawn_agent"),
                       ("bare", "spawn_agent"),
                       ("renamed namespace", "my_teamspawn_agent")]:
        code, out = run({**event(**spawn(message=DEEP)), "tool_name": tool,
                         "model": "gpt-6-luna"},
                        plan={**PLAN, "model": "astra", "effort": "high"})
        check(f"a {name} spawn_agent is routed",
              code == 0 and (updated(out) or {}).get("model") == "gpt-6-astra",
              f"tool_name={tool!r} exit={code} out={out[:60]!r}")

    # Off-ladder parent model: decline rather than guess. The Claude hook
    # declines on fable for the same reason; here the model arrives as a field
    # instead of being dug out of a transcript.
    code, out = run({**event(**spawn()), "model": "gpt-6.1-terra"})
    check("off-ladder parent declines", code == 0 and out == "",
          f"exit={code} out={out[:60]!r}")

    # v1 spawns are routable when they say so, which is the one case the
    # fork_context branch exists to let through.
    code, out = run({**event(**spawn(fork_context=False, fork_turns=None)),
                     "model": "gpt-6-astra"},
                    plan={**PLAN, "model": "luna", "effort": "low"})
    check("v1 fork_context=false still routes",
          code == 0 and (updated(out) or {}).get("model") == "gpt-6-luna",
          f"got {out[:80]!r}")

    # A genuine upgrade, end to end. astra is the top rung, so the parent here
    # is luna and the plan asks for more than it.
    code, out = run({**event(**spawn(message=DEEP, task_name="epoch_rollover")),
                     "model": "gpt-6-luna"},
                    plan={**PLAN, "model": "astra", "effort": "high"})
    upd = updated(out) or {}
    hso = json.loads(out)["hookSpecificOutput"] if out else {}
    check("an upgrade emits a well-formed updatedInput",
          code == 0
          and hso.get("hookEventName") == "PreToolUse"
          and hso.get("permissionDecision") == "allow"
          and "jev: luna -> astra" in hso.get("permissionDecisionReason", "")
          and upd.get("model") == "gpt-6-astra"
          and upd.get("reasoning_effort") == codex.EFFORT["astra"],
          f"got {out[:160]!r}")

    # SpawnAgentArgs is deny_unknown_fields and updatedInput replaces the whole
    # argument object, so anything we drop or invent fails the spawn outright.
    check("updatedInput preserves the rest of the call verbatim",
          upd.get("message") == DEEP and upd.get("task_name") == "epoch_rollover"
          and upd.get("fork_turns") == "none"
          and set(upd) == {"message", "task_name", "fork_turns", "model",
                           "reasoning_effort"},
          f"keys {sorted(upd)}")

    # A downgrade carries the cheap rung's effort with it, rather than leaving
    # the effort the caller chose for a model that is no longer being used.
    code, out = run({**event(**spawn(reasoning_effort="high")),
                     "model": "gpt-6-astra"},
                    plan={**PLAN, "model": "luna", "effort": "low"})
    upd = updated(out) or {}
    check("a downgrade replaces the inherited effort",
          code == 0 and upd.get("model") == "gpt-6-luna"
          and upd.get("reasoning_effort") == codex.EFFORT["luna"],
          f"got {out[:120]!r}")

    # An effort of None means "let Codex pick this model's default", which is
    # the absence of the key. A literal null is not that: it deserializes as a
    # ReasoningEffort and is validated against the new model. No rung is None
    # today, so the branch is reached by pinning one.
    code, out = run({**event(**spawn(reasoning_effort="high")),
                     "model": "gpt-6-astra"},
                    plan={**PLAN, "model": "luna", "effort": None},
                    effort={"luna": None})
    upd = updated(out) or {}
    check("no effort means the key is gone, not null",
          code == 0 and upd.get("model") == "gpt-6-luna"
          and "reasoning_effort" not in upd,
          f"got {out[:120]!r}")

    # A decision that matches the tier already in use changes nothing, and is
    # exactly when nothing but the log can tell you the router ran at all.
    decisions.LOG_DIR = pathlib.Path(LOGDIR)
    code, out = run({**event(**spawn(message=DEEP, task_name="epoch_rollover")),
                     "session_id": "sess-hold"},
                    plan={**PLAN, "model": "sol"})
    logged = decisions.read("sess-hold")
    check("a decision that changed nothing is still logged",
          code == 0 and out == "" and len(logged) == 1
          and logged[0]["model"] == "sol" and logged[0]["default"] == "sol"
          and logged[0]["label"] == "epoch_rollover",
          f"exit={code} out={out[:40]!r} logged {logged!r:.80}")

    # What the hook scored, not just what it decided. The state is the Codex
    # analogue of the Claude hook's three-line block, and a front end that
    # scores the wrong fields is wrong in a way no model id reveals.
    state = logged[0]["task"] if logged else ""
    check("the state carries agent type, task name and message",
          state.startswith("Agent type: default\nDescription: epoch_rollover\nTask:\n")
          and DEEP in state,
          f"scored {state[:90]!r}")

    code, out = run({**event(**spawn(agent_type="reviewer")),
                     "session_id": "sess-role"},
                    plan={**PLAN, "model": "sol"})
    logged = decisions.read("sess-role")
    check("agent_type reaches the state and the label",
          len(logged) == 1 and logged[0]["task"].startswith("Agent type: reviewer")
          and logged[0]["label"] == "gofmt_release_load",
          f"logged {logged!r:.80}")

    # A spawn the hook declines must not leave a decision behind either: there
    # was no decision to make.
    run({**event(**spawn(fork_turns="all")), "session_id": "sess-fork"})
    check("a declined spawn logs nothing", decisions.read("sess-fork") == [],
          f"logged {decisions.read('sess-fork')!r:.60}")

    width = max(len(n) for n, _, _ in cases)
    failed = 0
    for name, ok, detail in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
