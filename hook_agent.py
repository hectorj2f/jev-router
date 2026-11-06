#!/usr/bin/env python3
"""PreToolUse hook on ^Agent$: route each subagent dispatch to a model.

The single place in Claude Code where a hook can actually set a model --
`PreModelSwitch` only allows/denies, and the `model` settings key is read once
at session start. Dispatch prompts describe the work rather than pointing at
it, which is the regime where the routing measures well; the same policy over
conversational prompts does not earn its keep.

Fails open everywhere. A router that can block a dispatch is worse than no
router, so every error path exits 0 with no output, leaving the dispatch alone.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decisions  # noqa: E402
import policy  # noqa: E402

STATE_LIMIT = 60_000


def main() -> None:
    ev = json.load(sys.stdin)
    ti = dict(ev.get("tool_input") or {})

    if ti.get("subagent_type") == "fork":
        return  # a fork always inherits the parent model; `model` is ignored
    if ti.get("model"):
        return  # an explicit choice, by the user or by me, wins

    # The default MUST be the session's own tier, not a constant. A dispatch
    # inherits the session model, so hardcoding "sonnet" in an opus session
    # makes every sonnet-grade task read as "no change" and skip the downgrade
    # gate entirely -- measured, that silently dropped 5 of 10 shards from opus
    # to sonnet, one of them at confidence 0.00.
    default = policy.session_tier(ev.get("transcript_path") or "")
    if default is None:
        return  # unreadable, or off-ladder (fable): decline rather than guess

    agent = ti.get("subagent_type") or "general-purpose"
    state = (f"Agent type: {agent}\n"
             f"Description: {ti.get('description') or ''}\n"
             f"Task:\n{ti.get('prompt') or ''}")
    session = ev.get("session_id") or pathlib.Path(
        ev.get("transcript_path") or "").stem
    label = ti.get("description") or agent
    try:
        plan = policy.route(state[:STATE_LIMIT], default=default)
    except (Exception, SystemExit) as e:
        # Both, and they arrive for different reasons: jev.py reports a missing
        # key or a dead API as SystemExit, which is not an Exception, while a
        # malformed answer raises KeyError inside decide(). Either way the
        # dispatch is left alone, which is indistinguishable from a hold until
        # the log says otherwise -- so fall through to the one record() below
        # rather than logging here, and keep the failure on the same path as
        # the decision. route.py:route_one does the same.
        plan = {"model": default, "effort": policy.EFFORT[default],
                "why": f"jev error: {decisions.why_of(e)}"}

    # Before the early return, not after: a decision that changed nothing is
    # still a decision, and it is the one /jev-explain exists to show.
    decisions.record(session, label, default, plan,
                     state, dispatch=ev.get("tool_use_id"))

    if plan["model"] == default:
        return

    ti["model"] = plan["model"]
    reason = (f"jev: {default} -> {plan['model']} "
              f"(need {plan['need']}, conf {plan['conf']})")
    # updatedInput REPLACES the arguments rather than merging, so the whole
    # tool_input has to go back, not just the key being changed.
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "updatedInput": ti,
        "permissionDecisionReason": reason,
    }}, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit):
        # SystemExit too: it is how jev.py reports a missing key, an HTTP error
        # or a network failure, and an uncaught one exits 1 with a message on
        # stderr rather than the silent 0 this hook promises.
        sys.exit(0)
