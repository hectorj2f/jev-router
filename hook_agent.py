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

    state = (f"Agent type: {ti.get('subagent_type') or 'general-purpose'}\n"
             f"Description: {ti.get('description') or ''}\n"
             f"Task:\n{ti.get('prompt') or ''}")
    plan = policy.route(state[:STATE_LIMIT], default=default)
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
    except Exception:
        sys.exit(0)
