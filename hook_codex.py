#!/usr/bin/env python3
"""Codex PreToolUse hook on spawn_agent: route each subagent dispatch to a model.

The sibling of hook_agent.py, against Codex's hooks engine, and the only seam
there that can change a call: PreModelRequest and PostModelRequest are logged
and never read back as control flow, and PermissionRequest carries no
`updatedInput`.

Codex hands the hook the parent's own resolved model, so the transcript scrape
policy.session_tier does for Claude Code has no counterpart here.

Fails open everywhere, for the reason hook_agent.py does. Codex sharpens it:
`updatedInput` replaces the whole argument object and SpawnAgentArgs is
`deny_unknown_fields`, so an argument object we got wrong kills the spawn
rather than being ignored. Every error path exits 0 with no output, which
leaves the arguments the model wrote exactly as it wrote them.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Set, not defaulted, and before policy is imported, which is where it is read.
# A JEV_PROVIDER left over from a Claude session would put an anthropic tier
# slug in `model`, and Codex answers an unknown slug with "Unknown model ... for
# spawn_agent" -- a dead dispatch rather than an unrouted one.
os.environ["JEV_PROVIDER"] = "codex"

import decisions  # noqa: E402
import policy  # noqa: E402
from providers import codex  # noqa: E402

STATE_LIMIT = 60_000


def full_history_fork(ti: dict) -> bool:
    """Whether the child inherits the parent's whole conversation.

    Such a child is not the self-contained dispatch this policy was measured
    on -- it continues a conversation it can already read -- and Codex's own
    spawn instructions tell the model it may not override the model on one, so
    rewriting it behind the model's back overturns a decision the model made
    on those terms. The engine does not in fact refuse the override; this
    declines because the routing would be wrong, not because it would fail.

    Two spellings, because the two multi-agent backends disagree. V2 reads
    `fork_turns`, treating missing, blank, "all" and a legacy turn count alike
    as a full fork. V1 has no `fork_turns` at all and defaults `fork_context`
    to false; V2 rejects `fork_context` outright, so the key being present is
    what identifies a V1 call.
    """
    if "fork_context" in ti:
        return bool(ti["fork_context"])
    turns = str(ti.get("fork_turns") or "").strip() or "all"
    return turns.lower() != "none"


def main() -> None:
    ev = json.load(sys.stdin)
    # Suffix, not equality. The V2 backend registers the tool inside a
    # namespace -- "collaboration" by default, and user-settable -- and only
    # the default and V1 namespaces get mapped back to the bare name, so what
    # arrives here is the namespace and the name run together. Every one of
    # the 443 real spawns on the machine this was written on came through as
    # `collaboration` + `spawn_agent`, which the old equality check dropped:
    # the hook was never reached for any dispatch it exists to price. Matching
    # on the tail keeps the bare V1 spelling working and survives a renamed
    # namespace without hardcoding one.
    if not str(ev.get("tool_name") or "").endswith("spawn_agent"):
        return

    ti = dict(ev.get("tool_input") or {})
    if full_history_fork(ti):
        return
    if ti.get("model"):
        return  # an explicit choice, by the user or by me, wins

    task = ti.get("message")
    if not isinstance(task, str) or not task.strip():
        return  # V1 can carry the dispatch in `items` instead; nothing to score

    # The one place Codex is better placed than Claude Code: PreToolUse carries
    # the parent's resolved model, so the default is a field rather than a
    # transcript to reconstruct it from. Off the ladder (gpt-6.1-terra),
    # decline rather than guess.
    default = codex.tier_of(ev.get("model") or "")
    if default is None:
        return

    agent = ti.get("agent_type") or "default"
    state = (f"Agent type: {agent}\n"
             f"Description: {ti.get('task_name') or ''}\n"
             f"Task:\n{task}")
    session = ev.get("session_id") or pathlib.Path(
        ev.get("transcript_path") or "").stem
    label = ti.get("task_name") or agent
    try:
        plan = policy.route(state[:STATE_LIMIT], default=default)
    except (Exception, SystemExit) as e:
        # Both, and they fall through to the one record() below, for the
        # reasons hook_agent.py gives: "could not reach Jev" and "held" both
        # leave the spawn alone, and only the log tells them apart.
        plan = {"model": default, "effort": policy.EFFORT[default],
                "why": f"jev error: {decisions.why_of(e)}"}

    # Before the early return, not after: a decision that changed nothing is
    # still a decision, and it is the one /jev-explain exists to show.
    decisions.record(session, label, default, plan,
                     state, dispatch=ev.get("tool_use_id"))

    if plan["model"] == default:
        return

    ti["model"] = codex.model_id(plan["model"])
    effort = codex.EFFORT[plan["model"]]
    if effort is None:
        # Dropped rather than sent as null, and not left at whatever the caller
        # asked for either: Codex validates the effort against the new model
        # and fails the spawn outright when it is not in that model's set, and
        # an effort chosen for the old model is evidence about the old model.
        # With the key absent Codex fills in the new model's own default, which
        # is by construction one it accepts.
        ti.pop("reasoning_effort", None)
    else:
        ti["reasoning_effort"] = effort

    reason = (f"jev: {default} -> {plan['model']} "
              f"(need {plan['need']}, conf {plan['conf']})")
    # updatedInput REPLACES the arguments rather than merging, so the whole
    # tool_input has to go back, not just the keys being changed. The
    # permissionDecision is what carries it: without `allow` Codex discards the
    # entire output as "updatedInput without permissionDecision:allow".
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": reason,
        "updatedInput": ti,
    }}, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit):
        # SystemExit too: it is how jev.py reports a missing key, an HTTP error
        # or a network failure. Uncaught, it exits 1 with a message on stderr,
        # and what Codex does with a failed PreToolUse hook is not something to
        # find out on a live spawn.
        sys.exit(0)
