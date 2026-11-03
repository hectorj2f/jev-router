"""Claude Code: three tier aliases, and a JSONL transcript to read them back.

The host resolves the alias itself, so the router's wire value is the tier name
rather than a model id -- which is also why the staleness check in
policy.available_tiers has anything to check.
"""

from __future__ import annotations

import json
import pathlib

NAME = "claude"

# Fable is deliberately NOT in the ladder. The /model picker flags it as
# "Requires usage credits" on some plans, and in -p mode Claude Code bills it
# without asking -- not something a router should reach for on its own.
TIERS = ["haiku", "sonnet", "opus"]

# Haiku 4.5 is absent from Claude Code's effort-level table, so passing
# --effort alongside it is an error rather than a no-op.
EFFORT = {"haiku": None, "sonnet": "medium", "opus": "high"}


def model_id(tier: str) -> str:
    """Identity: Claude Code takes the alias and resolves it.

    Emitting a concrete id instead would defeat ANTHROPIC_DEFAULT_<TIER>_MODEL,
    which is how a Vertex or Bedrock deployment points a tier at the one id it
    was actually granted.
    """
    return tier


def tier_of(model: str) -> str | None:
    """A resolved model id back to its tier; None for anything off the ladder.

    A substring test rather than a table, because the tier name is a component
    of every Claude id -- claude-sonnet-4-5, claude-sonnet-5, and the Vertex and
    Bedrock spellings of both -- so this survives a version bump that a table
    would have to be edited for.
    """
    for tier in TIERS:
        if tier in model:
            return tier
    return None


def alias_env(tier: str) -> str | None:
    return f"ANTHROPIC_DEFAULT_{tier.upper()}_MODEL"


def session_tier(transcript_path: str, lookback: int = 200) -> str | None:
    """The tier the session is actually running on, from its last assistant turn.

    This is what `default` has to be for a subagent: a dispatch inherits the
    session model, so routing against a hardcoded "sonnet" in an opus session
    makes every sonnet-grade task look like "no change" and skip the downgrade
    gate entirely. Returns None for a model outside the ladder (fable) so the
    caller can decline to route rather than guess.
    """
    try:
        lines = pathlib.Path(transcript_path).read_text(errors="replace").splitlines()
    except OSError:
        return None

    for line in reversed(lines[-lookback:]):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        model = entry.get("message", {}).get("model") or ""
        return tier_of(model)  # None: a known assistant turn on something off-ladder
    return None


def last_user_prompt(transcript_path: str, lookback: int = 400) -> str | None:
    """The most recent real user turn in a transcript.

    Tool results, meta turns, sidechains, and the wrappers Claude Code writes
    around slash commands all arrive as `type: "user"` too, so they have to go.
    """
    try:
        lines = pathlib.Path(transcript_path).read_text(errors="replace").splitlines()
    except OSError:
        return None

    for line in reversed(lines[-lookback:]):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isSidechain"):
            continue
        content = entry.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.lstrip().startswith(("<command-name>", "<local-command-",
                                        "<command-message>", "<system-reminder>")):
            continue
        return content
    return None
