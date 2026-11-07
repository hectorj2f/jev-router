"""Claude Code: three tier aliases, and a JSONL transcript to read them back.

The host resolves the alias itself, so the router's wire value is the tier name
rather than a model id -- which is also why the staleness check in
policy.available_tiers has anything to check.
"""

from __future__ import annotations

import io
import json

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


def _tail(path: str, n: int) -> list[str] | None:
    """The last `n` lines of a file, oldest first. None if it cannot be read.

    Seeking backward rather than reading the whole thing, because a transcript
    is append-only and grows for the life of a session while the two readers
    below want a fixed couple of hundred lines off the end. Measured against a
    169 MB transcript on this machine, `read_text().splitlines()` cost 1,151 MB
    of peak RSS and 0.3s -- per dispatch, in a separate process per concurrent
    dispatch, to look at 200 lines. Reading 64 KB at a time off the end instead
    returns the same answer in 18 MB and 1 ms.
    """
    try:
        with open(path, "rb") as f:
            pos = f.seek(0, io.SEEK_END)
            buf = b""
            while pos > 0 and buf.count(b"\n") <= n:
                step = min(65536, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
    except OSError:
        return None

    if not buf:
        return []  # before the strip below: b"\n" is one empty line, b"" is none
    # splitlines() drops a trailing newline; split() would leave an empty
    # final element, which would eat one slot of the lookback for nothing.
    if buf.endswith(b"\n"):
        buf = buf[:-1]
    lines = buf.split(b"\n")
    # Unless the whole file was read, the first element is the tail of a line
    # whose start is in a block that was never read -- and may begin mid
    # codepoint, which is the other reason it goes rather than being decoded.
    if pos > 0:
        lines = lines[1:]
    # Only `\n` and `\r\n`, where str.splitlines() also breaks on \v, \f and
    # U+2028. That difference is deliberate and in the safe direction: those
    # are legal inside a JSON string, so splitting on one tore a record in half
    # and lost it, where keeping it whole parses.
    return [line.decode(errors="replace").removesuffix("\r") for line in lines[-n:]]


def session_tier(transcript_path: str, lookback: int = 200) -> str | None:
    """The tier the session is actually running on, from its last assistant turn.

    This is what `default` has to be for a subagent: a dispatch inherits the
    session model, so routing against a hardcoded "sonnet" in an opus session
    makes every sonnet-grade task look like "no change" and skip the downgrade
    gate entirely. Returns None for a model outside the ladder (fable) so the
    caller can decline to route rather than guess.
    """
    lines = _tail(transcript_path, lookback)
    if lines is None:
        return None

    for line in reversed(lines):
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
    lines = _tail(transcript_path, lookback)
    if lines is None:
        return None

    for line in reversed(lines):
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
