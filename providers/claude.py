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
    below want a fixed couple of hundred lines off the end. Measured against the
    169 MB transcript on this machine, `read_text().splitlines()` cost 1,151 MB
    of peak RSS and 0.3s to look at 200 lines -- per dispatch, in a separate
    process per concurrent dispatch. Seeking costs 1 ms at n=200 and 4 ms at the
    n=400 `last_user_prompt` wants. Peak RSS lands at 18 MB, but nearly all of
    that is bare interpreter: the read itself is about 2 MB.
    """
    if n <= 0:
        return []  # `lines[-0:]` is the whole file, which is not "the last none"
    try:
        with open(path, "rb") as f:
            pos = f.seek(0, io.SEEK_END)
            # Collected and joined once at the end. Prepending each block to an
            # accumulator instead -- `buf = f.read(step) + buf` -- recopies
            # everything read so far on every pass, which is quadratic in the
            # bytes read and so only cheap while the tail is short. The loop
            # stops on a newline count, not a byte count, so a transcript of few
            # but very large records reads the whole file that way: over 32 MB
            # that cost 3.7s and 2.0 GB, against 27 ms for the whole-file read
            # this replaced. Joining once makes the same case 22 ms and 87 MB.
            chunks, seen = [], 0
            while pos > 0 and seen <= n:
                step = min(65536, pos)
                pos -= step
                f.seek(pos)
                block = f.read(step)
                seen += block.count(b"\n")
                chunks.append(block)
    except OSError:
        return None

    buf = b"".join(reversed(chunks))
    if not buf:
        return []  # before the strip below: b"\n" is one empty line, b"" is none
    # splitlines() drops a trailing newline; split() would leave an empty
    # final element, which would eat one slot of the lookback for nothing.
    if buf.endswith(b"\n"):
        buf = buf[:-1]
    lines = buf.split(b"\n")
    # Unless the whole file was read, the first element may be the tail of a
    # line whose start is in a block that was never read -- and may begin mid
    # codepoint, which is the other reason it goes rather than being decoded.
    # Dropped unconditionally rather than tested, because a block boundary that
    # happens to land just after a newline makes it a whole line: the loop exits
    # only once it holds n+1 newlines, so `lines[-n:]` cannot reach the first
    # element either way, and discarding a whole one costs nothing.
    if pos > 0:
        lines = lines[1:]
    # Splits on `\n` alone, stripping one trailing `\r`. str.splitlines() also
    # breaks on \v, \f, \x1c, \x1d, \x1e, a lone \r, U+0085, U+2028 and U+2029 --
    # and the old reader opened in text mode, so universal newlines collapsed a
    # lone \r before splitlines() ever saw it. None of the nine appears in any
    # of the 514 transcripts on this machine. RFC 8259 forbids a raw C0 control
    # inside a string and Node's JSON.stringify escapes them, so a record
    # carrying one is unparseable whether it is split or not; of the three that
    # could legally appear -- U+0085, U+2028, U+2029 -- not splitting is the
    # direction that keeps the record whole.
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
