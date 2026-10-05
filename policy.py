#!/usr/bin/env python3
"""The routing policy, in one place.

Every surface that routes -- the `cj` launcher, the PreToolUse hook on the Agent
tool, the PreModelSwitch guard, the status line advisor -- imports from here, so
tuning happens once rather than in four drifting copies.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jev import system_one  # noqa: E402

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_TTL = 7 * 24 * 3600

# Atomic questions rather than one "which model" choice. Which tier a task
# deserves depends on our cost tolerance, not on the task, so that half belongs
# in code where it is a coefficient instead of a prompt.
#
# `irreversible` and `ambiguous` are asked but deliberately do NOT feed the tier
# (see decide): in a session the answer to either is to stop and talk, not to
# buy a bigger model.
QUESTIONS = {
    "depth": {
        "type": "score",
        "instructions": "How much reasoning does carrying out this task take, for an "
                        "engineer who already knows this codebase?",
        "criteria": [
            "None. The answer is stated in the task itself.",
            "Little. Look something up and report it.",
            "Some. Follow an established pattern to a new place.",
            "Substantial. Trace behaviour across components and decide what is correct.",
            "Heavy. Novel design, or weighing trade-offs with no established answer.",
        ],
    },
    "breadth": {
        "type": "score",
        "instructions": "How much of the codebase does this task touch?",
        "criteria": [
            "One known location, named in the task.",
            "One file or one small package.",
            "Several files inside one module.",
            "Several modules, or a search with no known starting point.",
            "Repository-wide, or spanning build, infrastructure, and code together.",
        ],
    },
    "mechanical": {
        "type": "noul",
        "instructions": "The task is mechanical: rename, reformat, regenerate, or apply "
                        "an edit the task already specifies. There is no judgement about "
                        "what the right answer is, only about carrying it out.",
    },
    # There was a "trivial" noul here -- "a small fast model would get this right
    # first time" -- as a direct haiku gate. Removed after measuring: across 36
    # calls it never once exceeded 0.70, and on the ten-task fan-out where four
    # shards did route to haiku, all four cleared on `mechanical` or on score
    # confidence instead. Jev answers it conservatively enough that at any
    # threshold loose enough to fire, it stops discriminating. `mechanical` asks
    # a narrower question and gets a sharper answer.
    "irreversible": {
        "type": "noul",
        "instructions": "Carrying this out writes to something outside the working tree "
                        "that is hard to undo: pushing, force-pushing, deploying, applying "
                        "terraform, mutating a database, posting to an external service, "
                        "or sending a message to another person.",
    },
    "ambiguous": {
        "type": "noul",
        "instructions": "The task is under-specified. A careful engineer would have to "
                        "choose between materially different readings of it.",
    },
}

# Fable is deliberately NOT in the ladder. The /model picker flags it as
# "Requires usage credits" on some plans, and in -p mode Claude Code bills it
# without asking -- not something a router should reach for on its own.
TIERS = ["haiku", "sonnet", "opus"]

# Haiku 4.5 is absent from Claude Code's effort-level table, so passing
# --effort alongside it is an error rather than a no-op.
EFFORT = {"haiku": None, "sonnet": "medium", "opus": "high"}

OPUS_LINE = 0.60
SONNET_LINE = 0.25


def available_tiers() -> list[str]:
    """The tiers this machine can actually serve, cheapest first.

    Routing to a model the endpoint does not have is worse than not routing:
    the dispatch dies on a 404 instead of running on the model it would have
    used. Measured the hard way on a Vertex project with no haiku enabled --
    the hook correctly picked haiku and the agent came back
    `model_not_found: claude-haiku-4-5@20251001`.

    Claude Code resolves each tier alias through ANTHROPIC_DEFAULT_<TIER>_MODEL
    and falls back to a built-in ID when one is unset. On the first-party API
    that fallback is always valid; on Vertex and Bedrock it names a model the
    project may never have been granted, so an unset tier there is a tier we
    cannot assume. Override with JEV_TIERS=sonnet,opus.
    """
    explicit = os.environ.get("JEV_TIERS", "").strip()
    if explicit:
        want = {t.strip() for t in explicit.split(",")}
        return [t for t in TIERS if t in want] or list(TIERS)

    if not (os.environ.get("CLAUDE_CODE_USE_VERTEX")
            or os.environ.get("CLAUDE_CODE_USE_BEDROCK")):
        return list(TIERS)

    pinned = [t for t in TIERS if os.environ.get(f"ANTHROPIC_DEFAULT_{t.upper()}_MODEL")]
    # Nothing pinned means the deployment leaves every alias to the fallback,
    # which tells us nothing either way -- guessing a narrower ladder from that
    # would disable routing wholesale on a perfectly healthy setup.
    return pinned or list(TIERS)


def nearest_tier(tier: str, avail: list[str]) -> str:
    """`tier` if it is served, else the next one up. Never silently cheaper."""
    for t in TIERS[TIERS.index(tier):]:
        if t in avail:
            return t
    return avail[-1] if avail else tier


def ask(state: str, cache: bool = True) -> dict:
    """Jev's raw answers for one task, memoized on disk by the task text.

    The status line re-runs on every assistant message, and the hooks can see
    the same task more than once, so an uncached call here would mean a request
    per redraw.
    """
    if not cache:
        return system_one(state, QUESTIONS)["answers"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The questions are part of the key, not just the state. Keying on the state
    # alone means editing QUESTIONS silently serves answers to the old ones.
    key = json.dumps([state, QUESTIONS], sort_keys=True).encode()
    path = CACHE_DIR / (hashlib.sha256(key).hexdigest()[:32] + ".json")
    if path.is_file() and time.time() - path.stat().st_mtime < CACHE_TTL:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass  # fall through and refetch

    answers = system_one(state, QUESTIONS)["answers"]
    path.write_text(json.dumps(answers))
    return answers


def decide(ans: dict, default: str = "sonnet", tiers: list[str] | None = None) -> dict:
    """Turn Jev's answers into a tier, applying the asymmetric gates."""
    avail = tiers if tiers is not None else available_tiers()
    depth = ans["depth"]["score"] / (len(QUESTIONS["depth"]["criteria"]) - 1)
    breadth = ans["breadth"]["score"] / (len(QUESTIONS["breadth"]["criteria"]) - 1)
    conf = min(ans["depth"]["confidence"], ans["breadth"]["confidence"])
    mechanical = ans["mechanical"]["noul"]

    # Only how hard the work is. `ambiguous` used to carry 0.25 of this and was
    # the reason nothing ever reached haiku: measured over eight real prompts it
    # never read below 0.62 and sat at a median of 0.80, so it contributed a
    # near-constant +0.20 against a 0.25 haiku line. That reading is not wrong --
    # a conversational prompt genuinely is under-specified -- it is just the
    # wrong response to it. See `advice` below.
    need = 0.60 * depth + 0.40 * breadth - 0.25 * mechanical

    wanted = "opus" if need >= OPUS_LINE else "sonnet" if need >= SONNET_LINE else "haiku"
    suggested = nearest_tier(wanted, avail)

    # An upgrade is ungated: if `need` clears the bar, take it, even when Jev is
    # unsure. Measured over ten real tasks, a confidence floor on the upgrade did
    # the opposite of what it was for -- the two heaviest scored need 0.60 and
    # 0.67 at score-confidence 0.00 and 0.25, and the floor pinned exactly those
    # to the cheaper tier. That is inherent to a 5-level score: probability split
    # across adjacent levels reads as low confidence while the weighted value
    # stays sound.
    #
    # A downgrade still has to be earned. `mechanical` is the sharper of the two
    # signals that can earn it -- a two-outcome question, so its distribution is
    # not diffused across levels the way a 5-level score's is.
    if TIERS.index(suggested) > TIERS.index(default):
        model = suggested
    elif TIERS.index(suggested) < TIERS.index(default):
        model = suggested if (mechanical > 0.75 or conf >= 0.70) else default
    else:
        model = default

    return {
        "model": model,
        "effort": EFFORT[model],
        "need": round(need, 3),
        "conf": round(conf, 3),
        "suggested": suggested,
        "held": model != suggested,
        # What the score alone asked for, before the ladder was narrowed to what
        # this endpoint serves. Differs from `suggested` only when a tier is
        # missing, and that is worth seeing rather than silently absorbing.
        "wanted": wanted,
        "tiers": avail,
        # Irreversibility and ambiguity do not belong in the tier. A cheap model
        # posting the right comment is fine; what the irreversibility is asking
        # for is a confirmation step, and what the ambiguity is asking for is a
        # question back. Buying a bigger model answers neither, and on a measured
        # run the old irreversible floor sent a depth-0.4 task ("reformat this
        # comment I already wrote") straight to opus on that basis alone.
        "advice": {
            "confirm": ans["irreversible"]["noul"] > 0.60,
            "clarify": ans["ambiguous"]["noul"] > 0.70,
        },
        "signals": {
            "depth": ans["depth"]["score"],
            "breadth": ans["breadth"]["score"],
            "mechanical": round(mechanical, 2),
            "irreversible": round(ans["irreversible"]["noul"], 2),
            "ambiguous": round(ans["ambiguous"]["noul"], 2),
        },
    }


def route(state: str, default: str = "sonnet", cache: bool = True) -> dict:
    return decide(ask(state, cache=cache), default=default)


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
        for tier in TIERS:
            if tier in model:
                return tier
        return None  # a known assistant turn on something off-ladder
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
