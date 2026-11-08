#!/usr/bin/env python3
"""The routing policy, in one place.

Every surface that routes comes through here -- the Claude Code mod on
`agent.spawn`, the PreToolUse hooks on Claude's `Agent` tool and on Codex's
`spawn_agent`, and route.py's fan-out, which is also what the mod shells out
to. So tuning happens once rather than in four drifting copies.

That is a claim this file has to keep earning. It once named four surfaces of
which three did not exist, which is the same drift one level up: a reader
checking whether a number is spelled anywhere else was sent to look in the
wrong places.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import providers  # noqa: E402
from jev import model_name, system_one  # noqa: E402

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_TTL = 7 * 24 * 3600

# How much of a task is worth scoring. Jev charges for input, and a dispatch
# prompt past this is a pathological case rather than a task description.
#
# Enforced in ask_meta, which every routing path reaches, because it was
# previously enforced by each caller instead: hook_agent.py, hook_codex.py,
# hooks/register.js and an assertion in tests/register.test.ts each spelled
# this number, while route.py -- the fan-out, and the one the mod shells out
# to -- applied no cap at all and sent whatever it was handed. jev.py's
# STATE_CHAR_LIMIT is a different number answering a different question; see
# the comment there.
STATE_LIMIT = 60_000

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

# Which rungs exist, what they are called on the wire, and what effort each
# takes are the host's business, not the policy's (see providers/). Bound at
# module scope so that `policy.TIERS` stays the name every front end reads.
PROVIDER = providers.current()
TIERS = PROVIDER.TIERS
EFFORT = PROVIDER.EFFORT

# Measured against Claude's ladder, and named for the rungs they were measured
# on, but what they mark is a position on a three-rung ladder rather than a
# model: route.py plots them by these names, so they keep them.
OPUS_LINE = 0.60
SONNET_LINE = 0.25


def _tiers_file(prov: types.ModuleType) -> pathlib.Path:
    """Where `prov`'s probe record lives.

    Per provider, because a ladder measured against one host says nothing
    about the other. The unsuffixed name stays readable for claude: probe.py
    writes cache/tiers.json, and the measurement already on disk is in it.
    """
    per_provider = CACHE_DIR / f"tiers-{prov.NAME}.json"
    if per_provider.is_file() or prov.NAME != "claude":
        return per_provider
    return CACHE_DIR / "tiers.json"


def available_tiers(provider: types.ModuleType | None = None) -> list[str]:
    """The tiers this machine can actually serve, cheapest first.

    Routing to a model the endpoint does not have is worse than not routing:
    the dispatch dies on a 404 instead of running on the model it would have
    used, and the hook cannot catch that -- it happens after the hook returns.

    JEV_TIERS wins, then a provider that can answer for itself (codex), then
    what `./probe.py` measured. There was
    an inference here once -- "pinned in the environment, therefore served" --
    and it was wrong in the expensive direction: a Vertex project pinned
    ANTHROPIC_DEFAULT_SONNET_MODEL to a model it had never been granted, so the
    tier the inference was surest of was the one that 404'd. With no probe on
    file, assume the full ladder rather than guess a narrower one, and let the
    first dead dispatch send you to probe.py.
    """
    prov = provider if provider is not None else PROVIDER
    ladder = prov.TIERS

    rejected = False
    explicit = os.environ.get("JEV_TIERS", "").strip()
    if explicit:
        want = [t.strip() for t in explicit.split(",") if t.strip()]
        unknown = [t for t in want if t not in ladder]
        if want and not unknown:
            return [t for t in ladder if t in want]
        # Honouring the half that parsed is how `JEV_TIERS=haiku,sonet` quietly
        # became "haiku only" -- every dispatch to the cheapest rung, off one
        # transposed letter, with nothing said.
        #
        # A list that does not describe this ladder is not about this ladder.
        # It is a typo, or it is another provider's rungs still exported in
        # the shell -- hook_codex.py sets JEV_PROVIDER itself, so a JEV_TIERS
        # written for Claude reaches the codex ladder intact. Say so, and fall
        # through to something measured rather than guess which half was meant.
        rejected = True
        print(f"jev: ignoring JEV_TIERS={explicit!r}: "
              f"{', '.join(unknown) or 'no tier names in it'} -- the "
              f"{prov.NAME} ladder is {', '.join(ladder)}", file=sys.stderr)

    # A provider that can answer this itself beats a probe record: codex-cli
    # caches the catalogue its own account was granted, so the measurement is
    # already on disk and as current as the last `codex` run, where probe.py's
    # is as current as whenever someone last remembered to run it. None means
    # no catalogue could be read, which falls through to the same places an
    # unprobed Claude does; an empty list is a real measurement and is
    # returned as one.
    own = getattr(prov, "served", None)
    if own is not None:
        got = own()
        if got is not None:
            return got

    try:
        rec = json.loads(_tiers_file(prov).read_text())
        recorded, resolved = rec["tiers"], rec.get("resolved", {})
    except (OSError, json.JSONDecodeError, KeyError):
        # The un-probed default is the full ladder -- but only for someone who
        # never claimed to know better. After a rejected JEV_TIERS it would be
        # the widest possible answer to the one input that said "these are the
        # tiers I have", and the docstring above calls emitting an unserved
        # tier the worst outcome there is. Empty disables routing in decide(),
        # which costs the saving and nothing else.
        return [] if rejected else list(ladder)

    out = []
    for t in ladder:
        if t not in recorded:
            continue
        # The probe measured one specific model ID. If the environment now
        # points this alias somewhere else, that measurement is about a
        # different model and says nothing about the one that would be used.
        # A provider with no alias to repoint (codex) has nothing that can go
        # stale this way: it emits the literal slug the probe measured.
        alias = prov.alias_env(t)
        was = resolved.get(t)
        if alias and was and was != "(built-in default)" and \
                os.environ.get(alias) != was:
            continue
        out.append(t)
    # Deliberately NOT `or list(TIERS)`. A record that yields nothing -- empty,
    # or wholly stale -- is a failed measurement, and falling back to the full
    # ladder turns that into permission to emit a tier nothing has ever
    # confirmed. Returning empty disables routing instead (see decide), which
    # is the direction that costs a saving rather than an agent. No record at
    # all still means the full ladder: that is the un-probed default, not a
    # measurement that came back blank.
    return out


def need_of(depth: float, breadth: float, mechanical: float) -> float:
    """How much model the work wants, from three signals already in [0, 1].

    A function rather than a line inside decide() because replay.py evaluates
    it once per (depth, breadth) level pair to get a distribution over rungs.
    Two copies of these weights would be two policies the moment either moved.
    """
    return 0.60 * depth + 0.40 * breadth - 0.25 * mechanical


def rung_of(need: float, tiers: list[str]) -> str:
    """Which rung a `need` falls in. The two lines, in one place."""
    cheap, mid, dear = tiers
    return dear if need >= OPUS_LINE else mid if need >= SONNET_LINE else cheap


def nearest_tier(tier: str, avail: list[str], ladder: list[str] | None = None) -> str:
    """`tier` if it is served, else the next one up. Never silently cheaper."""
    ladder = ladder if ladder is not None else TIERS
    for t in ladder[ladder.index(tier):]:
        if t in avail:
            return t
    return avail[-1] if avail else tier


# Which question set a logged answer was given to. Short because it is a label
# rather than a security property: two records with the same hash were scored
# by the same questions and can be compared, and two with different ones cannot.
QUESTIONS_HASH = hashlib.sha256(
    json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:12]

# The fields each answer type has to carry for decide() to read it, and the
# range each has to be in. None means "0 to the last index of this question's
# own criteria list", which only a score has.
_NUMERIC = {
    "score": (("score", None), ("confidence", (0.0, 1.0))),
    "noul": (("noul", (0.0, 1.0)),),
    "choice": (("confidence", (0.0, 1.0)),),
}


def check_answers(answers, questions: dict = QUESTIONS) -> None:
    """Raise ValueError unless every question got an answer decide() can read.

    Checked before anything is cached, because the cache keeps what it is given
    for a week: an answer missing a field used to be stored, then raise KeyError
    inside decide() on every dispatch of that task until it expired.

    Range and finiteness are part of "can read", not pedantry. Both lines in
    rung_of are `>=` comparisons, and every comparison against NaN is False, so
    a NaN need falls through to the *cheapest* rung with no error anywhere --
    json.load parses a bare NaN token by default, so it arrives this far. An
    out-of-range noul is as bad in the other direction: mechanical = 7.0 gives
    need = -0.75 and clears the `> 0.75` downgrade gate at the same time.
    """
    if not isinstance(answers, dict):
        raise ValueError(f"jev: answers is {type(answers).__name__}, not an object")
    for name, q in questions.items():
        a = answers.get(name)
        if not isinstance(a, dict) or a.get("type") != q["type"]:
            raise ValueError(f"jev: no {q['type']} answer for {name!r}")
        if q["type"] not in _NUMERIC:
            # Presence-only was the silent default here, in a function whose
            # whole promise is that decide() can read what survives it.
            raise ValueError(f"jev: {name!r} has unchecked type {q['type']!r}; "
                             f"teach check_answers about it before asking it")
        for field, bounds in _NUMERIC[q["type"]]:
            v = a.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"jev: {name}.{field} is {v!r}, not a number")
            if not math.isfinite(v):
                raise ValueError(f"jev: {name}.{field} is {v!r}, not finite")
            lo, hi = bounds or (0, len(q["criteria"]) - 1)
            if not lo <= v <= hi:
                raise ValueError(
                    f"jev: {name}.{field} is {v!r}, outside [{lo:g}, {hi:g}]")


def _cache_path(state: str, requested: str) -> pathlib.Path:
    # The questions are part of the key, not just the state. Keying on the state
    # alone means editing QUESTIONS silently serves answers to the old ones. The
    # requested model is part of it for the same reason: a week-old answer from
    # one version of Jev is not an answer from the version now pinned.
    key = json.dumps([state, QUESTIONS, requested], sort_keys=True).encode()
    return CACHE_DIR / (hashlib.sha256(key).hexdigest()[:32] + ".json")


def ask_meta(state: str, cache: bool = True) -> dict:
    """Jev's raw answers for one task, with which Jev gave them.

    Returns {"answers", "model", "requested", "cached"}. `model` is the version
    the response says answered, which is not necessarily `requested`: an alias
    such as jev-latest resolves server-side, and only the response says to what.

    Memoized on disk by task text, question set and requested model. The hooks
    can see the same task more than once, so an uncached call here would mean a
    request per sighting.

    `state` is cut to STATE_LIMIT here rather than by each caller -- see the
    constant for what that cost.
    """
    requested = model_name()
    # Before the cache key, not after, so one task does not land under two
    # keys depending on which caller trimmed it first.
    state = state[:STATE_LIMIT]
    path = _cache_path(state, requested) if cache else None
    if path is not None:
        # OSError included, and the stat inside the try: one unreadable entry
        # (bad mode, or a sibling pruning it between the stat and the read)
        # otherwise raised on every dispatch of that task for the file's whole
        # seven-day life, never refetching -- the same shape of bug the answer
        # check above was added to fix, one layer out.
        try:
            if path.is_file() and time.time() - path.stat().st_mtime < CACHE_TTL:
                rec = json.loads(path.read_text())
                check_answers(rec["answers"])
                return {"answers": rec["answers"], "model": rec.get("model"),
                        "requested": requested, "cached": True}
        except (ValueError, TypeError, KeyError, OSError):
            pass  # unreadable or incomplete: refetch rather than serve it

    resp = system_one(state, QUESTIONS, model=requested)
    answers = resp.get("answers") if isinstance(resp, dict) else None
    check_answers(answers)
    model = resp.get("model")
    if path is not None:
        # An answer already fetched, validated and paid for is not worth
        # throwing away because the cache dir is unwritable.
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "answers": answers}))
        except OSError:
            pass
    return {"answers": answers, "model": model, "requested": requested,
            "cached": False}


def ask(state: str, cache: bool = True) -> dict:
    """Jev's raw answers for one task. See ask_meta for which Jev gave them."""
    return ask_meta(state, cache=cache)["answers"]


def decide(ans: dict, default: str | None = None, tiers: list[str] | None = None,
           provider: types.ModuleType | None = None) -> dict:
    """Turn Jev's answers into a tier, applying the asymmetric gates.

    `default` is the tier the caller is already on. It defaults to the middle
    rung, which on Claude is the "sonnet" every existing caller that omits it
    meant; naming it by position is what lets the same call work on a ladder
    with no sonnet on it.
    """
    prov = provider if provider is not None else PROVIDER
    # Unpacking rather than indexing: the two lines below cut a ladder into
    # three, so a provider that offered four rungs would be silently routed as
    # if it had three. Fail where the assumption is, not two frames later.
    cheap, mid, dear = prov.TIERS
    default = default if default is not None else mid
    avail = tiers if tiers is not None else available_tiers(prov)
    # Nothing known to be serviceable: the one model we can be sure this
    # endpoint serves is the one the session is already running on. Clamping to
    # it makes every decision a no-op rather than a guess.
    avail = avail or [default]
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
    need = need_of(depth, breadth, mechanical)

    wanted = rung_of(need, prov.TIERS)
    suggested = nearest_tier(wanted, avail, prov.TIERS)

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
    if prov.TIERS.index(suggested) > prov.TIERS.index(default):
        model = suggested
    elif prov.TIERS.index(suggested) < prov.TIERS.index(default):
        model = suggested if (mechanical > 0.75 or conf >= 0.70) else default
    else:
        model = default

    return {
        "model": model,
        "effort": prov.EFFORT[model],
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


def route(state: str, default: str | None = None, cache: bool = True) -> dict:
    """decide() over a fresh or cached answer, carrying the answer with it.

    The raw answer rides along under `jev` so the decision log keeps what was
    measured, not only what was concluded from it. Without the full level
    probabilities a different gate cannot be replayed over past dispatches, and
    every threshold change would have to be judged on new traffic only.
    """
    meta = ask_meta(state, cache=cache)
    plan = decide(meta["answers"], default=default)
    plan["jev"] = {
        "model": meta["model"],
        "requested": meta["requested"],
        "questions": QUESTIONS_HASH,
        "cached": meta["cached"],
        # The legend repeats QUESTIONS' own criteria text on every answer, and
        # the hash above already says which criteria those were.
        "answers": {name: {k: v for k, v in a.items() if k != "legend"}
                    for name, a in meta["answers"].items()},
    }
    return plan


def decision_reason(default: str, plan: dict) -> str:
    """The one line a PreToolUse hook gets to say, on both hosts.

    A hook's only voice is `permissionDecisionReason`, which rides on a verdict
    -- so this is reachable only where the router actually re-tiered something.
    The mod says the same things through `$.ui.log` and is not limited that
    way. Shared rather than written twice because it was written twice, and
    the half that was missing from both was the irreversibility warning the
    advice section of the README spends a page arguing for.
    """
    line = (f"jev: {default} -> {plan['model']} "
            f"(need {plan['need']}, conf {plan['conf']})")
    if (plan.get("advice") or {}).get("confirm"):
        # Appended rather than sent as its own output: a second message would
        # mean speaking about a spawn on a path where the hook would otherwise
        # stay silent, and the only way a PreToolUse hook can speak is by
        # returning a verdict on the tool call. Turning a router into a thing
        # that approves tool calls is the wrong trade for a warning.
        irreversible = (plan.get("signals") or {}).get("irreversible")
        line += ("; writes outside the working tree"
                 + (f" (irreversible {irreversible:.2f})"
                    if isinstance(irreversible, (int, float)) else "")
                 + " -- a warning, not a gate")
    return line


def _reader(name: str):
    """A provider's transcript reader, or a failure that names who lacks one.

    A wrapper rather than `session_tier = PROVIDER.session_tier`, because that
    assignment makes the name disappear from this module under a provider that
    has no transcript to read, and hook_agent.py calls it unconditionally. The
    hook fails open on either error; what differs is whether the traceback
    names the provider that lacks the reader or just this file.
    """
    fn = getattr(PROVIDER, name, None)
    if fn is None:
        raise NotImplementedError(
            f"provider {PROVIDER.NAME!r} has no {name}: it reads a Claude Code "
            f"transcript, and nothing else writes one")
    return fn


def session_tier(transcript_path: str, lookback: int = 200) -> str | None:
    """The tier the session is running on -- see the provider for the why."""
    return _reader("session_tier")(transcript_path, lookback)


def last_user_prompt(transcript_path: str, lookback: int = 400) -> str | None:
    """The most recent real user turn -- see the provider for the why."""
    return _reader("last_user_prompt")(transcript_path, lookback)
