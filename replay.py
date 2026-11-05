#!/usr/bin/env python3
"""Replay a different downgrade gate over decisions already made.

    ./replay.py                      # the most recent session
    ./replay.py --session <id>       # another one
    ./replay.py --all                # every session on disk
    ./replay.py --tau 0.85           # a different downgrade bar
    ./replay.py --tau 'opus>haiku=0.95'   # ... for one transition only
    ./replay.py --sweep              # how the disagreement count moves with tau
    ./replay.py --json

Calls nothing. Every number here comes out of the decision log, which is why
schema 2 keeps Jev's raw answer: without the level probabilities a new gate
could only ever be judged on new traffic, one dispatch at a time.

**This changes no routing.** It is the measuring instrument, not the policy.
It is imported by explain.py, for an extra column, and by selftest_route.py.

## The gate being replayed

The acting policy (a) collapses `depth` and `breadth` to their weighted means,
computes one `need`, and gates a downgrade on Jev's own `confidence` field --
which the JEV-as-a-Judge study pointedly does not use, its formula being
unpublished. There is no probability anywhere for the thing actually being
decided.

(b) keeps the levels apart. Jev returns a probability per level, so the pair
(depth, breadth) is a distribution over 25 outcomes; push each through the same
`need_of` and the same two lines and that becomes a distribution over the three
rungs. Then a downgrade needs P(this rung is enough) >= tau -- a probability of
the decision, which is the quantity the paper's cascade is validated on.

Worked, on the answer in selftest_route.py: depth .04/.13/.68/.15/0, breadth
.03/.01/.02/.79/.15, mechanical .88 gives P(haiku) .182, P(sonnet) .795,
P(opus) .022, so P(sonnet or cheaper) = .977 and the downgrade clears any tau
below that.

## What the defaults are worth

Nothing. TAU and UP_BAR below are placeholders so the tool runs, not findings.
Picking them needs outcome labels, which the log does not have yet -- `outcome`
is still null on every record. An exact one-sided binomial bound wants 149
failure-free downgrades to certify a failure rate under 2%, and the study's own
result is that a threshold does not transfer between workloads. So treat a
number printed here as "what (b) would have done at a tau someone made up",
and read --sweep rather than the default.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import decisions  # noqa: E402
import policy  # noqa: E402
import providers  # noqa: E402

# Provisional. See the docstring: these are not measured, and the point of the
# tool is to find out what they should be.
TAU = 0.70
UP_BAR = 0.50

# How far the level probabilities may miss summing to 1 before the answer is
# refused rather than renormalised. The JEV-as-a-Judge study (arXiv 2609.26550,
# Appendix J) measures a noul's two outcomes missing by as much as 0.19; a
# Score's levels come off one softmax and should be far tighter, so a miss this
# size is a shape problem rather than noise worth rescaling away.
SUM_TOLERANCE = 0.025


def level_probs(answer: object, levels: int) -> list[float] | None:
    """Renormalised level probabilities, or None if they cannot be trusted.

    None rather than a guess: every caller treats it as "hold at the parent
    tier", because a gate fed a distribution it does not understand should
    decline rather than act on the part it does.
    """
    if not isinstance(answer, dict):
        return None
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        return None
    # Exactly the levels this question has, no more: an extra key means the
    # question changed under the log and the levels no longer line up with the
    # criteria `need_of` weights them by.
    if set(raw) != {str(i) for i in range(levels)}:
        return None
    p = [raw[str(i)] for i in range(levels)]
    # Numbers, not things float() would accept. check_answers refuses a number
    # sent as a string on the live path, and two readers of one wire format
    # that disagree about what it is are worse than either rule alone.
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in p):
        return None
    # isfinite first, and in the same test as the sign: every comparison
    # against NaN is False, so a NaN level used to pass the sign check *and*
    # the sum check below, and arrive at rung_of as a need that falls through
    # both lines to the cheapest rung. json.load parses a bare NaN token by
    # default, so one can reach here straight off the wire.
    if not all(math.isfinite(v) and v >= 0 for v in p):
        return None
    total = sum(p)
    if abs(total - 1.0) > SUM_TOLERANCE or total <= 0:
        return None
    return [v / total for v in p]


def rung_probs(answers: dict, tiers: list[str]) -> dict[str, float] | None:
    """P(each rung), from the level probabilities of depth and breadth.

    Assumes the two are independent, which is untested and is the main thing
    wrong with this: a task that is deep is plausibly also broad, and a
    positive correlation would put more mass in the tails than this computes.
    Testing it needs the joint, which one request per question would not give
    either: the study measures joint and standalone Choice answers differing by
    0.011, so the answers are near-independent as *measurements*. Whether the
    quantities behind them are is a different question, and not one the log can
    settle.

    `mechanical` stays a point value. It shifts every cell by the same amount,
    so it moves the lines rather than spreading the mass.
    """
    if not isinstance(answers, dict):
        return None
    d_levels = len(policy.QUESTIONS["depth"]["criteria"])
    b_levels = len(policy.QUESTIONS["breadth"]["criteria"])
    pd = level_probs(answers.get("depth"), d_levels)
    pb = level_probs(answers.get("breadth"), b_levels)
    if pd is None or pb is None:
        return None
    mech_answer = answers.get("mechanical")
    if not isinstance(mech_answer, dict):
        return None
    mech = mech_answer.get("noul")
    if isinstance(mech, bool) or not isinstance(mech, (int, float)) \
            or not math.isfinite(mech):
        # A NaN here is the worst of the malformed shapes: it makes every one
        # of the 25 cells fall through rung_of to the cheapest rung, so the
        # tool reports a confident point mass on haiku with no nan in sight.
        return None

    out = {t: 0.0 for t in tiers}
    for i, pi in enumerate(pd):
        for j, pj in enumerate(pb):
            need = policy.need_of(i / (d_levels - 1), j / (b_levels - 1), mech)
            out[policy.rung_of(need, tiers)] += pi * pj
    return out


def tau_for(frm: str, to: str, taus: dict) -> float:
    """The bar for one transition, falling back to the global one."""
    return taus.get((frm, to), taus["*"])


def summed_gate(probs: dict, default: str, tiers: list[str], avail: list[str],
                taus: dict, up_bar: float) -> tuple[str, str]:
    """(b): the rung this answer earns, and the sentence that says why.

    Upgrades on probability mass rather than the top label -- the argmax of a
    5-level score is noisy, and a rung holding 0.3 of the mass can be the
    argmax while 0.6 sits above it. Downgrades go to the *cheapest* rung whose
    cumulative mass clears tau, so a confident answer saves as much as it has
    earned rather than one rung at a time.

    With tau and up_bar both > 0.5 the two rules cannot both fire: the mass
    they ask for would have to overlap. At exactly 0.5 they can -- a
    distribution split cleanly in half satisfies both -- and the upgrade is
    checked first, so that is what wins. Erring dear is the safe direction.
    """
    here = tiers.index(default)

    # Upgrade: the highest rung r with P(rung >= r) >= up_bar.
    acc = 0.0
    for t in reversed(tiers):
        acc += probs[t]
        if acc >= up_bar:
            if tiers.index(t) > here:
                return (policy.nearest_tier(t, avail, tiers),
                        f"P(>= {t}) {acc:.3f} >= {up_bar:g}")
            break

    # Downgrade: the cheapest rung r below the parent with P(rung <= r) >= tau.
    acc = 0.0
    missed = ""
    gap = float("inf")
    for t in tiers:
        acc += probs[t]
        if tiers.index(t) >= here:
            break
        bar = tau_for(default, t, taus)
        if acc >= bar:
            # Clamped up to what the endpoint served, exactly as (a) does. The
            # rungs themselves were computed on the full ladder, so a tier
            # being unavailable never moves the probabilities.
            return (policy.nearest_tier(t, avail, tiers),
                    f"P(<= {t}) {acc:.3f} >= {bar:g}")
        # The closest the answer came, kept by how far it fell short rather
        # than by which rung came last. Under one global bar those agree, but
        # per transition they do not: a 0.01 miss at opus>haiku and a 0.30
        # miss at opus>sonnet would otherwise report the 0.30. A bare "held"
        # next to a tau sweep says nothing about whether the bar missed by
        # 0.002 or by 0.6.
        if bar - acc < gap:
            gap = bar - acc
            missed = f"P(<= {t}) {acc:.3f} < {bar:g}"

    return default, f"held at {default}" + (f": {missed}" if missed else "")


def replay_one(rec: dict, taus: dict, up_bar: float) -> dict:
    """What (b) would have done with the answer this record kept.

    Returns {"model", "why", "probs", "tiers"} or {"skip": reason}. A record is
    skippable for reasons that are not failures -- schema 1 predates the raw
    answer, and a front end that could not reach Jev logged no answer to
    replay. `tiers` rides along because the log mixes providers: a reader
    comparing this to the decision taken needs the ladder that record was
    decided on, not this process's.
    """
    jev = rec.get("jev")
    if not isinstance(jev, dict) or not isinstance(jev.get("answers"), dict):
        if rec.get("why"):
            return {"skip": "no decision was made"}
        return {"skip": "schema 1: no answer kept"}

    # The hash is recorded for exactly this: policy.QUESTIONS_HASH says two
    # records scored by the same questions can be compared and two scored by
    # different ones cannot. Reword one criterion without changing how many
    # there are and level_probs still accepts the answer, while `need_of` now
    # weights those probabilities by criteria the scorer never saw. That is
    # the one edit the level count cannot catch, and the retention window is
    # long enough to straddle it.
    if jev.get("questions") != policy.QUESTIONS_HASH:
        return {"skip": f"scored under question set {jev.get('questions')!r}, "
                        f"not {policy.QUESTIONS_HASH!r}"}

    try:
        prov = providers.get(rec.get("provider") or providers.DEFAULT)
    except ValueError as e:
        return {"skip": str(e)}
    tiers = prov.TIERS

    default = rec.get("default")
    if default not in tiers:
        # Off the ladder, or a record written under the other provider. Either
        # way there is no rung to move from.
        return {"skip": f"parent {default!r} is not on the {prov.NAME} ladder"}

    probs = rung_probs(jev["answers"], tiers)
    if probs is None:
        return {"skip": "level probabilities missing or malformed"}

    avail = [t for t in (rec.get("tiers") or tiers) if t in tiers] or [default]
    model, why = summed_gate(probs, default, tiers, avail, taus, up_bar)
    return {"model": model, "why": why, "probs": probs,
            "tiers": list(tiers), "default": default}


def rows(recs: list[dict], taus: dict, up_bar: float) -> list[dict]:
    out = []
    for i, rec in enumerate(recs, 1):
        got = replay_one(rec, taus, up_bar)
        out.append({"n": i, "label": rec.get("label") or "", "rec": rec,
                    "took": rec.get("model"), **got})
    return out


def cell(r: dict) -> str:
    """One row's (b) verdict, for the column explain.py borrows."""
    if "skip" in r:
        return "--"
    return r["model"] + ("" if r["model"] == r["took"] else " *")


def show(session: str, recs: list[dict], taus: dict, up_bar: float) -> int:
    rs = rows(recs, taus, up_bar)
    done = [r for r in rs if "skip" not in r]
    differ = [r for r in done if r["model"] != r["took"]]

    hdr = (f"{'decision':<30} {'(a) took':<9} {'(b)':<9} "
           f"{'P cheap':>7} {'P mid':>7} {'P dear':>7}  why (b)")
    print(f"\nsession {session}: {len(recs)} decisions, {len(done)} replayable")
    print(hdr)
    print("-" * len(hdr))
    for r in rs:
        name = f"{r['n']}  {r['label']}"[:30]
        if "skip" in r:
            print(f"{name:<30} {str(r['took'] or '--'):<9} {'--':<9} "
                  f"{'':>7} {'':>7} {'':>7}  {r['skip']}")
            continue
        # Insertion-ordered cheapest first: rung_probs seeds the dict from the
        # ladder. The heading reads cheap/mid/dear rather than naming rungs, so
        # a log holding both providers still lines up as one table.
        cheap, mid, dear = r["probs"].values()
        print(f"{name:<30} {r['took']:<9} {r['model']:<9} "
              f"{cheap:>7.3f} {mid:>7.3f} {dear:>7.3f}  {r['why']}")

    if not done:
        print("\nNothing to replay. Records keep Jev's answer from schema 2 on;\n"
              "anything logged before that has only the conclusion.")
        return 0

    print(f"\n{len(differ)} of {len(done)} disagree")
    for r in differ:
        print(f"  {r['n']:>3}  {r['label'][:34]:<34} "
              f"(a) {r['took']} -> (b) {r['model']}   {r['why']}")
    return len(differ)


def sweep(recs: list[dict], up_bar: float) -> None:
    """Where tau actually bites. One line per bar, over every record given.

    The useful reading is the flat stretch: a tau whose neighbours give the
    same answer is one the data is not sensitive to, and a cliff is where a
    guessed threshold would have been doing the deciding.
    """
    print(f"\n{'tau':>6} {'downgrades':>11} {'vs (a)':>8}  decisions")
    print("-" * 46)
    for step in range(50, 100, 5):
        tau = step / 100
        rs = [r for r in rows(recs, {"*": tau}, up_bar) if "skip" not in r]
        if not rs:
            print("nothing replayable")
            return
        down = sum(1 for r in rs
                   if r["tiers"].index(r["model"]) < r["tiers"].index(r["default"]))
        differ = sum(1 for r in rs if r["model"] != r["took"])
        print(f"{tau:>6.2f} {down:>11} {differ:>8}  {len(rs)}")


def known_tiers() -> dict[str, list[str]]:
    """Every ladder a logged record could have been decided on, by provider."""
    return {name: providers.get(name).TIERS for name in providers.KNOWN}


def a_bar(value: str, what: str) -> float:
    """A probability bar, refused unless it is one.

    Both gates accumulate before testing and test inclusively, so a bar of 0
    is cleared by the first rung examined -- the dearest for the upgrade, the
    cheapest for the downgrade. `--up-bar 0` reads as "no bar on upgrades" and
    would do the exact opposite: route every record to the dearest rung on no
    evidence at all.
    """
    try:
        bar = float(value)
    except ValueError as e:
        raise SystemExit(f"{what} {value!r}: not a number") from e
    if not math.isfinite(bar) or not 0 < bar <= 1:
        raise SystemExit(f"{what} {value!r}: a probability bar, in (0, 1]")
    return bar


def parse_tau(values: list[str]) -> dict:
    """`0.85`, or `opus>haiku=0.95`, repeatable. Global default under "*".

    Per transition because the two are not the same bet: opus -> sonnet keeps
    a capable model, opus -> haiku is the one that produces a confident wrong
    answer, and one bar for both prices them as though they were equal.
    """
    ladders = known_tiers()
    taus = {"*": TAU}
    for v in values:
        if "=" not in v:
            taus["*"] = a_bar(v, "--tau")
            continue
        where, _, num = v.partition("=")
        frm, sep, to = where.partition(">")
        if not sep:
            raise SystemExit(f"--tau {v!r}: expected 'from>to=value'")
        frm, to = frm.strip(), to.strip()
        # A bar on a transition that can never be consulted is almost always a
        # typo, and the table it produces is indistinguishable from one
        # computed at the bar you meant -- which for a tool whose whole output
        # is "what the bar did" is the worst way to be wrong.
        ladder = next((t for t in ladders.values() if frm in t and to in t), None)
        if ladder is None:
            raise SystemExit(
                f"--tau {v!r}: {frm!r} and {to!r} are not two rungs of one "
                f"ladder ({'; '.join(' '.join(t) for t in ladders.values())})")
        if ladder.index(to) >= ladder.index(frm):
            raise SystemExit(
                f"--tau {v!r}: only downgrades are gated, and {to!r} is not "
                f"below {frm!r}")
        taus[(frm, to)] = a_bar(num, "--tau")
    return taus


def main() -> None:
    p = argparse.ArgumentParser(
        description="replay a different downgrade gate over logged decisions")
    p.add_argument("--session", default="", help="a session other than the newest")
    p.add_argument("--all", action="store_true", help="every session on disk")
    p.add_argument("--tau", action="append", default=[], metavar="BAR",
                   help=f"downgrade bar: a number, or 'from>to=number'. "
                        f"Default {TAU}, and unmeasured")
    p.add_argument("--up-bar", type=lambda v: a_bar(v, "--up-bar"),
                   default=UP_BAR,
                   help=f"upgrade bar on P(rung >= r). Default {UP_BAR}")
    p.add_argument("--sweep", action="store_true",
                   help="disagreement count against tau, instead of a table")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    try:
        taus = parse_tau(args.tau)
    except ValueError as e:
        raise SystemExit(f"--tau: {e}") from e

    known = decisions.sessions()
    if not known:
        raise SystemExit(
            "Nothing recorded yet. Either front end writes one entry per\n"
            "dispatch, so this fills up on its own.")

    picked = ([s["session"] for s in known] if args.all
              else [args.session or known[0]["session"]])

    if args.sweep:
        if args.tau:
            # --sweep sets the bar itself, one line per value. Silently
            # ignoring the bar you asked for would print a table that answers
            # a different question than the one on the command line.
            raise SystemExit("--sweep sets tau itself; drop --tau")
        every = [r for s in picked for r in decisions.read(s)]
        sweep(every, args.up_bar)
        return

    if args.json:
        out = {s: [{k: v for k, v in r.items() if k != "rec"}
                   for r in rows(decisions.read(s), taus, args.up_bar)]
               for s in picked}
        # allow_nan=False: the default emits a bare NaN token, which is not
        # JSON -- JSON.parse rejects it and jq silently reads it as null. If
        # one ever reaches here the guards above have a hole, and an error is
        # the right way to find out.
        json.dump(out, sys.stdout, indent=2, allow_nan=False)
        print()
        return

    for s in picked:
        recs = decisions.read(s)
        if not recs:
            # Distinct from an empty session: explain.py says the same thing
            # rather than exiting 0 on a typo'd id.
            print(f"no decisions recorded for session {s}")
            continue
        show(s, recs, taus, args.up_bar)


if __name__ == "__main__":
    main()
