#!/usr/bin/env python3
"""Exercise route.py, the Jev client and the decision log without calling Jev.

The network is replaced at the lowest seam each case needs -- urlopen for the
client, policy.system_one for the cache, policy.route for the fan-out -- so
this runs offline, needs no key, and costs nothing. The Jev answer it replays
is a real one: what jev-1.14.0 returned on 5 November 2026 for "Rename the
Deadline field to DueAt and fix every caller", legend and all.

    ./selftest_route.py    # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import traceback
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.pop("JEV_PROVIDER", None)  # these cases are about the Claude ladder
os.environ.pop("JEV_MODEL", None)
os.environ.setdefault("TYPESAFE_API_KEY", "selftest-not-a-key")
import decisions  # noqa: E402
import jev  # noqa: E402
import policy  # noqa: E402
import replay  # noqa: E402
import route  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

_LEGEND = {"depth": [c for c in policy.QUESTIONS["depth"]["criteria"]],
           "breadth": [c for c in policy.QUESTIONS["breadth"]["criteria"]]}
REAL = {
    "model": "jev-1.14.0",
    "answers": {
        "depth": {"type": "score", "score": 1.94, "confidence": 0.7,
                  "legend": dict(enumerate(_LEGEND["depth"])),
                  "probabilities": {"0": 0.04, "1": 0.13, "2": 0.68, "3": 0.15, "4": 0.0}},
        "breadth": {"type": "score", "score": 3.02, "confidence": 0.77,
                    "legend": dict(enumerate(_LEGEND["breadth"])),
                    "probabilities": {"0": 0.03, "1": 0.01, "2": 0.02, "3": 0.79, "4": 0.15}},
        "mechanical": {"type": "noul", "noul": 0.88},
        "irreversible": {"type": "noul", "noul": 0.09},
        "ambiguous": {"type": "noul", "noul": 0.66},
    },
    "usage": {"input_tokens": 773, "output_tokens": 137},
}
TASK = "Rename the Deadline field to DueAt and fix every caller."


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def client_fails_as(exc) -> str:
    """What system_one raises when urlopen raises `exc` (or returns bytes)."""
    def urlopen(req, timeout=None):
        if isinstance(exc, bytes):
            return FakeResponse(exc)
        raise exc
    with patched(jev.urllib.request, "urlopen", urlopen):
        try:
            jev.system_one("x", {"q": {"type": "noul", "instructions": "x"}})
        except SystemExit as e:
            return f"SystemExit: {e}"
        except BaseException as e:  # noqa: BLE001 -- the point is to see what escapes
            return f"{type(e).__name__}: {e}"
    return "returned"


def _run(tmp: pathlib.Path, cases: list) -> None:
    """Every case, appended to `cases` as it is decided.

    The list is the caller's so that an exception escaping one case does not
    take the ones already decided with it -- see main().
    """
    def check(name, ok, detail=""):
        cases.append((name, ok, detail))

    # --- the client: every failure is one SystemExit with a reason ---------
    for name, exc, says in [
        ("a refused connection", urllib.error.URLError("Connection refused"), "unreachable"),
        ("a timeout", socket.timeout("timed out"), "unreachable"),
        ("a body that is not JSON", b"<html>bad gateway</html>", "not JSON"),
    ]:
        got = client_fails_as(exc)
        check(f"client: {name} is a SystemExit", got.startswith("SystemExit") and says in got,
              got[:90])

    # --- the client asks for the pinned version, or the one you name -------
    sent = []

    def capture(req, timeout=None):
        sent.append(json.loads(req.data))
        return FakeResponse(json.dumps(REAL).encode())
    with patched(jev.urllib.request, "urlopen", capture):
        with env(JEV_MODEL=None):
            jev.system_one("x", {})
        with env(JEV_MODEL="jev-1.15.0"):
            jev.system_one("x", {})
    check("client: the pinned version is requested by default",
          sent and sent[0]["model"] == jev.DEFAULT_MODEL != "jev-latest", f"sent {sent[:1]}")
    check("client: JEV_MODEL overrides the pin",
          len(sent) == 2 and sent[1]["model"] == "jev-1.15.0", f"sent {sent[1:]}")

    # --- the cache: keyed on the requested model, never stores a bad answer -
    calls = []

    def fake_jev(state, questions, model=None, timeout=30.0):
        calls.append(model)
        return copy.deepcopy(REAL)
    with patched(policy, "system_one", fake_jev):
        with env(JEV_MODEL=None):
            first = policy.ask_meta(TASK)
            again = policy.ask_meta(TASK)
        with env(JEV_MODEL="jev-1.15.0"):
            other = policy.ask_meta(TASK)
    check("cache: a second ask is served from disk, model and all",
          not first["cached"] and again["cached"] and again["model"] == "jev-1.14.0"
          and len(calls) == 2, f"calls {calls} cached {first['cached']}/{again['cached']}")
    check("cache: a different pinned model is a different entry",
          not other["cached"] and calls == [jev.DEFAULT_MODEL, "jev-1.15.0"],
          f"calls {calls}")

    # The bare answers object is how the old code wrote an entry. Putting the
    # requested model into the key means nothing it wrote can be found at a
    # path the new code looks up, so the branch that used to read that shape
    # was unreachable. What has to hold now is that such a file is refetched
    # rather than served -- which is also what happens to any entry too
    # damaged to read.
    legacy_task = "legacy-format task"
    policy._cache_path(legacy_task, jev.DEFAULT_MODEL).write_text(
        json.dumps(REAL["answers"]))
    with patched(policy, "system_one", fake_jev):
        legacy = policy.ask_meta(legacy_task)
    check("cache: a bare-answers entry is refetched, not served",
          not legacy["cached"] and legacy["model"] == "jev-1.14.0",
          f"{legacy!r:.80}")

    # An entry that cannot be read at all. This used to escape as OSError from
    # outside the try and disable routing for the task until the file expired.
    unreadable_task = "an entry we cannot open"
    bad_path = policy._cache_path(unreadable_task, jev.DEFAULT_MODEL)
    bad_path.write_text(json.dumps(REAL))
    bad_path.chmod(0o000)
    try:
        with patched(policy, "system_one", fake_jev):
            recovered = policy.ask_meta(unreadable_task)
    finally:
        bad_path.chmod(0o600)
    check("cache: an unreadable entry refetches rather than raising",
          not recovered["cached"] and recovered["model"] == "jev-1.14.0",
          f"{recovered!r:.80}")

    broken = copy.deepcopy(REAL)
    del broken["answers"]["breadth"]
    broken_task = "a task Jev answered badly"
    with patched(policy, "system_one", lambda *a, **k: copy.deepcopy(broken)):
        try:
            policy.ask_meta(broken_task)
            raised = "nothing"
        except ValueError as e:
            raised = str(e)
    check("cache: an answer missing a question raises and is not stored",
          "breadth" in raised
          and not policy._cache_path(broken_task, jev.DEFAULT_MODEL).exists(),
          f"raised {raised!r}")

    stringly = copy.deepcopy(REAL)
    stringly["answers"]["mechanical"]["noul"] = "0.88"
    try:
        policy.check_answers(stringly["answers"])
        raised = "nothing"
    except ValueError as e:
        raised = str(e)
    check("cache: a number sent as a string is refused", "mechanical.noul" in raised,
          f"raised {raised!r}")

    # --- route(): the decision carries what was measured --------------------
    with patched(policy, "system_one", fake_jev), env(JEV_MODEL=None):
        plan = policy.route(TASK, default="opus")
    j = plan.get("jev") or {}
    check("route: the real answer still routes opus -> sonnet",
          plan["model"] == "sonnet" and abs(plan["need"] - 0.373) < 0.001,
          f"model {plan['model']} need {plan['need']}")
    check("route: the raw answer rides along, with version and question hash",
          j.get("model") == "jev-1.14.0" and j.get("requested") == jev.DEFAULT_MODEL
          and j.get("questions") == policy.QUESTIONS_HASH
          and j["answers"]["depth"]["probabilities"]["2"] == 0.68,
          f"jev {json.dumps(j)[:100]}")
    check("route: the legend is not repeated into every record",
          all("legend" not in a for a in j.get("answers", {}).values()), "")

    # --- the fan-out: one bad shard costs one shard -------------------------
    def flaky(state, default=None, cache=True):
        if "keyerror" in state:
            raise KeyError("breadth")
        if "network" in state:
            raise urllib.error.URLError("Connection refused")
        if "timeout" in state:
            raise TimeoutError("timed out")
        if "http" in state:
            raise SystemExit("jev: HTTP 500: boom")
        return {**policy.decide(REAL["answers"], default=default), "jev": {}}

    shards = [{"id": i, "task": i, "dispatch": f"toolu_{i}"}
              for i in ("ok", "keyerror", "network", "timeout", "http")]
    out = io.StringIO()
    with patched(policy, "route", flaky), \
            patched(sys, "argv", ["route.py", "--default", "opus", "--session", "fan", "--json"]), \
            patched(sys, "stdin", io.StringIO(json.dumps(shards))), \
            contextlib.redirect_stdout(out):
        try:
            route.main()
            crashed = ""
        except BaseException as e:  # noqa: BLE001
            crashed = f"{type(e).__name__}: {e}"
    routed = json.loads(out.getvalue()) if not crashed else []
    by_id = {r["id"]: r for r in routed}
    check("fan-out: every shard comes back, whatever its own failure",
          not crashed and len(routed) == 5, crashed or f"{len(routed)} shards")
    check("fan-out: the good shard is still routed",
          by_id.get("ok", {}).get("model") == "sonnet", f"{by_id.get('ok', {}).get('model')}")
    check("fan-out: a failed shard runs on the default and says why",
          all(by_id.get(i, {}).get("model") == "opus" and "jev error" in by_id[i].get("why", "")
              for i in ("keyerror", "network", "timeout", "http"))
          and "KeyError" in by_id.get("keyerror", {}).get("why", ""),
          f"{[(i, by_id.get(i, {}).get('why')) for i in ('keyerror', 'network')]}")

    logged = decisions.read("fan")
    check("fan-out: the failures are logged too", len(logged) == 5,
          f"{len(logged)} records")
    rec = next((r for r in logged if r["label"] == "ok"), {})
    check("log: a record carries schema, provider, dispatch id and an empty outcome",
          rec.get("schema") == decisions.SCHEMA == 2 and rec.get("provider") == "claude"
          and rec.get("dispatch") == "toolu_ok" and "outcome" in rec
          and rec["outcome"] is None, f"{ {k: rec.get(k) for k in ('schema', 'provider', 'dispatch', 'outcome')} }")

    # The replay renders a log mixing old and new records.
    old = {"at": 1.0, "label": "pre-schema", "default": "opus", "task": "t",
           "task_len": 1, **policy.decide(REAL["answers"], default="opus")}
    with open(decisions.path_for("mixed"), "w") as f:
        f.write(json.dumps(old) + "\n" + json.dumps(rec) + "\n")
    shown = subprocess.run([sys.executable, str(HERE / "explain.py"), "--session", "mixed"],
                           capture_output=True, text=True, timeout=60,
                           env={**os.environ, "JEV_DECISIONS_DIR": str(decisions.LOG_DIR)})
    check("explain: a log mixing schema 1 and 2 still replays",
          shown.returncode == 0 and "pre-schema" in shown.stdout,
          (shown.stderr or shown.stdout)[-120:])

    # --- replay.py: the summed gate, over a record carrying a real answer ----
    # Built from `plan` rather than from the fan-out's records: those went
    # through a stub that returns no `jev`, which is itself one of the cases
    # below. This is the shape decisions.record writes.
    full = {"schema": decisions.SCHEMA, "at": 2.0, "label": "rename DueAt",
            "dispatch": "toolu_r", "provider": "claude", "default": "opus",
            "task": TASK, "task_len": len(TASK), "outcome": None, **plan}

    # Hand-computed from REAL, which is why the fixture is a real answer:
    # need(i,j) = 0.15i + 0.10j - 0.22 against the 0.25 and 0.60 lines, summed
    # over the outer product of depth .04/.13/.68/.15/0 and breadth
    # .03/.01/.02/.79/.15. If the weights or the lines move, these move with
    # them and the case is meant to fail.
    probs = replay.rung_probs(full["jev"]["answers"], ["haiku", "sonnet", "opus"])
    check("replay: the rung distribution is the outer product of the levels",
          probs is not None and all(abs(probs[t] - want) < 5e-4 for t, want in
                                    (("haiku", .1822), ("sonnet", .7953), ("opus", .0225))),
          f"{probs}")

    at = lambda tau, r=full: replay.replay_one(r, {"*": tau}, replay.UP_BAR)  # noqa: E731
    check("replay: a bar the mass clears downgrades, agreeing with the gate that ran",
          at(0.70)["model"] == full["model"] == "sonnet", f"{at(0.70)}")
    check("replay: a bar it does not clear holds at the parent, and says by how much",
          at(0.99)["model"] == "opus" and "0.977 < 0.99" in at(0.99)["why"],
          f"{at(0.99)}")
    # The cheapest rung that clears, not one step at a time: at a bar below
    # P(haiku) the same answer should skip sonnet entirely.
    check("replay: a very low bar takes the cheapest rung that clears, not the next one",
          at(0.15)["model"] == "haiku", f"{at(0.15)}")
    # ... but never past what the endpoint served. The probabilities are
    # computed on the full ladder either way, so the clamp must not move them.
    unserved = {**full, "tiers": ["sonnet", "opus"]}
    got = replay.replay_one(unserved, {"*": 0.15}, replay.UP_BAR)
    check("replay: a rung the endpoint does not serve is clamped up, not emitted",
          got["model"] == "sonnet" and abs(got["probs"]["haiku"] - .1822) < 5e-4,
          f"{got}")

    # The upgrade arm, which a top-rung parent can never reach: every case
    # above dispatches from opus, so `tiers.index(t) > here` is false on all of
    # them and the whole block could be deleted without turning the suite red.
    from_haiku = {**copy.deepcopy(full), "default": "haiku", "model": "haiku"}
    up = replay.replay_one(from_haiku, {"*": 0.70}, replay.UP_BAR)
    check("replay: mass above the parent upgrades, and names the bar it cleared",
          up["model"] == "sonnet" and "P(>= sonnet)" in up["why"]
          and ">= 0.5" in up["why"], f"{up}")
    # P(>= opus) is .0225 here, so only a bar under that reaches opus. This
    # fails if the upgrade takes the first rung it finds rather than the
    # dearest one clearing the bar.
    check("replay: the upgrade takes the dearest rung the mass clears",
          replay.replay_one(from_haiku, {"*": 0.70}, 0.02)["model"] == "opus",
          f"{replay.replay_one(from_haiku, {'*': 0.70}, 0.02)}")
    # ... and is clamped to what the endpoint serves, exactly as a downgrade is.
    thin = {**from_haiku, "tiers": ["haiku", "opus"]}
    check("replay: an upgrade to a rung not served is clamped up too",
          replay.replay_one(thin, {"*": 0.70}, replay.UP_BAR)["model"] == "opus",
          f"{replay.replay_one(thin, {'*': 0.70}, replay.UP_BAR)}")

    # A per-transition bar has to beat the global one, or --tau 'opus>haiku='
    # is decoration. At a global 0.15 this answer reaches haiku (above); naming
    # that transition alone must hold it at sonnet without touching the rest.
    split = replay.replay_one(full, {"*": 0.15, ("opus", "haiku"): 0.95},
                              replay.UP_BAR)
    check("replay: a bar on one transition binds only that transition",
          split["model"] == "sonnet" and "P(<= sonnet) 0.977" in split["why"],
          f"{split}")
    # The near miss is the smallest gap, not the last rung tried. haiku misses
    # 0.95 by .768 and sonnet misses 0.99 by .013, so sonnet is the one to say.
    near = replay.replay_one(full, {"*": 0.95, ("opus", "sonnet"): 0.99},
                             replay.UP_BAR)
    check("replay: a hold reports the bar it came closest to clearing",
          near["model"] == "opus" and "P(<= sonnet) 0.977 < 0.99" in near["why"],
          f"{near}")

    for name, broken in [
        ("levels that do not sum to 1", {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1}),
        ("a level that is not there", {"0": 0.2, "1": 0.3, "2": 0.5}),
        ("a label the questions do not have", {"0": .04, "1": .13, "2": .68,
                                               "3": .15, "4": 0.0, "5": 0.0}),
        # Every comparison against NaN is False, so this passed the sign test
        # and the sum test and arrived at rung_of as a need that falls through
        # both lines to the cheapest rung.
        ("a level that is NaN", {"0": float("nan"), "1": 9.0, "2": 9.0,
                                 "3": 9.0, "4": 9.0}),
        ("a level that is negative", {"0": -.04, "1": .17, "2": .68,
                                      "3": .15, "4": 0.04}),
        ("a level sent as a string", {"0": "0.04", "1": .13, "2": .68,
                                      "3": .15, "4": 0.0}),
    ]:
        bad = copy.deepcopy(full)
        bad["jev"]["answers"]["depth"]["probabilities"] = broken
        out = replay.replay_one(bad, {"*": 0.70}, replay.UP_BAR)
        check(f"replay: {name} is skipped, not renormalised into a decision",
              "skip" in out and "model" not in out, f"{out}")

    # ... but a miss inside SUM_TOLERANCE is accepted and renormalised, which
    # is the whole reason the tolerance is not zero.
    near_one = copy.deepcopy(full)
    near_one["jev"]["answers"]["depth"]["probabilities"] = {
        "0": .04, "1": .13, "2": .68, "3": .15, "4": .02}
    out = replay.replay_one(near_one, {"*": 0.70}, replay.UP_BAR)
    check("replay: levels missing 1 by less than the tolerance are renormalised",
          out.get("model") == "sonnet"
          and abs(sum(out["probs"].values()) - 1.0) < 1e-9, f"{out}")

    # A malformed answer object, rather than malformed probabilities inside
    # one. This used to be an AttributeError that took the whole session with
    # it -- including the rows that were fine.
    for name, bad_answer in [("a bare float", 0.88), ("a list", ["x"]),
                             ("a string", "x")]:
        bad = copy.deepcopy(full)
        bad["jev"]["answers"]["mechanical"] = bad_answer
        out = replay.replay_one(bad, {"*": 0.70}, replay.UP_BAR)
        check(f"replay: an answer that is {name} is skipped, not raised",
              "skip" in out, f"{out}")

    # The hash is why it is recorded: rewording one criterion leaves the level
    # count intact, so nothing else here can tell that the probabilities were
    # given to different questions.
    stale = copy.deepcopy(full)
    stale["jev"]["questions"] = "0123456789ab"
    out = replay.replay_one(stale, {"*": 0.70}, replay.UP_BAR)
    check("replay: an answer scored under other questions is skipped",
          "skip" in out and "question set" in out["skip"], f"{out}")

    # The bars are probabilities. Both gates accumulate before testing and
    # test inclusively, so a bar of 0 is cleared by the first rung examined --
    # `--up-bar 0` reads as "no upgrade bar" and would mean "always opus".
    for bad_bar in ["0", "-1", "1.5", "nan", "abc"]:
        try:
            replay.parse_tau([bad_bar])
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check(f"replay: --tau {bad_bar} is refused, not used as a bar",
              "probability bar" in refused or "not a number" in refused,
              f"{refused!r}")
    # A transition that can never be consulted is almost always a typo, and
    # the table it produces is indistinguishable from one computed at the bar
    # you meant.
    for bad_pair in ["sonet>haiku=0.95", "opus>opus=0.5", "haiku>opus=0.9",
                     "opus>luna=0.9"]:
        try:
            replay.parse_tau([bad_pair])
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check(f"replay: --tau {bad_pair} is refused rather than ignored",
              bool(refused), f"{refused!r}")
    check("replay: a real downgrade transition is accepted, on either ladder",
          replay.parse_tau(["opus>haiku=0.95"])[("opus", "haiku")] == 0.95
          and replay.parse_tau(["astra>luna=0.9"])[("astra", "luna")] == 0.9)

    # A record written under the other provider replays on that ladder, not on
    # this process's. One log holds both.
    codex_rec = {**copy.deepcopy(full), "provider": "codex", "default": "astra",
                 "model": "sol", "tiers": ["luna", "sol", "astra"]}
    got = replay.replay_one(codex_rec, {"*": 0.70}, replay.UP_BAR)
    check("replay: a codex record replays on the codex ladder",
          got.get("model") == "sol" and got.get("tiers") == ["luna", "sol", "astra"],
          f"{got}")

    # Three rows: one schema 1 with nothing to replay, one where (b) agrees,
    # and one where it does not. The disagreeing row is what pins the `*` --
    # without it, marking every row and marking none both pass.
    held_rec = {**copy.deepcopy(full), "label": "deploy the bump",
                "model": "opus", "held": True, "suggested": "sonnet"}
    with open(decisions.path_for("replayed"), "w") as f:
        for r in (old, full, held_rec):
            f.write(json.dumps(r) + "\n")
    shown = subprocess.run([sys.executable, str(HERE / "explain.py"),
                            "--session", "replayed", "--replay"],
                           capture_output=True, text=True, timeout=60,
                           env={**os.environ, "JEV_DECISIONS_DIR": str(decisions.LOG_DIR)})
    out = shown.stdout
    # Read the column by its offset in the header rather than by searching the
    # whole row: "does the word sonnet appear twice" is satisfied just as well
    # by a renderer that prints the replay column in the wrong place.
    hdr = next((ln for ln in out.splitlines() if ln.startswith("decision")), "")
    at = hdr.find("(b)")
    # The numbered rows only: the tally line below the table also starts with
    # a digit.
    rows_by_label = {ln.split("  ")[1].strip(): ln for ln in out.splitlines()
                     if ln.startswith(("1  ", "2  ", "3  "))}
    col = {k: v[at:at + 9].strip() for k, v in rows_by_label.items()} if at > 0 else {}
    check("explain --replay: the (b) column sits between conf and advice",
          shown.returncode == 0 and at > hdr.find("conf") > 0
          and at < hdr.find("advice"), f"header {hdr!r}")
    check("explain --replay: a record with no answer kept blanks the column",
          col.get("pre-schema") == "--", f"{col}")
    check("explain --replay: a row the gate agrees with is unmarked",
          col.get("rename DueAt") == "sonnet", f"{col}")
    check("explain --replay: a row it disagrees with is marked",
          col.get("deploy the bump") == "sonnet *", f"{col}")
    # The footnote and the heading are separate producers of the string "(b)",
    # so each has to be asserted on its own or neither is pinned.
    check("explain --replay: the footnote names the tau the column used",
          f"unmeasured tau of {replay.TAU}" in out, out[-200:])
    # The views with nowhere to put a column say so rather than ignoring it.
    for flag in ("--json", "--last", "--sessions"):
        refused = subprocess.run(
            [sys.executable, str(HERE / "explain.py"), "--session", "replayed",
             "--replay", flag], capture_output=True, text=True, timeout=60,
            env={**os.environ, "JEV_DECISIONS_DIR": str(decisions.LOG_DIR)})
        check(f"explain --replay {flag}: refused, not silently dropped",
              refused.returncode != 0 and "nowhere to go" in
              (refused.stderr + refused.stdout), f"{refused.returncode} "
              f"{(refused.stderr or refused.stdout)[-90:]}")

    # --- the default tier follows the provider -------------------------------
    helped = subprocess.run([sys.executable, str(HERE / "route.py"), "--help"],
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, "JEV_PROVIDER": "codex"})
    check("route.py: under codex the default tier is the middle rung",
          "(default sol)" in " ".join(helped.stdout.split()), helped.stdout[-120:])

    # --- JEV_TIERS: a typo narrows nothing and widens nothing silently -------
    def tiers_with(value):
        err = io.StringIO()
        with env(JEV_TIERS=value), contextlib.redirect_stderr(err):
            return policy.available_tiers(), err.getvalue()

    got, warned = tiers_with("haiku,opus")
    check("JEV_TIERS: a list of real rungs is honoured, in ladder order",
          got == ["haiku", "opus"] and not warned, f"{got} {warned!r}")
    got, warned = tiers_with("haiku, opus ")
    check("JEV_TIERS: spaces around the names are not part of them",
          got == ["haiku", "opus"] and not warned, f"{got} {warned!r}")
    # The regression: one transposed letter used to route everything to haiku.
    # With no probe record to fall back to, a rejected list must not widen
    # either -- the full ladder is the un-probed default, and handing it to the
    # one input that claimed to know the ladder is the wider wrong answer.
    got, warned = tiers_with("haiku,sonet")
    check("JEV_TIERS: a typo'd name does not silently narrow the ladder",
          got == [] and "sonet" in warned, f"{got} {warned!r}")
    got, warned = tiers_with("sonet")
    check("JEV_TIERS: a wholly unknown list is refused out loud",
          got == [] and "sonet" in warned, f"{got} {warned!r}")
    got, warned = tiers_with("luna,sol,astra")
    check("JEV_TIERS: another provider's rungs name the ladder they missed",
          got == [] and "haiku, sonnet, opus" in warned, f"{got} {warned!r}")
    got, warned = tiers_with(" , ")
    check("JEV_TIERS: punctuation alone is not an empty ladder",
          got == [] and warned != "", f"{got} {warned!r}")
    # An unset variable is not a rejected one: that caller never claimed to
    # know the ladder, so the un-probed full ladder is still right for it.
    err = io.StringIO()
    with env(JEV_TIERS=None), contextlib.redirect_stderr(err):
        check("JEV_TIERS: unset still means the un-probed full ladder",
              policy.available_tiers() == policy.TIERS and not err.getvalue(),
              f"{policy.available_tiers()} {err.getvalue()!r}")
    # And a rejected list falls through to a measurement when there is one,
    # which is what the warning promises. Every case above runs against an
    # empty CACHE_DIR, so without this the promise is never exercised.
    (policy.CACHE_DIR / "tiers.json").write_text(json.dumps(
        {"tiers": ["sonnet", "opus"], "resolved": {"sonnet": "(built-in default)",
                                                   "opus": "(built-in default)"}}))
    try:
        got, warned = tiers_with("haiku,sonet")
        check("JEV_TIERS: a rejected list falls through to the probe record",
              got == ["sonnet", "opus"] and "sonet" in warned, f"{got} {warned!r}")
    finally:
        (policy.CACHE_DIR / "tiers.json").unlink()

    # --- the state cap is applied once, where every path reaches it ----------
    seen = {}

    def capture(state, questions, model=None):
        seen["state"] = state
        return copy.deepcopy(REAL)

    with patched(policy, "system_one", capture):
        policy.ask_meta("x" * (policy.STATE_LIMIT * 2), cache=False)
    check("state cap: policy.ask_meta cuts to STATE_LIMIT",
          len(seen["state"]) == policy.STATE_LIMIT, len(seen.get("state", "")))
    # Spelled out rather than compared to the constant. `len(state) ==
    # policy.STATE_LIMIT` passes for any value the constant happens to hold,
    # and the whole point of this PR is that the number exists exactly once --
    # which leaves nothing else in the tree asserting what it is.
    check("state cap: and STATE_LIMIT is 60,000",
          policy.STATE_LIMIT == 60_000, policy.STATE_LIMIT)
    # The other half of "exactly once": nothing stops a front end from cutting
    # the state again on its way in, and three of them used to. A re-spelled
    # literal is invisible to every behavioural test on the surfaces that have
    # one, and tests/register.test.ts cannot run below Claude Code 2.1.287 --
    # so the mod is only reachable from here as text.
    respelled = []
    for name in ("hook_agent.py", "hook_codex.py", "hooks/register.js",
                 "route.py"):
        body = (HERE / name).read_text()
        for digits in ("60_000", "60000"):
            # Not in a comment saying it used to be there, which is how two of
            # these files explain the removal.
            if any(digits in ln and not ln.lstrip().startswith(("#", "//"))
                   for ln in body.splitlines()):
                respelled.append(f"{name}:{digits}")
    check("state cap: and no front end spells it a second time", not respelled,
          respelled)

    # The reason it is cut before the cache key and not after: two states that
    # differ only past the cap are the same request, and must not be billed or
    # stored twice. Needs cache=True, which is why the cases around it -- all
    # cache=False -- could not see the ordering at all.
    calls = []

    def counted(state, questions, model=None):
        calls.append(state)
        return copy.deepcopy(REAL)

    with env(JEV_API_KEY="selftest"), patched(policy, "system_one", counted):
        long_a = "q" * policy.STATE_LIMIT + "...tail A"
        long_b = "q" * policy.STATE_LIMIT + "...tail B, quite different"
        policy.ask_meta(long_a, cache=True)
        policy.ask_meta(long_b, cache=True)
    check("state cap: it is applied before the cache key, so one task is one key",
          len(calls) == 1, f"{len(calls)} requests for two states sharing a prefix")

    seen.clear()
    with patched(policy, "system_one", capture):
        # The gap this closes: route.py applied no cap of its own, and it is
        # both the fan-out and what the mod shells out to.
        route.route_one({"id": "big", "task": "y" * (policy.STATE_LIMIT * 2)},
                        policy.TIERS[1], False)
    check("state cap: route.py's fan-out is bounded by it too",
          len(seen["state"]) == policy.STATE_LIMIT, len(seen.get("state", "")))

    sent = {}

    def urlopen_capture(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return FakeResponse(json.dumps(REAL).encode())

    with patched(jev.urllib.request, "urlopen", urlopen_capture):
        jev.system_one("z" * (jev.STATE_CHAR_LIMIT * 2), policy.QUESTIONS)
    check("state cap: jev.system_one holds the API ceiling for every caller",
          len(sent["body"]["state"]) == jev.STATE_CHAR_LIMIT,
          len(sent.get("body", {}).get("state", "")))

    # --- the one line a PreToolUse hook gets to say --------------------------
    plan = policy.decide(REAL["answers"], default="opus",
                         tiers=["haiku", "sonnet", "opus"])
    line = policy.decision_reason("opus", plan)
    # The whole line, not a prefix. Asserting `startswith("... (need ")` stops
    # at the paren and never looks at the number, so printing conf in the need
    # slot read as correct.
    check("reason: it names the move, the need and the confidence",
          line == f"jev: opus -> {plan['model']} "
                  f"(need {plan['need']}, conf {plan['conf']})", line)
    check("reason: a reversible dispatch gets no warning bolted on",
          "warning" not in line, line)
    risky = copy.deepcopy(REAL["answers"])
    risky["irreversible"]["noul"] = 0.94
    warn = policy.decision_reason("opus", policy.decide(
        risky, default="opus", tiers=["haiku", "sonnet", "opus"]))
    # The exact wording, because the README and the PR body both quote it. The
    # looser `"not a gate" in warn` let the words around it be reworded freely,
    # and the negative case's `"warning" not in line` then went vacuously true.
    check("reason: an irreversible one carries the warning and its score",
          warn.endswith("; writes outside the working tree "
                        "(irreversible 0.94) -- a warning, not a gate"), warn)
    check("reason: a plan with no signals still renders",
          "not a gate" in policy.decision_reason(
              "opus", {"model": "haiku", "need": 0, "conf": 0,
                       "advice": {"confirm": True}}), "")

    # --- probe.py: unreachable is a third answer, not 'not served' -----------
    import probe  # noqa: PLC0415 -- only this block needs it

    def post_raising(exc, model=""):
        def urlopen(req, timeout=None):
            raise exc
        with patched(probe.urllib.request, "urlopen", urlopen):
            return probe.post("https://example.invalid", {}, {}, model)

    gone, _ = post_raising(urllib.error.HTTPError(
        "u", 404, "Not Found", {}, io.BytesIO(b"no such model")))
    check("probe: a 404 is a tier that is not served", gone is False, gone)
    served, _ = post_raising(urllib.error.HTTPError(
        "u", 400, "Bad Request", {}, io.BytesIO(b"bad body")))
    check("probe: a 400 reached a model, so the tier is served",
          served is True, served)
    # The regression: this used to be False, and main() wrote the tier off.
    for name, exc in (("a timeout", socket.timeout("timed out")),
                      ("a dead network", urllib.error.URLError("no route"))):
        unknown, why = post_raising(exc)
        check(f"probe: {name} is unknown, not absent",
              unknown is None, f"{unknown} {why}")
    # The half of the tri-state that shipped unapplied: every status except
    # 403/404 was read as "served", so an expired key wrote a ladder nothing
    # had confirmed.
    for code, label in ((401, "an expired key"), (429, "a rate limit"),
                        (503, "a bad gateway")):
        got, why = post_raising(urllib.error.HTTPError(
            "u", code, label, {}, io.BytesIO(b"")))
        check(f"probe: {label} ({code}) is unknown, not a confirmed tier",
              got is None, f"{got} {why}")
    # A 403 is two facts in one code: only the one that names the model is an
    # answer about the model.
    named, _ = post_raising(urllib.error.HTTPError(
        "u", 403, "Forbidden", {}, io.BytesIO(b"model claude-x not allowed")),
        model="claude-x")
    check("probe: a 403 naming the model rules the tier out",
          named is False, named)
    vague, _ = post_raising(urllib.error.HTTPError(
        "u", 403, "Forbidden", {}, io.BytesIO(b"caller lacks aiplatform.user")),
        model="claude-x")
    check("probe: a 403 about the caller is unknown, not a missing tier",
          vague is None, vague)
    # The model is what makes the distinction readable, so a caller that does
    # not pass one gets the cautious half of it rather than the confident one.
    anon, _ = post_raising(urllib.error.HTTPError(
        "u", 403, "Forbidden", {}, io.BytesIO(b"model claude-x not allowed")))
    check("probe: a 403 with no model to match on stays unknown",
          anon is None, anon)

    # main()'s half of it. The tri-state only matters because it changes what
    # gets written, and nothing reached main() at all: deleting the entire
    # refuse-to-record block left every suite green.
    runs = itertools.count()

    def probe_main(answers, prior=None):
        """Run probe.main with the network stubbed, against a temp record.

        A fresh file per call, counted rather than derived from `answers`: the
        no-prior case asks the same questions as the prior case above it, so a
        name hashed from the answers handed it that case's written record and
        it passed while testing nothing.
        """
        rec = tmp / f"tiers-{next(runs)}.json"
        if prior is not None:
            rec.write_text(json.dumps(prior))
        out = io.StringIO()
        with patched(probe, "TIERS_FILE", rec), \
                patched(probe, "pinned", lambda t: f"model-{t}"), \
                patched(probe, "endpoint", lambda: "anthropic"), \
                patched(probe, "probe_anthropic",
                        lambda m: (answers[m.removeprefix("model-")], "why")), \
                contextlib.redirect_stdout(out):
            rc = probe.main()
        saved = json.loads(rec.read_text()) if rec.is_file() else None
        return rc, saved, out.getvalue()

    rc, saved, said = probe_main({"haiku": True, "sonnet": True, "opus": None},
                                 prior={"tiers": ["haiku", "sonnet", "opus"]})
    check("probe: one unreachable tier does not overwrite a good record",
          rc == 1 and saved == {"tiers": ["haiku", "sonnet", "opus"]},
          f"{rc} {saved}")
    check("probe: and the unreachable tier prints as unknown, not as gone",
          "????" in said and "GONE" not in said, said[-120:])
    # The claim the message makes has to be true. With nothing on file there is
    # no previous measurement to keep, and no record reads as the full ladder.
    rc, saved, said = probe_main({"haiku": True, "sonnet": True, "opus": None})
    check("probe: with no prior record it keeps what did confirm",
          saved is not None and saved["tiers"] == ["haiku", "sonnet"],
          f"{rc} {saved}")
    # Written, but not a clean measurement: the text says to re-run, and the
    # exit code has to say the same thing to whatever ran it.
    check("probe: and still exits non-zero, having not reached every tier",
          rc == 1, rc)
    rc, saved, said = probe_main({"haiku": False, "sonnet": False, "opus": False})
    check("probe: a tier that answered 'not served' still reads as GONE",
          "GONE" in said and saved is None, f"{saved} {said[-90:]}")

    # --- the transcript tail reader ------------------------------------------
    from providers import claude as claude_provider  # noqa: PLC0415

    tail_file = tmp / "transcript.jsonl"
    turns = [json.dumps({"type": "assistant", "message": {"model": m}})
             for m in ["claude-haiku-4-5"] * 5000 + ["claude-opus-5"]]
    tail_file.write_text("\n".join(turns) + "\n")
    check("tail: the newest assistant turn wins over 5,000 older ones",
          claude_provider.session_tier(str(tail_file)) == "opus")
    for name, raw in (("an empty file", ""), ("one line, no newline", '{"a":1}'),
                      ("a blank line", "\n"), ("trailing newlines", "a\nb\n")):
        p = tmp / "edge.jsonl"
        p.write_text(raw)
        check(f"tail: {name} reads the same as splitlines() did",
              claude_provider._tail(str(p), 200) == raw.splitlines(),
              claude_provider._tail(str(p), 200))
    check("tail: a file that is not there is None, not an empty tail",
          claude_provider._tail(str(tmp / "absent.jsonl"), 10) is None)

    # The bound itself. Without this the function can return every line it
    # happened to read and still look right, since the callers only scan
    # backwards for the first match.
    many = tmp / "many.jsonl"
    many.write_text("".join(f"line {i}\n" for i in range(1000)))
    check("tail: it returns n lines, not every line it had to read",
          claude_provider._tail(str(many), 7) ==
          [f"line {i}" for i in range(993, 1000)],
          claude_provider._tail(str(many), 7)[:3])

    # Crossing the 64 KB seek boundary, which none of the small fixtures do.
    # Two separate failures hide here: the partial first line of a block being
    # kept, and the loop stopping one line early.
    big = tmp / "big.jsonl"
    rows = [f"{i:06d} " + "p" * 900 for i in range(400)]
    big.write_text("\n".join(rows) + "\n")
    for n in (1, 2, 50, 199, 200):
        check(f"tail: {n} lines across the 64 KB boundary match splitlines()",
              claude_provider._tail(str(big), n) == rows[-n:],
              f"{len(claude_provider._tail(str(big), n))} lines, "
              f"first {claude_provider._tail(str(big), n)[0][:12]!r}")

    # A line longer than one seek step, so the loop must run several passes
    # before it has even one newline.
    huge = tmp / "huge.jsonl"
    huge.write_text("short\n" + "z" * 200_000 + "\ntail\n")
    check("tail: a single line larger than the seek step is read whole",
          claude_provider._tail(str(huge), 2) == ["z" * 200_000, "tail"],
          [len(x) for x in claude_provider._tail(str(huge), 2)])

    # CRLF: the PR claims this was found and fixed while writing _tail, and it
    # shipped with nothing holding it.
    crlf = tmp / "crlf.jsonl"
    crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    check("tail: a CRLF file does not keep the carriage returns",
          claude_provider._tail(str(crlf), 2) == ["beta", "gamma"],
          claude_provider._tail(str(crlf), 2))

    # `lines[-0:]` is the whole file, which is not "the last none of them".
    check("tail: asking for no lines gets none, not all of them",
          claude_provider._tail(str(many), 0) == [],
          claude_provider._tail(str(many), 0))

    # The docstring's actual claim -- that it seeks instead of reading the file
    # -- held by nothing until now. Every case above passes just as well for
    # `read_text().splitlines()[-n:]`, which is the implementation this
    # replaced and the one whose cost was the reason to replace it.
    wide = tmp / "wide.jsonl"
    wide.write_text("".join(f"{i:06d} " + "w" * 2000 + "\n" for i in range(2000)))
    read_bytes = []
    real_open = open

    def counting_open(path, *a, **k):
        f = real_open(path, *a, **k)
        inner = f.read

        def read(*args):
            got = inner(*args)
            read_bytes.append(len(got))
            return got
        f.read = read
        return f

    # setattr rather than `patched`: the module has no `open` of its own, it
    # falls through to builtins, and a module-level name shadows that.
    claude_provider.open = counting_open
    try:
        claude_provider._tail(str(wide), 5)
    finally:
        del claude_provider.open
    check("tail: it reads a bounded tail, not the whole 4 MB file",
          sum(read_bytes) < 200_000 and sum(read_bytes) > 0,
          f"{sum(read_bytes)} bytes read of {wide.stat().st_size}")

    # --- detail(): no criterion beside a score it did not produce ------------
    with patched(policy, "system_one", lambda *a, **k: copy.deepcopy(REAL)):
        scored = route.route_one({"id": "shown", "task": TASK}, "opus", False)
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        route.detail(scored, "opus")
    check("detail: a record scored under these questions keeps its criteria",
          "~ " in shown.getvalue() and "different question set"
          not in shown.getvalue(), shown.getvalue()[-120:])
    # The scale the score is on, not just the score. Both question sets have
    # five levels, so the denominator is the one part of the line that says
    # which of them this 3.1 was measured against.
    top = len(policy.QUESTIONS["depth"]["criteria"]) - 1
    check("detail: a kept score prints the scale it was measured on",
          f"/{top}" in shown.getvalue(), shown.getvalue()[-120:])
    stale = copy.deepcopy(scored)
    stale["jev"]["questions"] = "0" * 12
    hidden = io.StringIO()
    with contextlib.redirect_stdout(hidden):
        route.detail(stale, "opus")
    check("detail: a record scored under others shows no criteria at all",
          "different question set" in hidden.getvalue() and
          "~ " not in hidden.getvalue(), hidden.getvalue()[-120:])
    # And no denominator either: /4 under a different five-level set would be
    # the same wrong caption one digit quieter.
    check("detail: nor the scale, which is also the current questions'",
          f"/{top}" not in hidden.getvalue(), hidden.getvalue()[-120:])
    # A record with no hash at all is the third state: it never said which
    # questions produced its scores, which is not a claim that they differed.
    old = copy.deepcopy(scored)
    old.pop("jev")
    schema1 = io.StringIO()
    with contextlib.redirect_stdout(schema1):
        route.detail(old, "opus")
    check("detail: a schema-1 record says it does not know, not that it differs",
          "does not say which questions" in schema1.getvalue() and
          "different question set" not in schema1.getvalue(),
          schema1.getvalue()[-120:])
    # Jev returns a weighted score, so nothing bounds it to the criteria list.
    # The guard is what keeps an out-of-range answer a missing gloss rather
    # than an IndexError out of the one command that explains a decision.
    for label, v in (("above the top level", 9.0), ("below the first", -3.0)):
        wild = copy.deepcopy(scored)
        wild["signals"]["depth"] = v
        out_of_range = io.StringIO()
        try:
            with contextlib.redirect_stdout(out_of_range):
                route.detail(wild, "opus")
            raised = ""
        except Exception as e:  # noqa: BLE001 -- the case is that it does not
            raised = f"{type(e).__name__}: {e}"
        check(f"detail: a score {label} prints without a criterion, not a crash",
              not raised and f"{v:>5.2f}" in out_of_range.getvalue(),
              raised or out_of_range.getvalue()[-90:])

    # --- replay.py: --all and --session are not quietly combined -------------
    clash = subprocess.run(
        [sys.executable, str(HERE / "replay.py"), "--all", "--session", "x"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "JEV_DECISIONS_DIR": str(decisions.LOG_DIR)})
    check("replay: --all with --session is refused, not silently widened",
          clash.returncode != 0 and "drop --session" in
          (clash.stderr + clash.stdout),
          f"{clash.returncode} {(clash.stderr or clash.stdout)[-90:]}")

    # --- install.py: the mod and the hook are not both left running ----------
    import install  # noqa: PLC0415

    reg = tmp / "installed_plugins.json"
    with patched(install, "INSTALLED_PLUGINS", reg):
        reg.write_text(json.dumps({"version": 2, "plugins": {}}))
        check("install: no plugins on record means no warning",
              install.mod_installed() is False)
        reg.write_text(json.dumps({"version": 2, "plugins": {
            "jev-router@somewhere-else": [{"scope": "user"}]}}))
        check("install: the mod is spotted under any marketplace",
              install.mod_installed() is True)
        reg.write_text(json.dumps({"version": 2, "plugins": {
            "renamed@mkt": [{"installPath": str(install.HERE)}]}}))
        check("install: and under a rename, by the path it was installed from",
              install.mod_installed() is True)
        reg.write_text("{ not json")
        check("install: an unreadable registry warns about nothing",
              install.mod_installed() is False)
        # A warning printed beside an install must not be able to stop one, and
        # `.items()` on a list raised straight out of claude_main before the
        # hook was written. `{"plugins": []}` is what a v1 registry looks like.
        reg.write_text(json.dumps({"version": 1, "plugins": []}))
        check("install: a registry shaped the old way warns, it does not crash",
              install.mod_installed() is False)

    # --- hook_agent.py: the Claude Code surface, run as Claude Code runs it --
    #
    # Driven as a subprocess for the same reason selftest_codex.py drives its
    # hook that way: the hook is a process with stdin and stdout, and calling
    # main() in-process tests a function that no host ever invokes. Until this
    # block existed the whole file was unexercised -- including the one line it
    # is allowed to say, which a mutation could blank without failing anything.
    agent_driver = tmp / "drive_agent.py"
    agent_driver.write_text('''
import json, os, runpy, sys
sys.path.insert(0, os.environ["JEV_REPO"])
import policy
plan = json.loads(os.environ["JEV_FAKE_PLAN"])
def fake_route(state, default=None, cache=True):
    seen = os.environ.get("JEV_SEEN_STATE")
    if seen:
        open(seen, "w").write(state)
    if os.environ.get("JEV_FAKE_FAIL"):
        raise SystemExit(os.environ["JEV_FAKE_FAIL"])
    return plan
policy.route = fake_route
runpy.run_path(os.environ["JEV_HOOK"], run_name="__main__")
''')

    transcript = tmp / "session.jsonl"
    transcript.write_text(json.dumps(
        {"type": "assistant", "message": {"model": "claude-sonnet-5"}}) + "\n")
    off_ladder = tmp / "fable.jsonl"
    off_ladder.write_text(json.dumps(
        {"type": "assistant", "message": {"model": "claude-fable-5-1"}}) + "\n")

    AGENT_PLAN = {"model": "opus", "effort": "high", "need": 0.71, "conf": 0.84,
                  "suggested": "opus", "held": False, "wanted": "opus",
                  "tiers": ["haiku", "sonnet", "opus"],
                  "signals": {"depth": 3.1, "breadth": 2.0, "mechanical": 0.1,
                              "irreversible": 0.9, "ambiguous": 0.2},
                  "advice": {"confirm": False, "clarify": False}}

    def run_agent(ti, plan=None, path=None, fail=None, seen=None):
        ev = {"hook_event_name": "PreToolUse", "tool_name": "Agent",
              "session_id": "sess-agent", "tool_use_id": "call-9",
              "transcript_path": str(path or transcript), "tool_input": ti}
        r = subprocess.run(
            [sys.executable, str(agent_driver)], input=json.dumps(ev),
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "JEV_REPO": str(HERE),
                 "JEV_HOOK": str(HERE / "hook_agent.py"),
                 "JEV_DECISIONS_DIR": str(decisions.LOG_DIR),
                 "JEV_FAKE_PLAN": json.dumps(plan or AGENT_PLAN),
                 **({"JEV_FAKE_FAIL": fail} if fail else {}),
                 **({"JEV_SEEN_STATE": str(seen)} if seen else {})})
        return r.returncode, r.stdout.strip(), r.stderr

    rc, out, err = run_agent({"subagent_type": "general-purpose",
                              "description": "design the rollover",
                              "prompt": "weigh the trade-offs"})
    sent = json.loads(out)["hookSpecificOutput"] if out else {}
    check("hook_agent: an upgrade comes back as updatedInput",
          rc == 0 and sent.get("updatedInput", {}).get("model") == "opus",
          f"{rc} {out[:90]}{err[:90]}")
    # updatedInput REPLACES the arguments, so dropping a key drops it from the
    # dispatch -- the prompt included.
    check("hook_agent: and carries the rest of tool_input with it",
          sent.get("updatedInput", {}).get("prompt") == "weigh the trade-offs",
          sent.get("updatedInput"))
    # Exact, not a substring: this is the only sentence the hook is allowed to
    # say, and policy.decision_reason is reachable from here through nothing
    # else. A mutation that emptied it left every other suite green.
    check("hook_agent: the reason is the shared decision_reason line, in full",
          sent.get("permissionDecisionReason")
          == policy.decision_reason("sonnet", AGENT_PLAN),
          sent.get("permissionDecisionReason"))

    rc, out, _ = run_agent(
        {"subagent_type": "general-purpose", "prompt": "x"},
        plan={**AGENT_PLAN, "advice": {"confirm": True, "clarify": False}})
    check("hook_agent: an irreversible task carries the warning on that line",
          "not a gate" in json.loads(out)["hookSpecificOutput"]
          ["permissionDecisionReason"], out[:120])

    for name, kw in (
            ("a fork, which ignores `model` anyway",
             {"ti": {"subagent_type": "fork", "prompt": "x"}}),
            ("a model the caller chose",
             {"ti": {"subagent_type": "general-purpose", "model": "haiku",
                     "prompt": "x"}}),
            ("a session on an off-ladder model",
             {"ti": {"subagent_type": "general-purpose", "prompt": "x"},
              "path": off_ladder}),
            ("an unreadable transcript",
             {"ti": {"subagent_type": "general-purpose", "prompt": "x"},
              "path": tmp / "nope.jsonl"}),
            ("a decision that changed nothing",
             {"ti": {"subagent_type": "general-purpose", "prompt": "x"},
              "plan": {**AGENT_PLAN, "model": "sonnet"}}),
            ("jev failing outright",
             {"ti": {"subagent_type": "general-purpose", "prompt": "x"},
              "fail": "jev: HTTP 401"})):
        rc, out, err = run_agent(**kw)
        check(f"hook_agent: {name} leaves the dispatch alone",
              rc == 0 and out == "", f"{rc} {out[:60]}{err[:60]}")

    # The trim belongs to policy.route, which is the only place that knows
    # what the API costs and what the cache key is. Each front end used to
    # spell the number itself, so a cut here would have been invisible to
    # every test: the hook hands over what it was given.
    seen_state = tmp / "state-seen.txt"
    long_prompt = "x" * 70_000
    run_agent({"subagent_type": "general-purpose", "prompt": long_prompt},
              seen=seen_state)
    check("hook_agent: a long task reaches policy.route uncut",
          long_prompt in seen_state.read_text(),
          len(seen_state.read_text()))

    # The failure still reaches the log, which is the half of "did it even
    # fire" that nothing else answers.
    logged = decisions.read("sess-agent")
    check("hook_agent: a jev failure is recorded, not silently swallowed",
          any("401" in (d.get("why") or "") for d in logged),
          [d.get("why") for d in logged])
    check("hook_agent: and so is the hold that changed nothing",
          any(d["model"] == d["default"] and not d.get("why") for d in logged),
          [(d["default"], d["model"]) for d in logged])


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    policy.CACHE_DIR = tmp / "cache"
    decisions.LOG_DIR = tmp / "decisions"
    cases: list = []
    try:
        _run(tmp, cases)
    except BaseException as e:  # noqa: BLE001 -- a crash is a failing case
        # The cases are straight-line code in one function, so anything that
        # raises outside a check() ends the run where it stands. Reported as a
        # failure rather than as a traceback: a bare traceback suppresses both
        # the summary line and every case already decided, so a typo in the
        # last block printed nothing at all and read like a suite that had not
        # been written yet.
        where = traceback.extract_tb(e.__traceback__)[-1]
        cases.append((f"the suite stopped at {where.filename.rsplit('/', 1)[-1]}"
                      f":{where.lineno}", False, f"{type(e).__name__}: {e}"))

    width = max(len(n) for n, _, _ in cases)
    failed = 0
    for name, ok, detail in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
