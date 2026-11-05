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
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
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


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    policy.CACHE_DIR = tmp / "cache"
    decisions.LOG_DIR = tmp / "decisions"
    cases = []

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

    width = max(len(n) for n, _, _ in cases)
    failed = 0
    for name, ok, detail in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
