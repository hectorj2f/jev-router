#!/usr/bin/env python3
"""Ask the endpoint which tiers it will actually serve, and record the answer.

    ./probe.py            # probe and write cache/tiers.json
    ./probe.py --show     # print what was recorded, probe nothing

Routing to a model the endpoint lacks is strictly worse than not routing: the
dispatch dies on a 404 instead of running on the model it would otherwise have
used, and the hook cannot catch that because it happens after the hook returns.

This existed first as an inference -- "the tier is pinned in the environment,
so it must work" -- which was wrong in the direction that costs you an agent.
A project here pinned ANTHROPIC_DEFAULT_SONNET_MODEL to a model it had never
been granted, so the one tier the inference was most confident about was the
one that 404'd. Hence: measure, don't infer.

The router emits a tier alias and Claude Code resolves it, so the ID that has
to work is exactly ANTHROPIC_DEFAULT_<TIER>_MODEL when set. When it is unset,
Claude Code falls back to a built-in ID that we cannot see from here -- always
valid on the first-party API, not necessarily granted on Vertex or Bedrock --
so an unpinned tier there stays off the ladder rather than being guessed at.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from providers import claude  # noqa: E402

TIERS_FILE = HERE / "cache" / "tiers.json"
# The ladder comes from the provider rather than being spelled again here. It
# was spelled again here, and a second copy of a list this file exists to
# measure is the drift the provider package was added to end.
TIERS = claude.TIERS
PING = {"max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}


def endpoint() -> str:
    """Which API serves Claude here -- a different axis from which model family
    the router is driving. Nothing to probe on Codex: it has no alias to
    resolve, and codex-cli caches the catalogue it was granted itself."""
    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        return "vertex"
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
        return "bedrock"
    return "anthropic"


def pinned(tier: str) -> str | None:
    return os.environ.get(claude.alias_env(tier))


def post(url: str, body: dict, headers: dict,
         model: str = "") -> tuple[bool | None, str]:
    """True served, False not served, None could not be measured.

    `model` is only read to tell the two meanings of a 403 apart; omitting it
    reads every 403 as unmeasured, which is the safe direction.

    The third state is the point. It used to be folded into False, under a
    comment that identified the distinction exactly -- "unknown, not absent" --
    and then discarded it, so main() wrote the tier down as GONE. One DNS blip
    or one timeout permanently narrowed the recorded ladder, and nothing in
    cache/tiers.json said it had happened.
    """
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True, "ok"
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:120].replace("\n", " ")
        # A 400 means the request reached a model that exists and disliked the
        # body -- which is all we are asking. Only "not found" rules a tier out.
        if e.code == 404:
            return False, f"HTTP 404 {detail}"
        # A 403 is two different facts wearing one code. Vertex returns
        # PERMISSION_DENIED both for "this project was never granted that model"
        # -- a real answer -- and for "the caller lacks aiplatform.user" or "the
        # API is off", which are about the machine. Only the first names the
        # model, so only the first is read as an answer.
        if e.code == 403:
            return (False if model and model in detail else None,
                    f"HTTP 403 {detail}")
        # Everything else is unmeasured, not unserved. This branch used to be
        # `e.code not in (403, 404)`, so an expired key (401), a proxy that
        # wanted its own auth (407), a rate limit (429) or a bad gateway all
        # recorded a fully-confirmed ladder nothing had confirmed -- while
        # probe_anthropic right below returned None for an API key that was
        # merely absent. The same fact about this machine, answered two ways.
        if e.code in (401, 407, 429) or e.code >= 500:
            return None, f"HTTP {e.code} {detail}"
        return True, f"HTTP {e.code} {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def probe_vertex(model: str) -> tuple[bool | None, str]:
    # None, not False, on all three of these: a missing project, a gcloud that
    # would not run and a missing token are facts about this machine, not about
    # what Vertex serves.
    project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        return None, "ANTHROPIC_VERTEX_PROJECT_ID unset"
    region = os.environ.get("CLOUD_ML_REGION", "global")
    host = ("aiplatform.googleapis.com" if region == "global"
            else f"{region}-aiplatform.googleapis.com")
    try:
        token = subprocess.run(["gcloud", "auth", "print-access-token"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return None, f"gcloud: {e}"
    if not token:
        return None, "no gcloud access token (gcloud auth login)"
    url = (f"https://{host}/v1/projects/{project}/locations/{region}"
           f"/publishers/anthropic/models/{model}:rawPredict")
    return post(url, {"anthropic_version": "vertex-2023-10-16", **PING},
                {"Authorization": f"Bearer {token}"}, model)


def probe_anthropic(model: str) -> tuple[bool | None, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "ANTHROPIC_API_KEY unset"  # unmeasurable, not unserved
    return post("https://api.anthropic.com/v1/messages", {"model": model, **PING},
                {"x-api-key": key, "anthropic-version": "2023-06-01"}, model)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="print the record, probe nothing")
    a = p.parse_args()

    if a.show:
        if not TIERS_FILE.is_file():
            print(f"nothing recorded at {TIERS_FILE}; run ./probe.py")
            return 1
        print(TIERS_FILE.read_text())
        return 0

    prov = endpoint()
    print(f"endpoint     {prov}")
    if prov == "bedrock":
        print("\nBedrock probing isn't implemented -- model access there is per-account\n"
              "and not readable from a ping. Declare the ladder yourself:\n"
              "    export JEV_TIERS=sonnet,opus")
        return 1

    probe = {"vertex": probe_vertex, "anthropic": probe_anthropic}[prov]
    good, unknown, resolved = [], [], {}

    for tier in TIERS:
        model = pinned(tier)
        if not model:
            if prov == "anthropic":
                good.append(tier)
                resolved[tier] = "(built-in default)"
                print(f"  {tier:<8} assumed -- unpinned, and the first-party fallback is valid")
            else:
                resolved[tier] = None
                print(f"  {tier:<8} SKIP  unpinned; set ANTHROPIC_DEFAULT_{tier.upper()}_MODEL "
                      f"to use this tier")
            continue
        ok, why = probe(model)
        resolved[tier] = model
        if ok:
            good.append(tier)
        elif ok is None:
            unknown.append(tier)
        verdict = "OK  " if ok else "????" if ok is None else "GONE"
        print(f"  {tier:<8} {verdict}  {model}{'' if ok else f'  -- {why}'}")

    if unknown and TIERS_FILE.is_file():
        # The record this writes is read back as a ladder, and a tier absent
        # from it is routed around. A tier we could not reach is not a tier
        # that is not there, so the honest thing is to write nothing and keep
        # whatever the last successful probe said -- the same stance the
        # zero-confirmed case below takes, for the same reason.
        print(f"\n{', '.join(unknown)} could not be reached, which is not the same "
              f"as not being\nserved -- not recording, leaving the previous "
              f"measurement in place.\nRe-run when whatever that was has cleared.")
        return 1

    if unknown and good:
        # Nothing on file, so there is no previous measurement to leave in
        # place -- and available_tiers() reads "no record at all" as the full
        # ladder, which is wider than what was just measured and is the one
        # direction this file exists to avoid. Recording the confirmed tiers is
        # narrower than the fallback, and narrow costs a saving where wide
        # costs an agent. Discarding them to return 1, as this used to, handed
        # a first-time user on a flaky network the whole ladder instead.
        print(f"\n{', '.join(unknown)} could not be reached, which is not the same as "
              f"not being\nserved -- but there is no previous measurement to keep, and "
              f"no record at\nall reads as the full ladder. Recording what did confirm; "
              f"re-run when\nwhatever that was has cleared.")

    if not good:
        # Finding nothing is a failed measurement, not a measurement of
        # nothing, and writing it would overwrite a good record with a blank.
        # On Vertex with not one alias pinned, the overwhelmingly likely cause
        # is the environment: Claude Code resolves these into the processes it
        # spawns, so a probe that cannot see any of them is running outside
        # that -- a plain terminal, or Claude Code's `!` prefix.
        print("\nno tier could be confirmed -- not recording, leaving any previous\n"
              "measurement in place.")
        if prov == "vertex" and not any(resolved.values()):
            print("\nNothing is pinned at all, which usually means this probe cannot see\n"
                  "Claude Code's resolved model config. Run it from a Claude Code tool\n"
                  "context rather than a bare shell or the `!` prefix -- or set the\n"
                  "ANTHROPIC_DEFAULT_*_MODEL variables where this process can read them.")
        return 1

    TIERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIERS_FILE.write_text(json.dumps(
        {"tiers": good, "resolved": resolved, "provider": prov,
         "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")

    print(f"\nladder       {', '.join(good)}")
    print(f"recorded     {TIERS_FILE}")
    if len(good) < 2:
        print("\nFewer than two tiers means there is nothing to route between -- the hook\n"
              "will leave every dispatch alone. Grant or pin another model, or set\n"
              "JEV_TIERS if you know better than this probe.")
    # Non-zero when a tier went unanswered, even though the confirmed ones were
    # just written. The branch above returns 1 for the same incompleteness when
    # there is a record to protect, and a probe that says "re-run when that has
    # cleared" while exiting 0 is telling a script the opposite of what it just
    # told the reader.
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
