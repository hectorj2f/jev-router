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
TIERS_FILE = HERE / "cache" / "tiers.json"
TIERS = ["haiku", "sonnet", "opus"]
PING = {"max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}


def provider() -> str:
    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        return "vertex"
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
        return "bedrock"
    return "anthropic"


def pinned(tier: str) -> str | None:
    return os.environ.get(f"ANTHROPIC_DEFAULT_{tier.upper()}_MODEL")


def post(url: str, body: dict, headers: dict) -> tuple[bool, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True, "ok"
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:120].replace("\n", " ")
        # A 400 means the request reached a model that exists and disliked the
        # body -- which is all we are asking. Only "not found" rules a tier out.
        return (e.code not in (403, 404), f"HTTP {e.code} {detail}")
    except Exception as e:  # network, DNS, timeout -- unknown, not absent
        return False, f"{type(e).__name__}: {e}"


def probe_vertex(model: str) -> tuple[bool, str]:
    project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        return False, "ANTHROPIC_VERTEX_PROJECT_ID unset"
    region = os.environ.get("CLOUD_ML_REGION", "global")
    host = ("aiplatform.googleapis.com" if region == "global"
            else f"{region}-aiplatform.googleapis.com")
    try:
        token = subprocess.run(["gcloud", "auth", "print-access-token"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return False, f"gcloud: {e}"
    if not token:
        return False, "no gcloud access token (gcloud auth login)"
    url = (f"https://{host}/v1/projects/{project}/locations/{region}"
           f"/publishers/anthropic/models/{model}:rawPredict")
    return post(url, {"anthropic_version": "vertex-2023-10-16", **PING},
                {"Authorization": f"Bearer {token}"})


def probe_anthropic(model: str) -> tuple[bool, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "ANTHROPIC_API_KEY unset"
    return post("https://api.anthropic.com/v1/messages", {"model": model, **PING},
                {"x-api-key": key, "anthropic-version": "2023-06-01"})


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

    prov = provider()
    print(f"provider     {prov}")
    if prov == "bedrock":
        print("\nBedrock probing isn't implemented -- model access there is per-account\n"
              "and not readable from a ping. Declare the ladder yourself:\n"
              "    export JEV_TIERS=sonnet,opus")
        return 1

    probe = {"vertex": probe_vertex, "anthropic": probe_anthropic}[prov]
    good, resolved = [], {}

    for tier in TIERS:
        model = pinned(tier)
        if not model:
            if prov == "anthropic":
                good.append(tier)
                resolved[tier] = "(built-in default)"
                print(f"  {tier:<8} assumed -- unpinned, and the first-party fallback is valid")
            else:
                resolved[tier] = None
                print(f"  {tier:<8} SKIP    unpinned; set ANTHROPIC_DEFAULT_{tier.upper()}_MODEL "
                      f"to use this tier")
            continue
        ok, why = probe(model)
        resolved[tier] = model
        if ok:
            good.append(tier)
        print(f"  {tier:<8} {'OK ' if ok else 'GONE'}    {model}{'' if ok else f'  -- {why}'}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
