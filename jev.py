#!/usr/bin/env python3
"""Call TypeSafe's Jev (System One) API. Zero dependencies, stdlib only.

  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY

The key is read from $TYPESAFE_API_KEY, or from ~/.typesafe/key if that is unset,
so it never has to appear on a command line or in a transcript.

Usage:
  jev.py --state-file task.txt --questions-file qs.json
  echo "some text" | jev.py --questions-file qs.json
  jev.py --state "ship it" --noul urgent="The request is time-sensitive."

Questions file is the `questions` map from the API docs, e.g.
  {"dept": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": null}},
   "heat": {"type": "score",  "instructions": "...", "criteria": ["low", "mid", "high"]},
   "urgent": {"type": "noul", "instructions": "..."}}
"""

from __future__ import annotations

import argparse
import datetime
import email.utils
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
KEY_FILE = pathlib.Path.home() / ".typesafe" / "key"

# The docs cap a request at 64k tokens total, and 32k for `state` plus the
# longest single question. Four chars per token is the usual rough conversion;
# this truncates well short of the cap rather than letting the API 422.
#
# A different number from policy.STATE_LIMIT and a different question. This one
# is the API's ceiling and belongs to every caller of system_one, including the
# CLI below; policy's is how much of a task is worth paying to score, and is
# the smaller of the two, so on a routing path this never bites. It used to be
# applied in main() alone, which left the ceiling unenforced on every path that
# was not someone typing at a shell.
STATE_CHAR_LIMIT = 100_000

RETRYABLE = {429, 529}

# Pinned rather than "jev-latest". A threshold is tuned against one version of
# the scorer, and "latest" moves under it without saying so: the policy here
# was tuned in October 2026, the JEV-as-a-Judge study (arXiv 2609.26550) ran on
# 1.13.0, and by 5 November the API was answering as 1.14.0. The study measures
# nothing across versions -- what it does report is that a threshold does not
# transfer between workloads, which is the same argument one step out: a number
# fitted against one scorer is evidence about that scorer. So a version change
# is a decision someone makes, on purpose, with JEV_MODEL -- and if the pinned
# version is ever retired, the router stops routing and says why in the
# decision log, rather than carrying on with numbers that mean something else.
DEFAULT_MODEL = "jev-1.14.0"

# The whole call, retries included, has to finish inside the caller's budget.
# install.py registers both hooks at timeout 8 and hooks/register.js caps
# route.py at 6s; past that the host kills the process mid-flight, so
# decisions.record() is never reached and the outage is never logged -- the
# one failure this module exists to make visible. An attempt count alone does
# not bound that: five attempts at a 30s timeout plus 15s of backoff is 165s
# worst case. So the ladder is bounded by wall clock and gives up out loud.
BUDGET = 5.0


def model_name() -> str:
    return os.environ.get("JEV_MODEL", "").strip() or DEFAULT_MODEL


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.is_file():
        return KEY_FILE.read_text().strip()
    sys.exit(
        f"no API key: set TYPESAFE_API_KEY or write it to {KEY_FILE} (chmod 600)"
    )


def retry_after(header: str | None, fallback: float) -> float:
    """Seconds to wait, from either form of Retry-After RFC 9110 allows.

    The date form used to reach `float()` and raise ValueError from inside the
    HTTPError handler, where the sibling `except ValueError` could not catch
    it: an exception raised in one handler is not offered to another on the
    same try. It failed open, but logged as a jev-router bug rather than as
    the API asking us to back off.
    """
    if not header:
        return fallback
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return fallback
    # A date with no zone is UTC per RFC 9110; without this the subtraction
    # below raises on the naive/aware mix.
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc))
               .total_seconds())


def system_one(
    state: str, questions: dict, model: str | None = None, timeout: float = 30.0,
    budget: float = BUDGET,
) -> dict:
    """One POST to /v1/systemone, with backoff on the two retryable statuses.

    Every failure leaves as SystemExit carrying a message, so a caller has one
    thing to catch. That used to be true only of HTTP errors: a refused
    connection, a timeout or a body that was not JSON escaped as their own
    exceptions, and route.py's fan-out lost every shard to one of them.

    `budget` bounds the whole call, retries and sleeps included. Giving up
    inside it is what lets the caller log the outage; being killed by the host
    for overrunning is what used to make a rate-limited Jev invisible.

    `state` is cut to STATE_CHAR_LIMIT here, where the request is built, so
    that no caller can reach the API over the documented cap.
    """
    model = model or model_name()
    state = state[:STATE_CHAR_LIMIT]
    body = json.dumps({"state": state, "model": model, "questions": questions})
    req = urllib.request.Request(
        ENDPOINT,
        data=body.encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    deadline = time.monotonic() + budget
    delay = 1.0
    for attempt in range(5):
        left = deadline - time.monotonic()
        if left <= 0:
            raise SystemExit(f"jev: gave up after {budget:g}s")
        try:
            # Never wait longer than the budget has left: a single hung
            # connection at the default 30s timeout outlives every caller.
            with urllib.request.urlopen(req, timeout=min(timeout, left)) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code not in RETRYABLE or attempt == 4:
                # 401 missing/invalid key, 422 names the offending field.
                raise SystemExit(f"jev: HTTP {e.code}: {detail}") from e
            # Honour retry-after when the response carries one.
            wait = retry_after(e.headers.get("retry-after"), delay)
            if time.monotonic() + wait >= deadline:
                # Say which it was. "Rate limited and we cannot wait that
                # long" is a capacity problem; "gave up" alone reads as ours.
                raise SystemExit(
                    f"jev: HTTP {e.code}, backing off {wait:g}s would exceed "
                    f"the {budget:g}s budget") from e
            time.sleep(wait)
            delay *= 2
        except (urllib.error.URLError, OSError) as e:
            # Not retried: the hooks run against a few-second budget, and a
            # network that refused once is not about to change its mind.
            raise SystemExit(f"jev: unreachable: {e}") from e
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            raise SystemExit(f"jev: response was not JSON: {e}") from e
    raise SystemExit("jev: unreachable")


def confidence_of(answer: dict) -> float:
    """Confidence for any answer type, on one scale.

    Choice and Score return `confidence` directly. Noul does not: the docs give
    |2p - 1| as the comparable number, since a noul near 0.5 is the model saying
    it is unsure.
    """
    if answer["type"] == "noul":
        return abs(2 * answer["noul"] - 1)
    return answer["confidence"]


def value_of(answer: dict):
    return answer[answer["type"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--state", help="state as a literal string")
    src.add_argument("--state-file", help="read state from a file")
    p.add_argument("--questions-file", help="JSON file holding the questions map")
    p.add_argument("--noul", action="append", default=[], metavar="KEY=INSTRUCTIONS",
                   help="add a noul question inline; repeatable")
    p.add_argument("--model", default=None,
                   help=f"defaults to $JEV_MODEL, else {DEFAULT_MODEL}")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--summary", action="store_true",
                   help="print one line per answer instead of the raw JSON")
    args = p.parse_args()

    if args.state is not None:
        state = args.state
    elif args.state_file:
        state = pathlib.Path(args.state_file).read_text()
    else:
        state = sys.stdin.read()
    # No truncation here: system_one applies STATE_CHAR_LIMIT itself now, for
    # every caller rather than for this one.

    questions: dict = {}
    if args.questions_file:
        questions.update(json.loads(pathlib.Path(args.questions_file).read_text()))
    for spec in args.noul:
        key, _, instructions = spec.partition("=")
        questions[key] = {"type": "noul", "instructions": instructions}
    if not questions:
        sys.exit("no questions: pass --questions-file or at least one --noul")

    resp = system_one(state, questions, model=args.model, timeout=args.timeout)

    if args.summary:
        for key, ans in resp["answers"].items():
            print(f"{key:24} {value_of(ans)!s:>10}   conf {confidence_of(ans):.2f}")
        u = resp["usage"]
        print(f"\n{resp['model']}  in={u['input_tokens']} out={u['output_tokens']}"
              f"  ${u['input_tokens'] * 0.042 / 1e6:.6f}")
    else:
        json.dump(resp, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
