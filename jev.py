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
STATE_CHAR_LIMIT = 100_000

RETRYABLE = {429, 529}


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.is_file():
        return KEY_FILE.read_text().strip()
    sys.exit(
        f"no API key: set TYPESAFE_API_KEY or write it to {KEY_FILE} (chmod 600)"
    )


def system_one(
    state, questions: dict, model: str = "jev-latest", timeout: float = 30.0
) -> dict:
    """One POST to /v1/systemone, with backoff on the two retryable statuses."""
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

    delay = 1.0
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code not in RETRYABLE or attempt == 4:
                # 401 missing/invalid key, 422 names the offending field.
                raise SystemExit(f"jev: HTTP {e.code}: {detail}") from e
            # Honour retry-after when the response carries one.
            wait = float(e.headers.get("retry-after") or delay)
            time.sleep(wait)
            delay *= 2
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
    p.add_argument("--model", default="jev-latest")
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
    if len(state) > STATE_CHAR_LIMIT:
        state = state[:STATE_CHAR_LIMIT]

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
