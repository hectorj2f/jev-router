#!/usr/bin/env python3
"""Re-route this session's real prompts under the old and new policies.

The old formula is reproduced here rather than imported, so the two run off one
set of Jev answers and any difference is the policy rather than a resample --
Jev is non-deterministic enough that two calls on identical state drift.
"""

from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import policy  # noqa: E402

TIERS = policy.TIERS


def old_decide(ans: dict, default: str = "sonnet") -> str:
    depth = ans["depth"]["score"] / 4
    breadth = ans["breadth"]["score"] / 4
    conf = min(ans["depth"]["confidence"], ans["breadth"]["confidence"])
    irr = ans["irreversible"]["noul"]

    need = 0.45 * depth + 0.30 * breadth + 0.25 * ans["ambiguous"]["noul"]
    need -= 0.25 * ans["mechanical"]["noul"]
    if irr > 0.6:
        need = max(need, 0.60)

    sug = "opus" if need >= 0.60 else "sonnet" if need >= 0.25 else "haiku"
    if TIERS.index(sug) > TIERS.index(default):
        return sug
    if TIERS.index(sug) < TIERS.index(default):
        clear = ans["mechanical"]["noul"] > 0.75 or conf >= 0.70
        return sug if clear and irr < 0.30 else default
    return default


def user_prompts(transcript: str) -> list[str]:
    out, seen = [], set()
    for line in pathlib.Path(transcript).read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain"):
            continue
        c = e.get("message", {}).get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        if c.lstrip().startswith(("<command-name>", "<local-command-",
                                  "<command-message>", "<system-reminder>")):
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> None:
    prompts = user_prompts(sys.argv[1])
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(policy.ask, prompts))

    hdr = f"{'prompt':<42} {'old':<7} {'new':<7} {'need':>5} {'mech':>5}  advice"
    print(hdr)
    print("-" * len(hdr))
    tally = {"old": {}, "new": {}}
    for p, ans in zip(prompts, answers):
        new = policy.decide(ans)
        old = old_decide(ans)
        tally["old"][old] = tally["old"].get(old, 0) + 1
        tally["new"][new["model"]] = tally["new"].get(new["model"], 0) + 1
        advice = " ".join(k for k, v in new["advice"].items() if v) or "-"
        flag = "" if old == new["model"] else "  <--"
        oneline = " ".join(p.split())[:40]
        print(f"{oneline:<42} {old:<7} {new['model']:<7} {new['need']:>5.2f} "
              f"{new['signals']['mechanical']:>5.2f}  {advice}{flag}")

    for which in ("old", "new"):
        split = ", ".join(f"{tally[which].get(t, 0)}x{t}" for t in TIERS)
        print(f"{which:>4}: {split}")


if __name__ == "__main__":
    main()
