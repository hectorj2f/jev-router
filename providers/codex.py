"""OpenAI Codex: three rungs, named by their slugs.

Codex has no tier indirection -- `model`, `default_subagent_model` and `-m` all
take a literal slug (codex-cli 0.162.1), so the tier-to-model mapping that
Claude Code owns is owned here instead. The spend between rungs is what makes
them rungs: $0.10/$0.50 per 1M in-out for luna, $2/$10 for sol, $10/$50 for
astra, so a shard misrouted from luna to astra costs a hundred times the input.
"""

from __future__ import annotations

import json
import pathlib

NAME = "codex"

# codex-cli caches the catalogue it was served under `identity`, a hash of the
# account. So the question probe.py has to spend a request on for Claude --
# will this endpoint serve this rung -- is already answered on disk here, for
# free and without a ping. See served().
CATALOGUE = pathlib.Path.home() / ".codex" / "models_cache.json"

# gpt-6.1-terra is deliberately NOT in the ladder, for the reason fable is not
# in Claude's: it is a specialist (vision and computer-use) rather than a rung,
# and it has no documented reasoning-effort ladder of its own, so routing to it
# would be a change of kind rather than of spend. At $1/$5 it would otherwise
# sit between luna and sol and look like one.
TIERS = ["luna", "sol", "astra"]

_SLUGS = {"luna": "gpt-6-luna", "sol": "gpt-6.1-sol", "astra": "gpt-6-astra"}
_TIER_OF = {slug: tier for tier, slug in _SLUGS.items()}

# luna does take an effort, unlike Claude's haiku: its accepted set is
# low|medium|high|xhigh|max, measured from the catalogue codex-cli 0.162.1
# caches at ~/.codex/models_cache.json. So the cheap rung asks for the cheap
# effort rather than omitting the parameter.
#
# Never emit `ultra`, on any rung. Codex describes it as "maximum reasoning
# with automatic task delegation": the model fans the work out to subagents of
# its own, and every one of those re-enters this router through the dispatch
# hook. That is a recursion the router cannot see, let alone budget for.
# `xhigh` and `max` are left to be asked for by hand -- nothing measured here
# says a score should buy them.
#
# astra is the one rung whose effort is raised above the model's own default,
# which the catalogue gives as `low` where luna and sol both default to
# `medium`. That is deliberate and symmetric with opus asking for `high` on the
# Claude side, but it is a second spend decision stacked on the first: a shard
# routed here is already paying 100x luna's input before the effort is set.
EFFORT = {"luna": "low", "sol": "medium", "astra": "high"}

# An upgrade on this ladder can cost context. sol carries a 500K window and
# astra 272K, so the rung `need` reaches for when breadth is high -- and
# breadth is 40% of the score -- is the one with less room to hold the breadth
# in. Both stretch to 872K on request, and astra's published price doubles
# above 272K, so the compensation is not free either. Nothing here acts on
# that: `need` has no context term, and inventing one unmeasured would be the
# same mistake as the `trivial` noul. It is written down so that whoever does
# measure it starts from the asymmetry rather than discovering it.


def served() -> list[str] | None:
    """Which rungs this account can actually reach, cheapest first.

    None when the catalogue cannot be read at all -- no measurement, which is
    not the same answer as a measurement that came back empty. The empty list
    is meaningful and is returned as itself: policy.available_tiers reads it
    as "measured, serves nothing" and stops routing, which costs a saving
    rather than an agent.
    """
    try:
        models = json.loads(CATALOGUE.read_text())["models"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None

    # `hide` is how the catalogue marks the entries that are not a user's to
    # pick -- gpt-reserve and codex-auto-review both carry it.
    reachable = {m.get("slug") for m in models
                 if m.get("visibility") == "list" and m.get("supported_in_api")}
    return [t for t in TIERS if _SLUGS[t] in reachable]


def model_id(tier: str) -> str:
    return _SLUGS[tier]


def tier_of(model: str) -> str | None:
    """A slug back to its tier; None for anything off the ladder.

    Exact match, not the substring scan the Claude side can afford: these
    slugs carry a family name where Claude's carry the tier name, and the only
    substring they share -- "gpt-6" -- is in all three, so a loose test would
    put the whole family in whichever rung is checked first.
    """
    return _TIER_OF.get(model)


def alias_env(tier: str) -> str | None:
    """None: there is no alias here to go stale.

    Every Codex config key takes a literal slug, so what a probe measured and
    what the router will emit are the same string by construction.
    """
    return None
