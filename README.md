# jev-router

Pick the Anthropic model for a Claude Code subagent based on what the task
actually needs, using [TypeSafe's Jev](https://docs.typesafe.ai) to score the
task first.

The interesting part of this repo is not the router. It is the measurements,
including the ones that say the idea doesn't work. Those are in
[Findings](#findings), and they are the reason this routes *subagent dispatches*
and nothing else.

## How it works

Five atomic questions to Jev about the task — `depth` and `breadth` as 5-level
scores, `mechanical`, `irreversible`, `ambiguous` as nouls — combined in code
rather than asking Jev "which model should I use". Which tier a task deserves
depends on your cost tolerance, not on the task, so that half belongs where it
is a coefficient instead of a prompt.

```python
need = 0.60 * depth + 0.40 * breadth - 0.25 * mechanical

advice = {"confirm": irreversible > 0.60,    # a confirmation step, not a bigger model
          "clarify": ambiguous   > 0.70}     # a question back, not a bigger model
```

`need >= 0.60` → opus, `>= 0.25` → sonnet, else haiku.

**Upgrades are ungated; downgrades must be earned.** Opus on a task that only
needed Haiku costs a fraction of a cent. Haiku on a task that needed Opus costs
a wrong answer that looks right. So `need` clearing a bar is enough to upgrade
even when Jev is unsure, while a downgrade additionally needs `mechanical > 0.75`
or score confidence `>= 0.70`.

That asymmetry is measured, not preferred. A confidence floor on the *upgrade*
did the opposite of its job: the two heaviest tasks of ten scored `need` 0.60
and 0.67 at score-confidence 0.00 and 0.25, and the floor pinned exactly those
to the cheaper tier. A 5-level score spreads probability across adjacent levels,
so it reads as low confidence while the weighted value stays sound. The nouls
are two-outcome, so their distributions aren't diffused the same way — which is
why the downgrade gate uses them instead.

## Install

Needs Python 3.10+, no dependencies. Put your TypeSafe key in `~/.typesafe/key`
(`chmod 600`) or `$TYPESAFE_API_KEY`.

```bash
git clone https://github.com/hectorj2f/jev-router ~/jev-router
chmod +x ~/jev-router/*.py
```

Then register the hook:

```bash
~/jev-router/install.py             # --check to see the state, --uninstall to undo
```

It backs up `~/.claude/settings.json`, merges rather than replaces, and
re-running swaps its own entry instead of stacking a second one. If you'd
rather do it by hand, the entry is:

```json
"hooks": {
  "PreToolUse": [{"matcher": "^Agent$",
                  "hooks": [{"type": "command", "command": "/absolute/path/to/hook_agent.py", "timeout": 8}]}]
}
```

Hooks hot-reload — no restart, and it applies to sessions already open. Every
subagent dispatch now gets a model chosen for it. The hook respects an explicit `model` in the dispatch, skips
`subagent_type: "fork"` (a fork always inherits the parent model), and swallows
every exception, so it can never block a dispatch.

### Check the ladder first if you're on Vertex or Bedrock

```bash
python3 -c 'import sys; sys.path.insert(0,"."); import policy; print(policy.available_tiers())'
```

Claude Code resolves each tier alias through `ANTHROPIC_DEFAULT_<TIER>_MODEL`
and falls back to a built-in ID when one is unset. On the first-party API that
fallback always works. On Vertex and Bedrock it names a model your project may
never have been granted — so the router drops any tier you haven't pinned
there, because routing to a model the endpoint lacks is strictly worse than
not routing: the dispatch dies on a 404 instead of running on the model it
would otherwise have used.

If that guess is wrong in either direction, say so explicitly:

```bash
export JEV_TIERS=sonnet,opus      # or haiku,sonnet,opus
```

### Try it without installing anything

Route a single task and see the whole decision:

```bash
./route.py --default opus "Apply the staging Terraform stage and confirm the new service came up."
```

```
  model        opus   effort high
               held at opus; haiku was suggested but a downgrade needs mechanical > 0.75 or confidence >= 0.70
  need          0.15  haiku ...........#..|.........|........... opus
  confidence    0.00

  signals
    depth         1.33/4  ~ Little. Look something up and report it.
    breadth       1.27/4  ~ One file or one small package.
    mechanical    0.69
    irreversible  0.93
    ambiguous     0.83

  advice
    writes something hard to undo -- confirm before running
    under-specified -- ask back rather than guess
```

That one output is the whole policy: Jev reads the task as small, the gate
refuses the downgrade anyway because it is unsure (confidence 0.00) and the
task isn't quite mechanical enough (0.69, against a 0.75 bar), and the fact
that it applies Terraform becomes a confirmation step rather than a tier.

Several tasks at once give the table instead:

```bash
./route.py --default opus --table < examples/shards.json
```

```
shard                        model    need  conf  advice
--------------------------------------------------------
gofmt-fixtures               haiku   -0.20  0.85  -
bump-dep                     sonnet   0.35  0.36  -
read-count                   haiku    0.05  0.99  -
rename-field                 sonnet   0.28  0.49  -
find-callers                 sonnet   0.39  0.85  -
alert-docs                   opus     0.38  0.18  clarify, held from sonnet
binding-guard                opus     0.60  0.06  clarify, held from sonnet
force-push                   opus     0.25  0.27  confirm, clarify, held from sonnet
tf-apply                     opus     0.15  0.00  confirm, clarify, held from haiku
design-epoch                 opus     0.86  0.71  clarify

2xhaiku, 3xsonnet, 5xopus
```

Five of ten held at opus, every one of them for weak evidence — `tf-apply` at
confidence 0.00, `binding-guard` at 0.06. That is the asymmetry working, not the
router failing.

Jev is non-deterministic: identical input drifts by a few hundredths of a point
between runs, so your numbers will differ slightly from these. Tier assignments
near a threshold can flip.

Piped without `--table` it emits JSON suitable as `args` to
`routed-fanout.workflow.js`.

## Verifying it works

**1. The hook's logic, without Claude Code.** Drives `hook_agent.py` with
synthetic `PreToolUse` events and asserts both the routing and every fail-open
path. Needs a Jev key; answers are cached, so re-runs are free.

```bash
./selftest.py
```

```
PASS  fork is never rerouted
PASS  explicit model wins
PASS  unreadable transcript
PASS  missing tool_input
PASS  empty object
PASS  malformed stdin
PASS  off-ladder session model declines
PASS  mechanical task leaves opus
PASS  deep task stays on opus
PASS  updatedInput preserves every original key

10/10 passed
```

The two routing cases assert *not opus* and *stays opus* rather than exact
tiers, since Jev drifts between calls and a task near a threshold can flip.

**2. What your subagents actually ran on.** The hook rewriting `model` proves
only that the hook ran — this proves Claude Code honoured it. Subagent
transcripts live at
`~/.claude/projects/<slug>/<session-id>/subagents/agent-*.jsonl` and record the
model on every assistant turn.

```bash
./verify.py --project ~/your/repo --all
```

```
session ddce018e-6787-449c-a080-12fbe80314fc
  Backcompat + axlotl compat review                      opus-5
  Security review of PyPI lane                           opus-5
  CSV data integrity audit                               opus-5
  ...
```

That is a ten-way review fan-out *before* the hook: every agent on opus,
including the ones doing bookkeeping.

**3. The A/B.** Run `verify.py` to capture the baseline, install the hook,
dispatch a mixed batch — something mechanical (*"run gofmt on X and commit"*)
alongside something deep (*"decide whether the guard should reject or coerce,
and implement it"*) — then re-run `verify.py`. The mechanical ones should come
back on a cheaper tier and the deep ones should still say opus.

If nothing changes, check in this order: the hook is registered in
`~/.claude/settings.json`; `claude --debug` shows it firing on `Agent`;
`./selftest.py` still passes; and the dispatch didn't already name a `model`,
which the hook deliberately respects.

## Findings

Measured over ten described tasks and 26 verbatim prompts from a real Claude
Code session, both on a private codebase. `examples/shards.json` is the same ten
tasks reworded to remove internal names — which is itself instructive, since the
rewording moved two of them a full tier. Re-measure on your own tasks with
`shardcmp.py` and `compare.py` rather than trusting these numbers; what should
transfer is the shape, not the thresholds.

**The default must be the session's own tier.** A dispatch inherits the session
model, so hardcoding `"sonnet"` in an opus session makes every sonnet-grade task
read as "no change" and skip the downgrade gate entirely — silently dropping 5
of 10 shards from opus to sonnet, one of them at confidence 0.00. `hook_agent.py`
reads the real tier from the transcript and declines to route if it can't.

**Irreversibility and ambiguity do not belong in the tier.** Both were in the
formula originally and both were pricing a conversational quirk as difficulty:

- `ambiguous` carried 0.25 of `need`, and across 26 real prompts it never read
  below 0.62, median 0.80 — a near-constant +0.20 against a 0.25 haiku line.
  The reading is correct; a terse prompt genuinely *is* under-specified. The
  response was wrong.
- An `irreversible > 0.6` floor forced `need` to the opus line. It sent a
  `depth` 0.4 task — "reformat this comment I already wrote" — straight to opus
  on that basis alone.

Moving both to `advice` cut opus from 8/26 to 4/26 on session prompts with no
task losing capability it needed. A cheap model posting the right comment is
fine; what irreversibility wants is a confirmation step, and what ambiguity
wants is a question back. Buying a bigger model answers neither.

**Asking Jev directly whether a task is haiku-able does not work.** A `trivial`
noul — *"a small, fast model would carry this out correctly first time"* — was
added as a direct cheap-tier gate and removed again: across 36 calls it never
once exceeded 0.70, and all four haiku routings cleared on `mechanical` or score
confidence instead. Jev answers it conservatively enough that any threshold
loose enough to fire stops discriminating. `mechanical` asks a narrower question
and gets a sharper answer.

**This does not work on conversational prompts, and you should not wire it to
your session model.** The same policy over 26 real session prompts routes
*nothing* to haiku, while ten described tasks route four (two, for the reworded
set shipped in `examples/`). Nine of the 26 score below the haiku line — "yes,
do it" scores 0.13 — and all nine are correctly held, because `mechanical` never
exceeds 0.60 on a conversational prompt.

That's not a tuning problem. "yes, do it" scored 0.13 and meant *run a ten-task
routing experiment across two model tiers*. The low score is an artifact of a
four-word prompt, not evidence of a simple task. **The router sees the prompt,
not the work.** Dispatch prompts describe the work; chat prompts point at it.
That gap is the whole reason this hooks `^Agent$` and nothing else.

There is also no mechanism for it even if you wanted one: no hook can set the
session model (`PreModelSwitch` only allows/denies/asks), the `model` settings
key is read once at session start, and the first request after any switch
re-reads the whole conversation uncached. The Agent SDK's `setModel()` is the
only true mid-session switch.

**How you phrase the task is a tier-level input.** On the private set,
`alert-docs` routed to **haiku** and shouldn't have — writing alert
documentation an on-call engineer can act on is not haiku work. It cleared the
gate on confidence 0.79. Rewording that same task for `examples/shards.json` to
say *what the documentation has to achieve* rather than just naming the artifact
moved it to **opus** (`need` 0.38, held from sonnet at confidence 0.18). Same
policy, same work, two tiers apart.

`bump-dep` moved the other way for the same reason: dropping the specific
version and file count took it from haiku to sonnet.

**The cheapest tier may not exist, and failing open doesn't save you.** The
hook swallows every exception so it can never block a dispatch — but it cannot
catch an error that happens *after* it returns. On a Vertex project with no
haiku granted, the router correctly read a one-file lookup as haiku work,
emitted `model: haiku`, and the agent died on
`model_not_found: claude-haiku-4-5@20251001`. The dispatch didn't fall back to
the session model; it just failed.

So the ladder is now built from what the endpoint serves, and a tier that
isn't served is clamped *upward* — the same task routes `opus → sonnet`, which
is a smaller saving and a live agent. The general shape: fail-open is a
property of the hook's own code paths, not of the decision it hands downstream,
and a decision that is cheap but unservable is worse than no decision at all.

**Feeding in session context doesn't poison the read — it just mostly isn't
the missing piece.** The obvious response to "the router sees the prompt, not
the work" is to put the conversation in the state. `experiments/context_dilution.py`
measures both directions of that.

Contamination barely exists. A trivial task — *"report the value of the
maxRetries constant"* — held haiku through 12k chars of unrelated hard
conversation, `need` moving only 0.05 → 0.12 at confidence 0.86–0.99, and it
*saturated*: 4× and 16× filler score identically. Jev locates the task rather
than averaging the state.

Enrichment is real, and narrow. An opaque pointer, *"review PR #4471 and tell
me whether it is safe to merge"*:

| state | depth | breadth | need |
|---|---|---|---|
| bare (56 chars) | 2.7 | **1.02** | 0.49 |
| + resolved PR facts (384 chars) | 2.06 | **3.05** | 0.60 |
| + hard conversation instead (815 chars) | 3.09 | 2.33 | 0.68 |

Breadth tripled once the state said *31 files, +2847 −1193* — exactly the
signal the router had been guessing at. Note the third row: raw conversation
moved `need` too, and further, but via `depth` and to a worse breadth. It is
difficulty by association, not information about the work, so it happens to
point the right way here and won't in general.

So the instrument is a few hundred characters of resolved reference, not bulk
transcript. Cost isn't the argument — 32k tokens of state is $0.0013. The
arguments are that the cache keys on state text, so conversation in the state
means every call is unique and always cold; that shipping a whole conversation
to a third party discloses far more than shipping a task description; and that
on the surface this actually hooks, dispatch prompts are already written to
stand alone, so the parent session's context describes the wrong thing.

This is the central finding again, at a smaller scale — the router reads the
description, so an under-described task gets under-served. If you hit a
false-cheap, the first thing to check is the prompt, and the next lever is
raising the confidence floor from 0.70. Do not reach for another signal; that
was tried and measured as contributing nothing.

## Cost and privacy

~$0.00003 per task routed ($0.042/Mtok input, output free). A 50-shard fan-out
is about a tenth of a cent. Answers are memoized for 7 days in `cache/`, keyed
on the task text *and* the question set — the latter matters, since keying on
the text alone silently serves answers to questions you've since edited.

**Every routed task is POSTed to `api.typesafe.ai`**, carrying task text, file
paths, and whatever else is in the prompt. That is a real disclosure decision,
not a footnote. Think hard before pointing this at anything whose inputs can be
customer-derived.

## Files

| | |
|---|---|
| `policy.py` | the questions, scoring, gates, cache, transcript readers. The only copy of the policy — everything imports it |
| `jev.py` | zero-dependency client for `POST /v1/systemone`; backs off on 429/529, normalises confidence across all three primitives (`noul` returns none, so it computes `\|2p − 1\|`) |
| `hook_agent.py` | the `PreToolUse` hook. The recommended entry point |
| `install.py` | registers/removes the hook in `~/.claude/settings.json`, with a backup |
| `route.py` | maps the policy over a list of tasks, concurrently, failing open to the default on a Jev error |
| `routed-fanout.workflow.js` | Claude Code workflow running one agent per routed shard |
| `selftest.py` | drives the hook with synthetic events; asserts routing and every fail-open path |
| `verify.py` | reads subagent transcripts and reports the model each one actually ran on |
| `compare.py` / `shardcmp.py` | the measurement harnesses behind the numbers above |
| `experiments/context_dilution.py` | whether session context in the state helps the router or buries the task |

## License

Apache 2.0.
