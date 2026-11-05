# jev-router

Pick the model for a subagent based on what the task actually needs, using
[TypeSafe's Jev](https://docs.typesafe.ai) to score the task first. Works on
Claude Code and on OpenAI Codex, off one copy of the policy.

The interesting part of this repo is not the router. It is the measurements,
including the ones that say the idea doesn't work. Those are in
[Findings](#findings), and they are the reason this routes *subagent dispatches*
and nothing else.

If you want a Jev router that routes *every turn*, you want
[gargpratyush/jev-router](https://github.com/gargpratyush/jev-router), which is
older, more complete, and not related to this one. See
[Related work](#related-work).

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

`need >= 0.60` → the dear rung, `>= 0.25` → the middle one, else the cheap one.
On Claude that is opus / sonnet / haiku; on Codex, `gpt-6-astra` / `gpt-6.1-sol`
/ `gpt-6-luna`. The two lines are the same numbers either way — they mark a
position on a three-rung ladder, and they were measured on Claude's.

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

Then pick a front end. There are three — two for Claude Code, one for Codex —
they do the same thing, and they read the same `policy.py`. **Run one per
host, not two**, or every dispatch gets priced twice.

### As a mod — Claude Code 2.1.287 or later

```bash
claude plugin marketplace add ~/jev-router
claude plugin install jev-router@jev-router
```

Or `claude --plugin-dir ~/jev-router` for one session, without installing.

### As a PreToolUse hook — any version

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

Hooks hot-reload — no restart, and it applies to sessions already open.

### On OpenAI Codex

```bash
~/jev-router/install.py --codex     # --check to see the state, --uninstall to undo
```

The front end is `hook_codex.py`, a `PreToolUse` hook matched on
`.*spawn_agent$`. Install writes `~/.codex/hooks.json` — a *sibling* of
`config.toml`, not `hooks/hooks.json`, which is the plugin-relative path. It
never writes `config.toml`, because TOML does not round-trip through a rewrite
the way JSON does.

The matcher has to be a regex. Codex compiles a matcher of only word characters
to an exact name match, and the V2 multi-agent backend registers the tool inside
a namespace and hands the hook the namespace and the name run together —
`collaborationspawn_agent`, on all 443 real spawns in the rollouts on the
machine this was written on. A bare `spawn_agent` is therefore the one spelling
that can never fire. Matching the tail keeps the V1 name working and survives a
renamed namespace.

Codex asks you to trust the hook the first time it loads it, and again after
any change to the command. If you already declare `[hooks]` in `config.toml`,
pick one file — Codex loads both and a doubled entry routes every dispatch
twice.

Codex is the better-placed of the three hosts, for one reason: `PreToolUse`
hands over the parent's own resolved model as a field, so there is no
transcript to scrape and `policy.session_tier` has no Codex counterpart at all.

Two precedence rules, both read off the source rather than guessed. A model
this hook supplies **beats** `[agents] default_subagent_model`, which is only
the fallback. It **loses** to an `agent_type` role that pins a model of its
own, because the role is applied after the override.

### Either way

Every subagent dispatch now gets a model chosen for it. All three front ends
respect an explicit `model` in the dispatch, decline to touch a fork, and fail
open, so none of them can block a dispatch.

Failing open matters more on Codex than on Claude: `updatedInput` replaces the
whole argument object rather than merging into it, and `SpawnAgentArgs` is
`deny_unknown_fields`, so an argument object this got wrong would kill the
spawn instead of being ignored. Every error path there exits 0 with no output,
which leaves the arguments exactly as the model wrote them.

On Claude Code the mod is the better of the two where you can run it, because
the facts the hook has to go and find arrive as fields of the event it listens
to, and because it listens to more than the `Agent` tool. Codex's hook is in
the right-hand column's position without needing a mod at all:

| | `hook_agent.py` | the mod | `hook_codex.py` |
|---|---|---|---|
| the session's tier | parsed back out of the transcript file | `e.parentModel`, already resolved | `model`, already resolved |
| is this a fork | inferred from `subagent_type` | `e.fork` | `fork_turns` / `fork_context` |
| did the caller pick a model | read out of the raw tool input | `e.model` | `tool_input.model` |
| agent-team teammates | never seen — they aren't `Agent` tool calls | routed, same as any dispatch | n/a |
| the `confirm` advice | reachable only on a dispatch it already re-tiers | printed at spawn, always | same as `hook_agent.py`; [see below](#the-advice-half-mostly-unbuilt) |

It also carries three commands — `/jev-probe`, `/jev-route` and
`/jev-explain` — which are the next three sections.

### Find out which tiers you can actually serve

```bash
/jev-probe          # from the mod
./probe.py          # same script, see the warning below about where you run it
```

Routing to a model your endpoint lacks is strictly worse than not routing: the
dispatch dies on a 404 rather than running on the model it would otherwise
have used, and the hook can't catch that because it happens after the hook
returns. So measure once and record it:

```
provider     vertex
  haiku    SKIP    unpinned; set ANTHROPIC_DEFAULT_HAIKU_MODEL to use this tier
  sonnet   GONE    claude-sonnet-4@20250514  -- HTTP 404 Publisher model ... not found
  opus     OK      claude-opus-5

ladder       opus
```

Claude Code resolves each tier alias through `ANTHROPIC_DEFAULT_<TIER>_MODEL`,
falling back to a built-in ID when unset — always valid on the first-party API,
not necessarily granted on Vertex or Bedrock. The probe pings whichever ID the
alias really resolves to, so it catches both an unpinned tier and a tier pinned
to something your project doesn't have. The result lands in `cache/tiers.json`
and is re-checked against the environment on every read, so moving a pin
invalidates it rather than silently routing on a stale measurement.

**It has to run where the hook runs.** Claude Code resolves each tier to a
concrete model and injects that into the processes it spawns — the hook, and
its own tool calls. A probe run anywhere else, including Claude Code's `!`
prefix and a plain terminal, sees none of those variables and reports every
tier unpinned, which is a fact about the probe's environment and not about
your endpoint. `/jev-probe` exists for exactly this: the mod's subprocess
inherits Claude Code's own environment, so the probe sees what the router will
see. Without the mod, ask Claude to run `./probe.py` for you, which gets you
the same thing by a longer road.

If every tier comes back unpinned on Vertex, that's the symptom: it recorded
nothing and told you so rather than overwriting a good measurement with a
blank.

Re-run it whenever you change an `ANTHROPIC_DEFAULT_*` variable. Bedrock isn't
probeable this way; declare it yourself. Either way `JEV_TIERS` wins:

```bash
export JEV_TIERS=sonnet,opus
```

A one-tier ladder is a working state, not an error — there's nothing to route
between, so the hook leaves every dispatch alone. So is no ladder at all: with
nothing confirmed serviceable, the only model known to work is the one the
session is already on, and every decision clamps to it.

**On Codex there is nothing to probe.** codex-cli caches the catalogue it was
served, keyed by a hash of the account, at `~/.codex/models_cache.json`, and
that file carries a `visibility` and a `supported_in_api` per model. So the
question `probe.py` has to spend a live request on for Claude is already
answered on disk, for free, and it is re-read on every decision rather than
whenever someone last remembered to probe. `JEV_TIERS` still wins. If the
catalogue can't be read at all the full ladder is assumed, exactly as an
unprobed Claude does; if it reads and lists none of the three rungs, that is a
measurement that came back empty and routing stops.

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

From inside Claude Code the mod spells the same thing `/jev-route <task>`,
defaulting to the tier the session is already on rather than asking you to
name it.

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

Five of ten on opus, but only four of those are holds — and every hold is
there for weak evidence, `tf-apply` at confidence 0.00, `binding-guard` at
0.06. The fifth, `design-epoch`, cleared the opus line outright at `need` 0.86.
That is the asymmetry working, not the router failing.

Jev is non-deterministic: identical input drifts by a few hundredths of a point
between runs, so your numbers will differ slightly from these. Tier assignments
near a threshold can flip.

Piped without `--table` it emits JSON suitable as `args` to
`routed-fanout.workflow.js`.

### See what it decided

```bash
/jev-explain        # from the mod, scoped to the session you are in
./explain.py        # from a shell, the most recently written log
```

A routing decision is invisible by construction. It is made before the agent
starts, the only trace it leaves is a model id on a transcript nobody opens,
and by the time the work comes back wrong the reasoning is gone. Worse, all
three front ends return early when the decision matches the tier already in
use — so a hold doing its job has always looked exactly like the router not
running.

Every decision is now appended to `cache/decisions/<session>.jsonl`, including
the ones that changed nothing, and `explain.py` reads them back:

```
session 018f2c91-...: 4 decisions, 6m ago to 40s ago

decision                     model    need  conf  advice
--------------------------------------------------------
1  review the retry guard    opus     0.71  0.48  clarify
2  gofmt the fixtures        haiku   -0.09  0.91  -
3  bump the pinned dep       opus     0.22  0.14  held from sonnet
4  design epoch rollover     opus     0.86  0.71  clarify

1xhaiku, 0xsonnet, 3xopus

./explain.py <n> for everything behind one of them
```

The table is the default view because a dispatch router decides many times in
one session, and a wrong decision is visible as the row that looks unlike its
neighbours — one bookkeeping task on opus, or six in a row held. A single
decision shown alone has nothing to look unlike. `./explain.py 3` then prints
everything behind one of them, in the same layout `./route.py` uses for a task
scored by hand: the task text, the gate that fired, the five signals.

```bash
./explain.py 3              # everything behind decision 3
./explain.py --last         # the most recent one
./explain.py --sessions     # every session with a log
./explain.py --session <id> # one that isn't the newest
./explain.py --json         # the records themselves
```

This is the other half of the `verify.py` join. `verify.py` reports what each
subagent actually ran on; the log reports what it was supposed to run on and
why, so a disagreement between the two localises to the engine rather than the
policy.

The log holds dispatches only — `/jev-route` prints its reasoning on the spot
and isn't recorded. It is pruned at seven days, task text is cut at 4,000
characters with the original length kept alongside, and `cache/` is
gitignored. `JEV_DECISIONS_DIR` moves it.

Each record (schema 2) also carries what a threshold will one day be fitted
on, rather than only what was concluded:

- `jev` — the raw answer: every level probability of `depth` and `breadth`,
  the three nouls, the version Jev says answered, the version requested, and a
  hash of the question set. A different gate can then be replayed over past
  dispatches instead of judged on new traffic alone.
- `dispatch` — the host's `tool_use_id` for the spawn, so an outcome found
  later joins back by id rather than by a label that repeats.
- `outcome` — `null` at decision time, because nothing then can know whether
  the work was good enough. It is the field calibration needs, and nothing
  fills it yet.
- `provider`, and `why` on any decision Jev could not make — a failure is
  logged by all three front ends, so "could not reach Jev" and "ran and held"
  no longer look the same.

The idea is [gargpratyush/jev-router](https://github.com/gargpratyush/jev-router)'s
— see [Related work](#related-work). Theirs explains one decision, because a
proxy that routes conversational turns has exactly one live at a time.

### Replay a different gate over the decisions already made

```bash
./replay.py                  # the newest session
./replay.py --all            # every session on disk
./replay.py --sweep          # how the answer moves with the bar
./explain.py --replay        # as an extra column beside what actually ran
```

Keeping the raw answer is only worth the bytes if something reads it back.
`replay.py` is that reader: it recomputes each logged dispatch under a
different downgrade gate and prints it beside the one that ran. It calls
nothing, so a gate can be judged on traffic that already happened rather than
on the next fortnight's.

The gate it replays is the one the current policy cannot express. Today
(**a**) collapses `depth` and `breadth` into a single weighted `need` and then
gates the downgrade on Jev's own `confidence` field — which the
JEV-as-a-Judge study pointedly does not use, its formula being unpublished.
Nowhere in that is a probability of the thing being decided.

(**b**) keeps the levels apart. Jev returns a probability per level, so
`(depth, breadth)` is a distribution over 25 outcomes; push each through the
same weights and the same two lines and it becomes a distribution over the
three rungs. A downgrade then needs **P(this rung is enough) ≥ τ** — a
probability of the decision itself, which is the quantity the paper's cascade
is actually validated on.

On the recorded answer the suite replays — depth `.04/.13/.68/.15/0`, breadth
`.03/.01/.02/.79/.15`, mechanical `.88`:

```
decision                       (a) took  (b)       P cheap   P mid  P dear  why (b)
-----------------------------------------------------------------------------------
1  rename DueAt                sonnet    sonnet      0.182   0.795   0.022  P(<= sonnet) 0.977 >= 0.7
```

Both reach sonnet, but only one of them says 0.977.

**The default τ is worth nothing.** It is a placeholder so the tool runs, not
a finding. Choosing it needs outcome labels, and `outcome` is still null on
every record. Use `--sweep`, which prints the disagreement count against the
bar: a τ whose neighbours give the same answer is one the data is insensitive
to, and a cliff is where a guessed threshold would have been doing the
deciding.

```bash
./replay.py --tau 0.85              # one bar for every transition
./replay.py --tau 'opus>haiku=0.95' # ... or one per transition
```

Per transition because the two are not the same bet: opus → sonnet keeps a
capable model, opus → haiku is the one that produces a confident wrong answer.

A record is skipped rather than guessed at when it cannot be replayed —
schema 1 predates the raw answer, a front end that could not reach Jev logged
no answer, and level probabilities that do not sum to 1 (±0.025), are missing
a level, or carry a label the questions never had are refused rather than
renormalised. Rungs are computed on the full ladder and only then clamped to
what that endpoint served, so a missing tier never moves a probability.

Two things it assumes, both untested. That `depth` and `breadth` are
independent — a deep task is plausibly also broad, and a positive correlation
would put more mass in the tails than this computes. And that `mechanical` can
stay a point value: a noul has a distribution too, and collapsing it to its
mean makes every rung probability here a conditional one, computed as though
`mechanical` were known exactly. Because it enters `need` as a constant it
shifts all 25 cells by the same amount, so it cannot reorder them — but a
sharp P(≤ sonnet) sitting just over τ may be sharp only because the spread in
one of its three inputs was dropped. Which direction that moves a real
decision is unmeasured.

**This changes no routing.** It is the measuring instrument, not the policy.

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
PASS  single-tier ladder routes nothing
PASS  never routes to an unavailable tier
PASS  routes to haiku when it is available
PASS  updatedInput preserves every original key
PASS  a decision survives a round trip
PASS  a long task is cut, and says how long it was
PASS  records accumulate in order
PASS  a session with a log can be found without being named
PASS  a torn line costs one record, not the log
PASS  a session id cannot escape the log dir
PASS  no session id, no log
PASS  a decision that changed nothing is still logged
PASS  a jev outage is logged, not swallowed
PASS  a stale log survives a concurrent prune

23/23 passed
```

The routing cases pin `JEV_TIERS` so they test the policy rather than your
Vertex grants — on a one-tier endpoint the hook correctly routes nothing, and
a red light that means "your project lacks sonnet" is testing the wrong thing.

The two routing cases assert *not opus* and *stays opus* rather than exact
tiers, since Jev drifts between calls and a task near a threshold can flip.

The log cases point `JEV_DECISIONS_DIR` at a temp directory, so a self-test
run never lands in the log you then read with `./explain.py`.

The Codex hook has its own suite, which needs no key at all — every case pins
the decision and runs the hook by path, so nothing is sent anywhere:

```bash
./selftest_codex.py                                      # 27/27
./selftest_route.py                                      # 61/61
```

`selftest_route.py` covers the plumbing under every front end, also offline:
that the Jev client turns a refused connection, a timeout or a non-JSON body
into one catchable error; that it asks for the pinned version; that the cache
is keyed on that version and never stores an answer missing a field; and that
one bad shard in a `route.py` fan-out costs that shard, not the whole run.
Its fixture is a real `jev-1.14.0` answer rather than an invented one.

The `replay.py` cases hold that fixture to arithmetic done by hand — the 25
cells of the outer product summing to `.1822/.7953/.0225` — so that moving a
weight or a line fails the suite rather than silently moving the counterfactual
too. Checked by mutation, not by assertion: each of the three weights and both
tier lines was moved in turn, and every one of the five reddens the suite.

The rest are the refusals, and they are most of the file: levels that do not
sum, a missing level, a label the questions never had, a level that is NaN,
negative, or a number sent as a string, an answer that is a list or a string
rather than an object, an answer scored under a question set that has since
been reworded, a rung the endpoint does not serve, and every malformed
`--tau` — a bar outside (0, 1], a tier name that is not on either ladder, and
a pair naming an upgrade or the same tier twice. A gate that silently ignored
a bar you typed would be worse than one that refused it.

Most of it is the shapes that must *not* route: both fork spellings and all
four ways `fork_turns` can mean "all", a V1 `fork_context` call, a parent on a
model off the ladder, malformed stdin, and a spawn of some other tool. Then
the ones that must: a well-formed `updatedInput` that preserves every other
key verbatim, an inherited effort replaced rather than left behind, the effort
key *absent* rather than null when a rung asks for none, and a hold still
reaching the decision log.

**2. The mod's logic, without Jev.** The same assertions against the event the
mod sees, with `route.py` stood in for, so no task leaves the machine and no
key is needed.

```bash
claude plugin test ~/jev-router
```

The harness is the one part of this repo that has never been run: these were
written on 2.1.267, where `claude plugin test` does not exist yet. Read a
failure before believing it.

The two pure functions behind the spawn line and the irreversible warning are
exported for this reason, and have been driven directly under `node` — so
what is unexercised is the mod's event plumbing, not the text those
assertions check.

**3. What your subagents actually ran on.** The hook rewriting `model` proves
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

On Codex the same job is `./verify_codex.py`. Codex writes one rollout per
spawned subagent — `~/.codex/sessions/<Y>/<M>/<D>/rollout-*.jsonl`, plain
JSONL — rather than nesting them under the parent, so the join is per file:
`session_meta.source.subagent.thread_spawn.agent_path`'s last segment is the
spawn's `task_name`, which is the label the hook recorded. Each row gets a
verdict — `ok`, `OVERRIDDEN`, `OFF LADDER`, `NO TURN`, `UNREADABLE`,
`UNROUTED` — and it exits non-zero when it cannot verify anything at all.

That last part is the point. The rollout format drifts: ten `cli_version`s
appear across the 442 rollouts this was built against, with fields appearing
and disappearing between them. A verifier that answered "all clear" because it
had failed to parse anything would be worse than none, so an unrecognised
rollout is reported as one, with the version named.

**Measured here, and worth repeating on your own account before trusting it:**
across those 442 rollouts, 116 of 116 spawns that named a `model` ran on
exactly that model — the mechanism works. But every one of those 116 asked for
`gpt-6-astra`, and only sol and astra have ever run on this machine. So what is
proven is that an *upgrade* is honoured. A downgrade to `gpt-6-luna` is
unexercised, and a silently ignored downgrade looks exactly like a dispatch
that worked. Run `./verify_codex.py` deliberately after your first few
dispatches: an `OVERRIDDEN` row is what that failure would look like.

**4. The A/B.** Run `verify.py` to capture the baseline, install the hook,
dispatch a mixed batch — something mechanical (*"run gofmt on X and commit"*)
alongside something deep (*"decide whether the guard should reject or coerce,
and implement it"*) — then re-run `verify.py`. The mechanical ones should come
back on a cheaper tier and the deep ones should still say opus.

If nothing changes, start with `./explain.py`, which separates the two cases
that used to look identical: an empty log means the front end never fired, and
a log full of holds means it fired and the gate refused. Then, in order: the
front end is actually loaded (`claude plugin list` for the mod, an entry in
`~/.claude/settings.json` for the hook, `./install.py --codex --check` for
Codex); `claude --debug` shows it firing; `./selftest.py` still passes; and
the dispatch didn't already name a `model`, which all three deliberately
respect.

On Codex add two of its own: the hook is only live once you have answered its
trust prompt, and a dispatch is only routed if it is a fresh spawn — a
full-history fork is left alone by design.

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

This used to say there was no mechanism for it anyway. That was wrong when it
was written, and in two ways.

The one I missed entirely: you do not need a hook. Put a loopback proxy on
`ANTHROPIC_BASE_URL`, expose a sentinel model in `/model`, and rewrite the
model on the way out.
[gargpratyush/jev-router](https://github.com/gargpratyush/jev-router) has
shipped exactly that since September 2026 — same Jev, same idea, every fresh
user turn, Claude Code and Codex both, on every CLI version rather than a
floor. Independent work, earlier than this repo, and it owns the npm name.

The one that arrived since: a mod's `turn.step` hook fires before each step of
a turn and may rewrite both `model` and `effort`, which is more than
`PreModelSwitch` (allow/deny/ask) or the `model` settings key (read once at
session start) ever offered.

So the thing is buildable twice over, and the measurements above are still the
reason this repo does not build it. A router that reads "yes, do it" as a 0.13
and downgrades the session is wrong on the one input that matters, and giving
it the session to be wrong about is worse than giving it a dispatch. The cost
argument survives too — the first request after any switch re-reads the whole
conversation uncached, so a router that flips tiers mid-turn pays for the
context it just saved on.

That is a disagreement about policy, not about capability, and it is the
honest way to state the difference between the two repos. Nobody has published
a measurement either way.

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
the session model; it just failed. Fail-open is a property of the hook's own
code paths, not of the decision it hands downstream.

The first fix was an inference — *the tier is pinned in the environment, so it
must work* — and it was wrong in the direction that costs you an agent. The
same project pinned `ANTHROPIC_DEFAULT_SONNET_MODEL` to `claude-sonnet-4@20250514`,
a model it had never been granted, so the tier the inference was surest of was
the second one to 404. Pinning a model and being able to call it are unrelated
facts, and only one of them is observable from the environment.

Hence `probe.py`: ping each alias's real target once, record the result, and
re-validate the record against the environment on every read. A tier that
isn't served is clamped *upward*, never down — a smaller saving and a live
agent. The general shape: a cheap decision that can't be served is worse than
no decision, and an environment is a thing to measure rather than infer.

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

## The advice half, mostly unbuilt

`decide()` returns `advice.confirm` and `advice.clarify`. The finding they come
from says the response to irreversibility is a confirmation step and the
response to ambiguity is a question back, and for a long time the router
delivered neither — they printed in `./route.py` output nobody was reading at
dispatch time.

**`confirm` now says so at spawn, in the mod.** A dispatch scoring
`irreversible > 0.60` gets a line of its own (one line, wrapped here):

```
deploy the staging bump -- irreversible 0.94: writes outside the working
tree. Tier unchanged; this is a warning, not a gate.
```

Printed even when the tier decision itself was worth no line, because a
dispatch the router agreed with is exactly the case where nothing was said
before, and "deploy this" agreeing with opus is still a deploy. That is also
the only case that earns the second sentence: where the router *did* re-tier
the dispatch, a line above this one has just said so, and "tier unchanged"
would contradict it. So the warning claims it only when it is the sole thing
printed.

It does **not** move the tier: a cheap model posting the right comment is
fine, and on a measured run an old `irreversible` floor sent "reformat this
comment I already wrote" straight to opus on that basis alone.

**Raising the confirmation posture is still blocked.** A mod sees `tool.check`
and could turn one into a prompt, so the shape is obvious: a dispatch that
scored `irreversible > 0.60` gets its writes confirmed, the rest of the session
doesn't. The blocker is identity and read/write: without a dispatch id on the
tool call you cannot tell which subagent it came from, and without a read/write
flag you cannot tell a lookup from a deploy. Lacking either, this prompts on
every tool call in the session, which nobody would keep installed for a week.

The documentation says both fields exist — `agentId` from v2.1.290,
`isReadOnly` from v2.1.304. Neither has been seen here: this machine is on
2.1.267, whose `sdk.d.ts` does not declare a `ToolCheckInput` at all, so what
those versions actually ship is unverified either way. **Read the declarations
for the version in front of you before writing this**, and treat a field the
docs promise as unavailable until you have seen it declared.

Neither hook gets this, and the reason is the same on both — it is not a Codex
quirk. Both carry a message to the user in `permissionDecisionReason`, and
both already send one on a dispatch they re-tier, so adding the warning there
would cost a few characters on a string already in flight. On Codex that
object additionally needs `permissionDecision: "allow"`, or it is discarded
along with the `updatedInput` it came with.

The half that cannot be had is the half that matters. When the decision
changes nothing, both hooks return silently with no output at all, and warning
there would mean emitting a `hookSpecificOutput` — on Codex, an outright
`allow` — for a spawn they would otherwise not have spoken about, turning a
router into a thing that approves tool calls. That trade is the wrong way
round, and it rules out precisely the case the mod's version exists for: a
deploy the router agreed with, where nothing else would be said.

This is the one capability gap in the table above that is about the seam
rather than the host. `$.ui.log` lets the mod speak without also deciding
something; a `PreToolUse` hook's only voice is attached to a verdict.

**`clarify` should reach the parent before the dispatch does.** A task reading
`ambiguous > 0.70` is one the subagent will guess at, and the parent is the
only thing that can resolve it — but it finds out three tool calls later, if at
all. Saying so at `agent.spawn` costs nothing and the API is already there.
This one is unbuilt for want of evidence rather than fields: 26 real prompts
say `ambiguous` reads high on nearly everything (median 0.80), and a warning
that fires on nearly everything is noise. It needs a threshold measured on
*dispatch* prompts, which is a measurement nobody has made yet.

## Cost and privacy

~$0.00003 per task routed ($0.042/Mtok input, output free). A 50-shard fan-out
is about a tenth of a cent. Answers are memoized for 7 days in `cache/`, keyed
on the task text, the question set *and* the Jev version requested — keying on
the text alone silently serves answers to questions you've since edited, or
from a scorer you've since moved off.

**The Jev version is pinned**, to `jev-1.14.0`, rather than `jev-latest`. The
thresholds were tuned against one version of the scorer, and "latest" moves
under them without saying so: the JEV-as-a-Judge study
([arXiv 2609.26550](https://arxiv.org/abs/2609.26550)) ran on 1.13.0, and by
November the API was answering as 1.14.0. The study measures nothing across
versions; what it does report is that a threshold does not transfer between
workloads, which is the same argument one step out. Moving is a decision:
`export JEV_MODEL=jev-1.15.0` (or `jev-latest`, knowingly). If the pinned
version is ever retired the router stops routing, fails open, and says why in
the decision log.

**Every routed task is POSTed to `api.typesafe.ai`**, carrying task text, file
paths, and whatever else is in the prompt. That is a real disclosure decision,
not a footnote. Think hard before pointing this at anything whose inputs can be
customer-derived.

On Codex that decision is sharper than it looks. Codex encrypts the `message`
of a spawn in its own rollouts — it is stored Fernet-wrapped on disk, which is
why `verify_codex.py` joins on `task_name` and never on the task text. This
router reads that same message before it is written and sends it out in the
clear. Nothing is bypassed and no file is decrypted; the hook simply sits
upstream of the encryption. But it does mean the content Codex chose to
protect at rest is content jev-router transmits, and that is worth deciding on
deliberately rather than discovering later.

The decision log keeps the first 4,000 characters of each of those tasks on
disk for seven days, in `cache/`, which is gitignored and goes nowhere. It is
a second copy of text already sent to Jev rather than a new exposure, but it
is a copy that outlives the session, so it is worth knowing about. `rm -rf
cache/decisions` ends it; `JEV_DECISIONS_DIR` moves it somewhere you control.

## Files

| | |
|---|---|
| `policy.py` | the questions, scoring, gates, cache. The only copy of the policy — everything imports it |
| `providers/` | what differs per host: the ladder, the slugs, the effort each rung takes, and how to find out which rungs are served. `claude.py` also holds the two transcript readers, which have no Codex counterpart |
| `jev.py` | zero-dependency client for `POST /v1/systemone`; backs off on 429/529, normalises confidence across all three primitives (`noul` returns none, so it computes `\|2p − 1\|`) |
| `hooks/register.js` | the mod: routes `agent.spawn`, and serves `/jev-probe`, `/jev-route` and `/jev-explain`. The recommended entry point, on 2.1.287+ |
| `hook_agent.py` | the Claude Code `PreToolUse` hook. The entry point below 2.1.287 |
| `hook_codex.py` | the Codex `PreToolUse` hook on `spawn_agent`. The entry point there |
| `codex/hooks.json` | the hook declaration, ready to copy into `~/.codex/` by hand |
| `install.py` | registers/removes the hook in `~/.claude/settings.json`, or in `~/.codex/hooks.json` with `--codex`. Backs up before every write |
| `probe.py` | pings each tier alias to find which ones the endpoint will actually serve |
| `route.py` | maps the policy over a list of tasks, concurrently, failing open to the default on a Jev error |
| `decisions.py` | the append-only decision log, one JSONL file per session. Written by all three front ends, including the decisions that changed nothing |
| `explain.py` | reads the log back: the session as a table, then everything behind any one row |
| `replay.py` | recomputes logged dispatches under a different downgrade gate, offline. Measures; routes nothing |
| `routed-fanout.workflow.js` | Claude Code workflow running one agent per routed shard |
| `selftest.py` | drives the Claude hook with synthetic events; asserts routing and every fail-open path |
| `selftest_codex.py` | the same for `hook_codex.py`, with the decision pinned so it needs no key |
| `selftest_route.py` | the Jev client, the cache, the fan-out and the decision log, offline, against a real recorded answer |
| `tests/register.test.ts` | the same assertions against the mod, with `route.py` stood in for. The harness needs 2.1.287+ and has never run; the two line builders it asserts on are exported and have been driven directly |
| `verify.py` | reads subagent transcripts and reports the model each one actually ran on |
| `verify_codex.py` | the same against Codex rollouts, joined to the decision log by `task_name`, one verdict per dispatch |
| `compare.py` / `shardcmp.py` | the measurement harnesses behind the numbers above |
| `experiments/context_dilution.py` | whether session context in the state helps the router or buries the task |

## Related work

[gargpratyush/jev-router](https://github.com/gargpratyush/jev-router) is an
independent project with the same name, the same scorer, and a different
answer. It came first — 16 September 2026, against 3 October for this one —
and it owns `jev-router` on npm. Neither repo is a fork of the other; I found
it after writing this one.

| | that one | this one |
|---|---|---|
| mechanism | loopback proxy on `ANTHROPIC_BASE_URL`, with a sentinel model in `/model` | a mod on `agent.spawn`, or a `PreToolUse` hook |
| scope | every fresh user turn | subagent dispatches only |
| CLIs | Claude Code and OpenAI Codex | Claude Code and OpenAI Codex |
| the decision | Jev picks the tier; code adjusts it | Jev answers five atomic questions; code combines them |
| version floor | none — it sits below the CLI | 2.1.287 for the mod, none for the hook |
| published measurements | none | [Findings](#findings) |

Use theirs if you want this working today on every turn, and on every CLI
version rather than above a floor. The reason this repo exists is the
*published measurements* row, and the reason it routes nothing but dispatches
is the *scope* row: scoring a conversational turn is the thing
[Findings](#findings) says does not work, measured over 26 real prompts.

Two places the difference is load-bearing rather than cosmetic:

- **A proxy cannot know the parent's tier.** Their proxy keys each sub-agent as
  its own conversation, which is right, but the model field on the request is
  the sentinel, so the session's real tier is erased before the proxy sees it.
  `src/proxy.mjs` falls back to `state.tier ?? "opus"`, and for a sub-agent the
  first turn *is* the first decision, so every dispatch is scored against opus
  whatever the session is on. That is the failure mode in "The default must be
  the session's own tier" above, and it is architectural rather than a
  slip — `e.parentModel` is a fact only a hook inside the CLI gets to see.
- **Catalogued is not granted.** Their tier availability comes from the
  account's model catalog. On the first-party API that is probably honest. It
  is the exact inference `probe.py` exists to replace, and the Vertex project
  that produced that finding would defeat it.

Neither is a reason to prefer this repo over that one today. They are the two
things worth stealing in the other direction.

Stolen in this direction: `/jev-explain`, theirs first. Their version renders
the current decision, which is the right shape for a proxy — routing
conversational turns, there is one decision live at a time and no history to
compare it against. A dispatch router makes several at once, so
[the version here](#see-what-it-decided) logs them all and leads with the
table. The thing worth copying was the observation that a router which cannot
account for itself is a router you cannot debug.

## License

Apache 2.0.
