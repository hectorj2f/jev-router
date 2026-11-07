/**
 * What the router does to a dispatch, with route.py stood in for.
 *
 * These are the assertions selftest.py makes about the hook, rewritten against
 * the event the mod sees instead of the stdin payload hook_agent.py read. The
 * policy itself is still tested in Python -- nothing here scores anything, and
 * `on('process.run')` answers in route.py's place so no task reaches Jev.
 *
 * UNRUN. `claude plugin test` arrived after the version on the machine these
 * were written on (2.1.267, mods need 2.1.287), so they have been typed
 * against the published declarations and never executed. Treat a failure here
 * as a bug in the test until you have read it.
 *
 * They assume a test's own `on` sits inside the plugin's, standing in for the
 * engine the way the docs describe. If that is backwards the mod's hook never
 * runs and every routing assertion fails at once, which is at least a loud way
 * to find out.
 */

import { test, expect } from 'claude-code/testing'
import type { AgentSpawnInput, On } from 'claude-code'
import { adviceLine, decisionLine } from '../hooks/register.js'

const PARENT = 'claude-opus-5'
const SESSION = 'sess_test'

// What the dispatch is called in the log and in the one-line report: the
// description, which is the only field written for a human to read.
const LABEL = 'rename Deadline'

const STATE = [
  'Agent type: general-purpose',
  'Description: rename Deadline',
  'Task:',
  'Rename the Deadline field and fix every caller',
].join('\n')

function dispatch(over: Partial<AgentSpawnInput> = {}): AgentSpawnInput {
  return {
    tool_use_id: 'toolu_01test',
    prompt: 'Rename the Deadline field and fix every caller',
    description: 'rename Deadline',
    subagentType: 'general-purpose',
    provider: { plugin: 'engine', tier: 'core' },
    parentModel: PARENT,
    background: false,
    fork: false,
    ...over,
  }
}

// The signals Jev returned for this task, and the `need` the policy computes
// from them: 0.60*(1.94/4) + 0.40*(3.02/4) - 0.25*0.88. Spelled out because a
// fixture whose numbers do not satisfy need_of is a shape route.py cannot
// emit, and a test built on one proves nothing about the real thing.
const SIGNALS = {
  depth: 1.94, breadth: 3.02, mechanical: 0.88,
  irreversible: 0.09, ambiguous: 0.66,
}
const NEED = 0.373

/** One element of the array route.py --json prints, with the keys it prints. */
function decision(over: Record<string, unknown> = {}) {
  return JSON.stringify([{
    id: LABEL,
    task: STATE,
    model: 'sonnet',
    effort: 'medium',
    need: NEED,
    conf: 0.82,
    suggested: 'sonnet',
    held: false,
    wanted: 'sonnet',
    tiers: ['haiku', 'sonnet', 'opus'],
    // `advice` and `signals` are on every successful decision route.py
    // prints, and the mod reads both: `advice.confirm` raises the irreversible
    // warning and `signals.irreversible` is the number it quotes.
    advice: { confirm: false, clarify: false },
    signals: SIGNALS,
    // So is `jev`, carrying the raw answer the decision log keeps. The mod
    // never reads it; it is here so the fixture is a shape route.py emits
    // rather than a subset someone assembled.
    jev: {
      model: 'jev-1.14.0',
      requested: 'jev-1.14.0',
      questions: '979664a4215b',
      cached: false,
      answers: {
        depth: { type: 'score', score: 1.94, confidence: 0.82 },
        breadth: { type: 'score', score: 3.02, confidence: 0.9 },
        mechanical: { type: 'noul', noul: 0.88 },
        irreversible: { type: 'noul', noul: 0.09 },
        ambiguous: { type: 'noul', noul: 0.66 },
      },
    },
    ...over,
  }])
}

/**
 * route.py replaced by a fixed answer, and a record of both sides.
 *
 * `spawned` is what the dispatch would actually run on, read where the engine
 * would read it, so a hook that declines and a hook that routes to the tier
 * the parent is already on stay distinguishable: declining leaves the resolved
 * parent id, routing writes the alias.
 */
function stub(on: On, out: { exitCode?: number; stdout?: string } = {}) {
  const runs: { argv: readonly string[]; stdin?: string }[] = []
  const spawned: string[] = []

  on('process.run', async ($, e) => {
    runs.push({ argv: e.argv, stdin: e.init?.stdin })
    return {
      exitCode: out.exitCode ?? 0,
      stdout: out.stdout ?? decision(),
      stderr: '',
    }
  })
  on('agent.spawn', async ($, e) => {
    spawned.push(e.model ?? e.parentModel)
    return { model: e.model ?? e.parentModel, agentId: 'agent_test' }
  })
  on('session.id', async () => SESSION)

  return { runs, spawned }
}

test('a dispatch is priced on its type, its description and its task', async ($, on) => {
  const { runs } = stub(on)

  await $.agent.spawn(dispatch())

  expect(runs).toHaveLength(1)
  expect(JSON.parse(runs[0].stdin ?? '')).toEqual(
    [{ id: LABEL, task: STATE, dispatch: 'toolu_01test' }])
})

test('the decision is logged against the session that made it', async ($, on) => {
  // route.py writes the log, not this file, and it only writes when it is told
  // which session to write under. Drop the flag and /jev-explain goes quiet
  // while everything else keeps working, which is the failure nobody notices.
  const { runs } = stub(on)

  await $.agent.spawn(dispatch())

  expect(runs[0].argv).toContain('--session')
  expect(runs[0].argv[runs[0].argv.indexOf('--session') + 1]).toBe(SESSION)
})

test("the parent's own tier is what the task is scored against", async ($, on) => {
  // Not a constant. Measured with a hardcoded "sonnet" in an opus session,
  // every sonnet-grade task read as "no change" and skipped the downgrade
  // gate, dropping 5 of 10 shards to sonnet, one of them at confidence 0.00.
  const { runs } = stub(on)

  await $.agent.spawn(dispatch())

  expect(runs[0].argv).toContain('--default')
  expect(runs[0].argv[runs[0].argv.indexOf('--default') + 1]).toBe('opus')
})

test('a cheaper tier the policy allows is applied', async ($, on) => {
  const { spawned } = stub(on)

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual(['sonnet'])
})

test('a decision that matches the parent changes nothing', async ($, on) => {
  const { spawned } = stub(on, { stdout: decision({ model: 'opus', wanted: 'opus', suggested: 'opus' }) })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('a hold is reported, not routed', async ($, on) => {
  // held means the score asked for something cheaper and the downgrade gate
  // refused it. The model stays put; the line says so.
  const { spawned } = stub(on, {
    stdout: decision({ model: 'opus', suggested: 'sonnet', held: true, wanted: 'sonnet', conf: 0.25 }),
  })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('an irreversible dispatch is warned about, not re-tiered', async ($, on) => {
  // The warning is a line, and the line is all it is. A cheap model posting
  // the right comment is fine; what the irreversibility asks for is a
  // confirmation step, and buying a bigger model answers none of it. On a
  // measured run an old `irreversible` floor sent "reformat this comment I
  // already wrote" straight to opus on this basis alone.
  //
  // Only `irreversible` moves from the base fixture: the other four signals
  // are what produce its `need`, and a deploy being irreversible says nothing
  // about how hard it is.
  const { spawned } = stub(on, {
    stdout: decision({
      advice: { confirm: true, clarify: false },
      signals: { ...SIGNALS, irreversible: 0.94 },
    }),
  })

  await $.agent.spawn(dispatch())

  // sonnet because the score earned it, not because of the advice: the
  // fixture's own model field is unchanged by the warning.
  expect(spawned).toEqual(['sonnet'])
})

test('the irreversible warning says so, and quotes the score', () => {
  // Asserted on the text, not through the hook. The only thing `agent.spawn`
  // exposes is the model the dispatch ran on, and not moving it is the whole
  // point of this line -- so a test driven through the hook passes equally
  // well if the warning is never printed at all.
  const plan = { advice: { confirm: true }, signals: { irreversible: 0.94 } }

  expect(adviceLine(plan, LABEL, true)).toBe(
    `${LABEL} -- irreversible 0.94: writes outside the working tree. ` +
    'Tier unchanged; this is a warning, not a gate.')
  expect(adviceLine({ advice: { confirm: false } }, LABEL, true)).toBeNull()
})

test('the warning does not claim the tier is unchanged under a re-tier', () => {
  // decisionLine prints first. An irreversible dispatch usually scores high on
  // `mechanical`, which is exactly what earns a downgrade, so "opus -> sonnet"
  // followed by "Tier unchanged" is the common pairing rather than a corner
  // case. The clause belongs only to the line that stands alone.
  const plan = {
    model: 'sonnet', need: NEED, conf: 0.82, wanted: 'sonnet',
    suggested: 'sonnet', held: false,
    advice: { confirm: true }, signals: { irreversible: 0.94 },
  }
  const decided = decisionLine(plan, 'opus', LABEL)

  expect(decided).toBe(`${LABEL} -- opus -> sonnet (need 0.37, conf 0.82)`)
  expect(adviceLine(plan, LABEL, !decided)).toBe(
    `${LABEL} -- irreversible 0.94: writes outside the working tree.`)
})

test('a plan route.py could not decide produces no line and no throw', () => {
  // The failure shape: route_one's except branch sets `why` and nothing else,
  // so neither builder may reach a number on it.
  const failed = { model: 'opus', effort: 'high', why: 'jev error: 503' }

  expect(decisionLine(failed, 'opus', LABEL)).toBe(
    `${LABEL} -- not routed: jev error: 503`)
  expect(adviceLine(failed, LABEL, false)).toBeNull()
})

test('a fork is never priced', async ($, on) => {
  // A fork inherits the parent and the engine ignores `model` on one, so the
  // call to Jev would buy an answer nobody reads.
  const { runs, spawned } = stub(on)

  await $.agent.spawn(dispatch({ fork: true }))

  expect(runs).toEqual([])
  expect(spawned).toEqual([PARENT])
})

test('a model already chosen is left alone', async ($, on) => {
  const { runs, spawned } = stub(on)

  await $.agent.spawn(dispatch({ model: 'haiku' }))

  expect(runs).toEqual([])
  expect(spawned).toEqual(['haiku'])
})

test('a parent off the ladder declines rather than guesses', async ($, on) => {
  // Fable is deliberately not a tier, and nothing in the policy says where it
  // sits relative to the three that are.
  const { runs, spawned } = stub(on)

  await $.agent.spawn(dispatch({ parentModel: 'claude-fable-5-1' }))

  expect(runs).toEqual([])
  expect(spawned).toEqual(['claude-fable-5-1'])
})

test('route.py failing leaves the dispatch alone', async ($, on) => {
  const { spawned } = stub(on, { exitCode: 1, stdout: '' })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('output that is not one decision leaves the dispatch alone', async ($, on) => {
  const { spawned } = stub(on, { stdout: 'Traceback (most recent call last):' })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('an empty result leaves the dispatch alone', async ($, on) => {
  const { spawned } = stub(on, { stdout: '[]' })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('a jev error reaches the dispatch as the tier it already had', async ($, on) => {
  // route.py catches a failed call and routes the shard to the default, with
  // `why` set. That is a report, not a decision to act on.
  const { spawned } = stub(on, {
    stdout: JSON.stringify([{ id: LABEL, task: STATE, model: 'opus', effort: 'high', why: 'jev error: 503' }]),
  })

  await $.agent.spawn(dispatch())

  expect(spawned).toEqual([PARENT])
})

test('a dispatch is never denied', async ($, on) => {
  // A refusal surfaces to the parent as an Agent tool error. A router has no
  // business failing a dispatch it was only asked to price.
  stub(on, { exitCode: 2, stdout: 'nope' })

  const result = await $.agent.spawn(dispatch())

  expect(result.model).toBe(PARENT)
})

test('a long task reaches route.py uncut', async ($, on) => {
  // The cap is policy.STATE_LIMIT, applied once in policy.ask_meta. This file
  // used to assert the mod did its own 60_000 cut -- a fourth copy of that
  // number, and the only one the de-duplication missed, because there is no
  // package.json here and nothing runs it.
  const { runs } = stub(on)

  await $.agent.spawn(dispatch({ prompt: 'x'.repeat(70_000) }))

  const sent = JSON.parse(runs[0].stdin ?? '')
  expect(sent[0].task).toContain('x'.repeat(70_000))
})

test('a missing description and type still make a scorable state', async ($, on) => {
  const { runs } = stub(on)

  await $.agent.spawn(dispatch({ description: '', subagentType: '' }))

  const sent = JSON.parse(runs[0].stdin ?? '')[0]
  expect(sent.task).toBe(
    'Agent type: general-purpose\nDescription: \nTask:\nRename the Deadline field and fix every caller',
  )
  // The far end of the label's fallback chain. A log row with no name in it is
  // still a row, and an empty one would sort and truncate strangely.
  expect(sent.id).toBe('dispatch')
})
