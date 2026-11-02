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

const PARENT = 'claude-opus-5'

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

/** One element of the array route.py --json prints, with the keys it prints. */
function decision(over: Record<string, unknown> = {}) {
  return JSON.stringify([{
    id: 'spawn',
    task: STATE,
    model: 'sonnet',
    effort: 'medium',
    need: 0.41,
    conf: 0.82,
    suggested: 'sonnet',
    held: false,
    wanted: 'sonnet',
    tiers: ['haiku', 'sonnet', 'opus'],
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

  return { runs, spawned }
}

test('a dispatch is priced on its type, its description and its task', async ($, on) => {
  const { runs } = stub(on)

  await $.agent.spawn(dispatch())

  expect(runs).toHaveLength(1)
  expect(JSON.parse(runs[0].stdin ?? '')).toEqual([{ id: 'spawn', task: STATE }])
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
    stdout: JSON.stringify([{ id: 'spawn', task: STATE, model: 'opus', effort: 'high', why: 'jev error: 503' }]),
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

test('a task past the state limit is truncated', async ($, on) => {
  // Jev charges for input, and a dispatch prompt this long is a pathological
  // case rather than a task description.
  const { runs } = stub(on)

  await $.agent.spawn(dispatch({ prompt: 'x'.repeat(70_000) }))

  const sent = JSON.parse(runs[0].stdin ?? '')
  expect(sent[0].task).toHaveLength(60_000)
})

test('a missing description and type still make a scorable state', async ($, on) => {
  const { runs } = stub(on)

  await $.agent.spawn(dispatch({ description: '', subagentType: '' }))

  expect(JSON.parse(runs[0].stdin ?? '')[0].task).toBe(
    'Agent type: general-purpose\nDescription: \nTask:\nRename the Deadline field and fix every caller',
  )
})
