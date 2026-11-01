/**
 * jev-router as a Claude Code mod: route each subagent dispatch to a model.
 *
 * Same policy as hook_agent.py, and still the single copy of it in policy.py --
 * this file shells out to route.py rather than reimplementing the scoring, so
 * compare.py and shardcmp.py keep measuring the thing that actually ships.
 *
 * What changes is where the facts come from. The PreToolUse hook had to
 * reconstruct the session's tier by reading the transcript, infer a fork from
 * the subagent type, and check whether the caller had already named a model.
 * `agent.spawn` hands over all three as fields of the event, and fires for
 * agent-team teammates too, which a matcher on `^Agent$` never saw.
 *
 * Needs Claude Code v2.1.287 or later; below that mods do not load at all and
 * hook_agent.py is the way in. The two are independent front ends onto one
 * policy -- run one or the other, not both (`./install.py --uninstall`).
 */

// Mirrors hook_agent.py. Jev charges for input, and a dispatch prompt past this
// is a pathological case rather than a task description.
const STATE_LIMIT = 60_000

// Mirrors policy.TIERS, and the order matters: this is a substring match
// against a resolved model id, the same way policy.session_tier reads one.
const TIERS = ['haiku', 'sonnet', 'opus']

// The dispatch waits on this hook, so the cap has to be ours: the engine's own
// ten-second budget for a hook excludes time spent inside $.process.run. A
// cached answer returns in milliseconds; a cold one is a round trip to Jev.
const ROUTE_TIMEOUT_MS = 6_000

// probe.py pings up to three endpoints, each with its own 30s ceiling, and on
// Vertex shells out to gcloud first.
const PROBE_TIMEOUT_MS = 120_000

/** The tier a resolved model id belongs to, or null for one off the ladder. */
function tierOf(model) {
  for (const tier of TIERS) if (String(model || '').includes(tier)) return tier
  return null
}

/**
 * One decision from route.py, or null if it could not produce one.
 *
 * Piped rather than passed as argv: route.py reads a JSON array on stdin, and
 * a 60k task description has no business on a command line.
 */
async function routeOne($, state, parent) {
  const run = await $.process.run(
    [`${$.plugin.root}/route.py`, '--default', parent, '--json'],
    { stdin: JSON.stringify([{ id: 'spawn', task: state }]), timeoutMs: ROUTE_TIMEOUT_MS },
  )
  if (run.exitCode !== 0) return null
  const routed = JSON.parse(run.stdout)
  return Array.isArray(routed) && routed.length === 1 ? routed[0] : null
}

/** The one line the decision is worth, or null when nothing was decided. */
function decisionLine(plan, parent, label) {
  // A Jev error reaches us as a routed shard on the default tier. Worth saying
  // out loud: otherwise a broken router and a quiet one look identical.
  if (plan.why) return `${label} -- not routed: ${plan.why}`

  const numbers = `need ${plan.need.toFixed(2)}, conf ${plan.conf.toFixed(2)}`
  if (plan.model !== parent) {
    const clamp = plan.wanted !== plan.suggested
      ? `, ${plan.wanted} is not served here` : ''
    return `${label} -- ${parent} -> ${plan.model} (${numbers}${clamp})`
  }
  // The holds are the asymmetry working rather than the router failing, and
  // they were invisible in the PreToolUse hook, which returned early on them.
  if (plan.held) {
    return `${label} -- held at ${parent}; ${plan.suggested} suggested, ` +
      `a downgrade needs mechanical > 0.75 or conf >= 0.70 (${numbers})`
  }
  return null
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'jev-probe',
      description: 'Measure which model tiers this endpoint will actually serve',
    })
    await $.command.register({
      name: 'jev-route',
      description: 'Score one task by hand and show the whole decision',
    })
    return next(e)
  })

  on('agent.spawn', async ($, e, next) => {
    // A fork always inherits the parent, and the engine ignores `model` on one.
    // Skipping here is about not paying Jev for an answer nobody reads.
    if (e.fork) return next(e)
    // An explicit choice, by the user or by the model that wrote the dispatch.
    if (e.model) return next(e)

    // The default MUST be the parent's own tier, not a constant. Measured with
    // a hardcoded "sonnet" in an opus session, every sonnet-grade task read as
    // "no change" and skipped the downgrade gate entirely, silently dropping 5
    // of 10 shards from opus to sonnet, one of them at confidence 0.00.
    //
    // `parentModel` is what `inherit` resolves to, which is exactly the fact
    // policy.session_tier used to dig out of the transcript. Off the ladder
    // (fable), decline rather than guess.
    const parent = tierOf(e.parentModel)
    if (!parent) return next(e)

    const state = [
      `Agent type: ${e.subagentType || 'general-purpose'}`,
      `Description: ${e.description || ''}`,
      `Task:\n${e.prompt || ''}`,
    ].join('\n').slice(0, STATE_LIMIT)

    // No .catch handler on this registration, deliberately. A hook that fails
    // before calling next is skipped and the spawn proceeds on the model it
    // would otherwise have used, which is already the behaviour we want; a
    // handler would add a concept and change nothing. The try is here because
    // a throw after a decision is made should also leave the dispatch alone.
    //
    // This is fail-open over our own code paths only. What protects the
    // dispatch from a tier the endpoint cannot serve is the ladder clamp in
    // policy.available_tiers, not anything here -- a 404 happens after the
    // hook has returned, where nothing can catch it.
    let plan = null
    try {
      plan = await routeOne($, state, parent)
    } catch {
      return next(e)
    }
    if (!plan) return next(e)

    const label = (e.description || e.subagentType || 'dispatch').slice(0, 48)
    const line = decisionLine(plan, parent, label)
    if (line) $.ui.log(line)
    // Never a deny: a refusal surfaces to the parent as an Agent tool error,
    // and a router has no business failing a dispatch it was only asked to
    // price. An unroutable task runs on the model it was already going to use.
    return plan.model === parent ? next(e) : next({ ...e, model: plan.model })
  })

  on('command.run', { command: 'jev-probe' }, async ($) => {
    // The reason this is a command and not a shell alias: probe.py has to run
    // where the hook runs. Claude Code resolves each tier alias through
    // ANTHROPIC_DEFAULT_<TIER>_MODEL and injects the result into the processes
    // it spawns, so a probe run from a plain terminal or the `!` prefix sees
    // none of them and reports every tier unpinned -- a fact about the probe's
    // environment, not about the endpoint. A child of this mod inherits the
    // host process's environment, so it sees what the router will see.
    const run = await $.process.run([`${$.plugin.root}/probe.py`],
      { timeoutMs: PROBE_TIMEOUT_MS })
    const out = `${run.stdout}${run.stderr}`.trim()
    return { text: out || `probe.py exited ${run.exitCode} and said nothing` }
  })

  on('command.run', { command: 'jev-route' }, async ($, e) => {
    const task = String(e.args || '').trim()
    if (!task) {
      return { text: 'Give it a task, as you would a subagent: ' +
        '/jev-route rename the Deadline field and fix every caller' }
    }
    // Default to the session's own tier for the same reason the hook does: the
    // gates are relative to it, so routing against anything else reads the
    // wrong half of the asymmetry.
    const parent = tierOf(await $.session.model()) || 'sonnet'
    const run = await $.process.run(
      [`${$.plugin.root}/route.py`, '--default', parent, task],
      { timeoutMs: ROUTE_TIMEOUT_MS })
    const out = `${run.stdout}${run.stderr}`.trim()
    return { text: out || `route.py exited ${run.exitCode} and said nothing` }
  })
}
