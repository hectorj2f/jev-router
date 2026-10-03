export const meta = {
  name: 'routed-fanout',
  description: 'Fan out one agent per shard, each on the model Jev routed it to',
  phases: [{ title: 'Work', detail: 'one agent per shard, model set per shard' }],
}

// args is the output of ~/jev-router/route.py: an array of
//   {id, task, model, effort, need, conf, signals, suggested}
// Routing happens inline, before this script runs -- workflow scripts have no
// network access, so Jev cannot be called from in here.

const shards = args ?? []
if (!shards.length) return { error: 'no shards: pass route.py output as args' }

const split = shards.reduce((acc, s) => ({ ...acc, [s.model]: (acc[s.model] ?? 0) + 1 }), {})
log(`${shards.length} shards: ` + Object.entries(split).map(([m, n]) => `${n}×${m}`).join(', '))

const RESULT = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    outcome: { type: 'string', description: 'what you did or found' },
    blocked: { type: 'boolean', description: 'true if you could not finish' },
  },
  required: ['id', 'outcome', 'blocked'],
}

const out = await parallel(shards.map(s => () =>
  agent(
    `${s.task}\n\nReturn a JSON object with id "${s.id}".`,
    {
      label: `${s.id} [${s.model}]`,
      phase: 'Work',
      model: s.model,
      // Omitted rather than passed as null: Haiku 4.5 is absent from the
      // effort-level table, so route.py emits null for it, and naming the key
      // at all would be an error rather than a no-op.
      ...(s.effort ? { effort: s.effort } : {}),
      schema: RESULT,
    },
  )))

const done = out.filter(Boolean)
if (done.length < shards.length) {
  // Never let a partial fan-out read as a complete one.
  log(`${shards.length - done.length} shard(s) returned nothing`)
}
return { routed: split, results: done }
