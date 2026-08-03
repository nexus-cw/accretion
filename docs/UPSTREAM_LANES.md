# Upstream lanes

The sorting rule for every change:

- **Upstream lane** — anything that makes DwarfStar better at being DwarfStar: correctness, measured performance, model compatibility — work serving upstream's own goals. These go to antirez/ds4 first, as focused PRs, and come back to us via the engine sync.
- **Platform lane** — multi-user sovereign-serving genericity: API surfaces, capability reporting, KV/slot policy, multi-user scheduling, I/O backend selection, model management, routing. These live in Accretion and never burden the engine.

Structure additive over invasive: platform code wraps and drives the engine rather than patching through it, so tracking upstream stays mechanical.

Engine fixes are upstream-first. If we need a fix before it lands upstream, it goes on the nexus-cw/ds4 `platform` branch and returns here via `scripts/sync-engine.sh`.
