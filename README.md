**Accretion** — the disk around DwarfStar. A consumer streaming platform that feeds [ds4](https://github.com/antirez/ds4) experts from NVMe on demand, so a 300B-class model serves real work from hardware you can actually buy.

## What it is

DwarfStar (ds4) is the engine; Accretion is the platform around it. The split is deliberate:

- Engine improvements — correctness fixes, measured performance wins, model compatibility — flow upstream to [antirez/ds4](https://github.com/antirez/ds4).
- Platform genericity — everything a multi-user serving deployment needs that the engine shouldn't carry — lives here.

## Status

Early. Seeded from the nexus-cw/ds4 `platform` branch. Running in production on a GB10 serving DeepSeek V4 Flash MXFP4 at 5-6 tok/s from a 156GB GGUF with a 75GB expert cache.

## Architecture sketch

- `engine/` — ds4 with expert streaming + CUDA expert LRU (subtree of nexus-cw/ds4 `platform`).
- Platform layer roadmap:
  - Anthropic + OpenAI API surface
  - Honest capability endpoint (what the box can actually do, right now)
  - KV pinning + slot-affinity for large system prompts
  - Prefill-quantum interleaving for multi-user serving
  - Tiered I/O backend: fadvise / O_DIRECT / GDS, auto-selected
  - Model setup-and-optimize downloader
  - Go routing proxy

## Relationship to upstream

Engine fixes go to antirez/ds4 first — see PRs [647](https://github.com/antirez/ds4/pull/647), [659](https://github.com/antirez/ds4/pull/659), [662](https://github.com/antirez/ds4/pull/662), [664](https://github.com/antirez/ds4/pull/664) — and we track upstream main continuously. See `docs/UPSTREAM_LANES.md` for the sorting rule and `scripts/sync-engine.sh` for the sync mechanics.

## Attribution

The engine (`engine/`) is DwarfStar by Salvatore Sanfilippo (antirez), MIT licensed. The platform layer is copyright the Accretion contributors, also MIT. See LICENSE.
