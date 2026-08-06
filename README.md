**Accretion** — the disk around DwarfStar. A consumer streaming platform that feeds [ds4](https://github.com/antirez/ds4) experts from NVMe on demand, so a 300B-class model serves real work from hardware you can actually buy.

## What it is

DwarfStar (ds4) is the engine; Accretion is the platform around it. The split is deliberate:

- Engine improvements — correctness fixes, measured performance wins, model compatibility — flow upstream to [antirez/ds4](https://github.com/antirez/ds4).
- Platform genericity — everything a multi-user serving deployment needs that the engine shouldn't carry — lives here.

## Status

**v0.1 reached** (release `v0.1.0`): one install command, browser console,
model picker (browse + switch the served model, admin-token gated), point
your harness at it. Seeded from the nexus-cw/ds4 `platform` branch.

Reference box: running in production on a GB10 (robo-dog) serving DeepSeek
V4 Flash MXFP4 at 5-6 tok/s from a 156GB GGUF with a 70GB expert cache.
Note: that box predates the install layout — its unit carries the model path
inline, so console model switching stays disabled there (safe default) until
it migrates to the env-file layout.

## Install

Consumers never build code — download a release, run the installer. Current
target: `gb10-arm64-cuda` (GB10 / DGX Spark). Other targets (generic arm64
CUDA, x86_64 CUDA, Apple/Metal) are planned; see `docs/RELEASING.md`.

```sh
# 1. Download the latest release tarball (and .sha256) from
#    https://github.com/nexus-cw/accretion/releases
tar xzf accretion-<version>-gb10-arm64-cuda.tar.gz
cd accretion-<version>-gb10-arm64-cuda

# 2. Install (idempotent; never overwrites an existing config)
sudo ./install.sh

# 3. Configure: set DS4_MODEL (and any tuning) then start
sudoedit /opt/accretion/etc/ds4-server.env
sudo systemctl enable --now ds4-server

# 4. Point your harness at it (Anthropic-compatible API)
export ANTHROPIC_BASE_URL=http://<host>:8000
export ANTHROPIC_API_KEY=none   # accepted but unused
claude   # or any Anthropic-API client
```

`sudo ./uninstall.sh` removes it (config kept unless `--purge`). Coming next:
a web console (#27) and a model downloader with setup-and-optimize UX, so
"pick a model" is a browser action rather than an env file.

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
