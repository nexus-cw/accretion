# The accretion web console (v0)

The box serves its own browser UI. Point a browser at the server root
(`http://<host>:8000/`, also `/console`) and you get a read-only dashboard —
a single self-contained HTML page embedded in the `ds4-server` binary. No
external assets, no CDN, no fonts, no build step: everything is inline, per
the consumer rules in [PRINCIPLES.md](../PRINCIPLES.md). The console is a thin
client over the same JSON APIs scripts use — there is no privileged
side-channel.

## What v0 shows

- **STATUS** — model name, quantization summary, and the honest-context pair
  (configured vs trained) rendered prominently. These facts are rendered
  server-side into the page, so STATUS works with JavaScript disabled.
  With JS enabled, the throughput reference (when the operator supplies
  `DS4_CAPS_THROUGHPUT_JSON`) and the expert-cache budget are added from
  `GET /v1/capabilities`.
- **ACTIVITY** — live view polling `GET /v1/activity` every 2.5 s: per-slot
  prefill progress bars (tokens done/total, measured t/s, derived ETA), a
  decode indicator with the generation token count, an expert-cache warmth
  bar (counted vs budget bytes, entries) with a "first sessions are slower
  until the cache warms" note below ~90% warmth, and KV anchor events
  ("restored N tokens in X ms" when a restore lands).
- **CONNECT** — copy-paste blocks for the claude CLI
  (`ANTHROPIC_BASE_URL`/`ANTHROPIC_MODEL`) and OpenAI-compatible clients
  (`base_url http://<host>:8000/v1`), with `<host>` filled from
  `window.location` when JS is available.

- **MODELS** — the model picker (v0.1). Lists every `*.gguf` the server can
  see (`GET /v1/models/available`): a **mode badge** (interactive / batch /
  unknown — see the sidecar convention below) with the recorded reference
  decode t/s beside it, name, size, quant summary, architecture,
  an `[active]` badge, and a switch button per loadable non-active model.
  A one-line explainer sits under the list: *interactive: resident, fast
  decode; batch: streamed, big-model capability*. The active model's mode
  badge also renders (server-side, JS-free) in **STATUS**.
  Switching is gated by the admin token (below); without one the picker is
  read-only with a note. A "prepare a new model" placeholder documents the
  CLI path (`accretion-prepare <source.gguf>`) — download-from-console is
  post-v0.1.

Everything except model switching is read-only; switching requires an
explicitly configured admin token and is disabled by default.

## `GET /v1/activity`

The missing "what is the box doing NOW" API. O(1), no GPU work, no hot-path
locks — safe to poll. All figures are lock-free reads of counters the serving
path already maintains, so **values may be slightly stale or torn across
fields**; treat the payload as a monitoring snapshot, not a transaction (the
payload carries this note verbatim).

```json
{
  "schema_version": 1,
  "note": "lock-free snapshot; values may be slightly stale",
  "slots": [
    {
      "id": 0,
      "state": "prefill",              // "idle" | "prefill" | "decode"
      "prefill": {                     // present when state == "prefill"
        "tokens_done": 4096,
        "tokens_total": 22000,
        "tokens_per_second": 63.10,    // measured live average, never hardcoded
        "eta_seconds": 283.8           // derived (total-done)/tps; omitted when tps==0
      }
      // when state == "decode": "generated_tokens": 128
    }
  ],
  "expert_cache": {
    "budget_bytes": 68719476736,
    "live_budget_bytes": 68719476736,  // CUDA builds only (same guard as capabilities);
    "counted_bytes": 61234567890,      // absent on CPU/Metal/ROCm builds
    "entries": 4021
  },
  "kv_events": {                       // this uptime
    "stores": 12,
    "chain_stores": 5,                 // layered anchor chain checkpoints (task #24)
    "prewarm_requests": 1,
    "restores": 3,
    "restored_tokens_total": 45000,
    "last_restore_tokens": 22000,
    "last_restore_ms": 1834.0
  }
}
```

`GET /activity` is an alias, mirroring `/capabilities`.

## `GET /v1/routing-stats`

Per-(layer,expert) routing-traffic telemetry (task #28), aggregated at
request time from persistent counters on the routed-MoE dispatch path.
Same conventions as `/v1/activity`: lock-free snapshot, no GPU work, safe
to poll; counters are cumulative across restarts (model-keyed file under
`~/.ds4/routing-stats/`). `GET /routing-stats` is an alias.

Sections: `totals` (selections, decode/prefill tokens, distinct keys, LRU
hits/misses/hit_rate), `persistence`, `top_experts` (hottest N by
selections, `DS4_ROUTING_TOPN` env, default 20), `per_layer` (selections,
unique experts, selection-distribution entropy in nats vs max,
high-router-entropy token counts), `router_entropy`, `coverage` (selection
share of the hottest 10/25/50% of keys), and — on CUDA streaming builds —
`advisor`: for budgets 50/55/60/63.6/70 GiB, `cache_experts` (K =
budget/expert_bytes) and `estimated_hit_rate_static`, the honestly-named
static popularity-skew approximation of "what would trimming the cache
cost".

```json
"advisor": {"expert_bytes": 13369344, "budgets": [
  {"budget_gib": 63.6, "cache_experts": 5107, "estimated_hit_rate_static": 0.9582}]}
```

The console's **ROUTING** section renders this: per-layer entropy bars,
top-10 hottest experts, and the advisor table, polled every 10 s.
`DS4_ROUTING_COUNTERS=0` disables the counters entirely (endpoint then
reports `enabled: false`). CPU builds serve the endpoint with zero counters
and no advisor.

## `POST /v1/prewarm`

Proactive prefill (task #24): body is a standard messages request in either
dialect. The server canonicalizes, restores matching KV anchors, prefills the
remainder, stores the layered anchor chain — exactly as a normal request —
but generates zero tokens and returns:

```json
{"restored_tokens": 0, "prefilled_tokens": 20177, "anchors_stored": 3, "wall_ms": 210557.2}
```

Idle-priority: refused with `503` unless the server is fully idle (never
queued behind generation); when admitted it runs on the idle prefill quantum.
This is the primitive for the ship/install/watch prewarming tiers.

## `GET /v1/models/available`

Scans the active model's directory plus `DS4_MODEL_DIRS` (colon-separated)
for `*.gguf`. Quant + architecture come from a bounded GGUF header peek,
cached per (path, size, mtime); `has_manifest` reports a sibling
`<name>.accretion.manifest.json`; `loadable` is a best-effort arch check
against what this binary supports (`yes` | `no` | `unknown` — honest
"unknown" when the header cannot be read).

```json
{
  "schema_version": 1,
  "active_path": "/data/models/ga.gguf",
  "select_enabled": false,
  "models": [
    {"name": "ga.gguf", "path": "/data/models/ga.gguf", "size_bytes": 191111111111,
     "quant": "Q4_K_M", "architecture": "glm-dsa", "active": true,
     "has_manifest": true, "has_sidecar": true, "loadable": "yes",
     "mode": "batch",
     "mode_reason": "sidecar flags use --ssd-streaming: experts stream from disk (big-model capability, slower decode)",
     "decode_tps_reference": 5.4}
  ]
}
```

### Per-model sidecar convention (`<model>.gguf.env`)

A model's proven launch profile lives in a plain `KEY=VALUE` env file next
to the gguf: `/path/to/model.gguf` → `/path/to/model.gguf.env`. Recognized
keys:

- `DS4_CTX`, `DS4_CACHE_BUDGET`, `DS4_EXTRA_FLAGS` — **applied on select**:
  a successful `POST /v1/models/select` copies these into the unit env file
  along with `DS4_MODEL`, so the restarted server runs the model's proven
  flags. With a sidecar present, `DS4_CACHE_BUDGET`/`DS4_EXTRA_FLAGS` are
  written even when absent from it (as empty) so streamed flags never leak
  onto a resident model; `DS4_CTX` is only overridden when the sidecar sets
  it non-empty. Without a sidecar, only `DS4_MODEL` changes.
- `DS4_DECODE_TPS_REFERENCE` — reported (as `decode_tps_reference`), never
  applied. A **recorded reference measurement**, not a live number.

### Mode semantics (interactive vs batch, first-class)

Derived from the sidecar, one honest rule: sidecar flags containing
`--ssd-streaming` → **batch** (experts stream from disk: big-model
capability, slower decode); a sidecar without it → **interactive**
(resident weights, fast decode); no sidecar, or an architecture this
binary cannot load → **unknown**. `mode_reason` carries the sentence-form
justification the console shows as the badge tooltip.

## `POST /v1/models/select`

The deliberate-teardown swap (one active model, explicit swap, no
auto-routing). Body: `{"path": "/abs/path/model.gguf"}`. The server
validates the path against the scanned list, rewrites `DS4_MODEL=` in the
env file that drives the unit (`DS4_ENV_FILE`, default
`/opt/accretion/etc/ds4-server.env` when writable), answers
`{"status":"swapping", ...}`, drains (stops accepting, finishes in-flight
generation), and exits with code 42; the systemd unit's
`Restart=on-failure` restarts it on the new selection. Both drain and total
swap timings are logged.

**Security gate**: requires `Authorization: Bearer <ACCRETION_ADMIN_TOKEN>`.
When `ACCRETION_ADMIN_TOKEN` is unset in the server environment, select is
**disabled** — `405` with an explanatory JSON body — and the console renders
the picker read-only with a "set admin token to enable switching" note.
Safe by default: a fresh install cannot have its model switched remotely.
Wrong token: `401`. Installs not driven by an env file (hand-managed units,
e.g. a service file with the model path inline): `409` — switch by hand
until the box migrates to the install layout.

Production robo-dog **is** env-file managed as of 2026-08-07: the unit's
`ExecStart` is parameterized from `/etc/accretion/ds4-server.env`
(`DS4_MODEL`/`DS4_CTX`/`$DS4_EXTRA_FLAGS`), `DS4_ENV_FILE` points at it,
`DS4_MODEL_DIRS` scans `/home/jacinta/src/ds4/gguf:/data/gguf`, and swaps
are exercised live (GA ↔ IQ2, per-model sidecars in place). The old
"hand-managed unit, switch by hand" caveat no longer applies there.

## Architecture swap (deepseek4 <-> inkling)

The model picker's deliberate-teardown swap (`POST /v1/models/select`,
above) was designed for swapping *models*; it extends to swapping
**architectures** — deepseek4 and inkling are two different server
binaries, and the box can hold one active model of each family.

**The wrapper**: systemd's `ExecStart` no longer runs a server binary
directly — it runs `/opt/accretion/bin/accretion-serve`, a thin POSIX
`sh` wrapper that reads `DS4_ARCH` from the unit's `EnvironmentFile` and
execs the matching binary:

```sh
case "${DS4_ARCH:-deepseek4}" in
  inkling)   exec /opt/accretion/bin/ds4-inkling-server -m "$DS4_MODEL" ... ;;
  *)         exec /opt/accretion/bin/ds4-server -m "$DS4_MODEL" ... ;;
esac
```

`DS4_ARCH` absent means `deepseek4` — existing installs need no env-file
change to keep working.

**Sidecar convention gains one key**: `DS4_ARCH` (`inkling` or
`deepseek4`) joins `DS4_CTX`/`DS4_CACHE_BUDGET`/`DS4_EXTRA_FLAGS` in the
per-model `<model>.gguf.env` sidecar, and is written on select exactly
like those: a successful `POST /v1/models/select` copies `DS4_MODEL`,
`DS4_ARCH`, `DS4_CTX`, `DS4_CACHE_BUDGET`, and `DS4_EXTRA_FLAGS` into the
unit env file, so the wrapper's next invocation execs the right binary
with the right flags.

**Loadable semantics across families**: `GET /v1/models/available`'s
`loadable` field is a best-effort check against what *this running
binary* supports — honestly, a ds4-server process cannot load an
inkling model, and vice versa. The unit env carries
`ACCRETION_ARCH_WRAPPER=1` to tell the server it is running under the
swap-capable wrapper, not launched bare: **a model of the OTHER family
reports `loadable:yes` only when `ACCRETION_ARCH_WRAPPER=1` is
present** — without it, cross-arch models correctly report `loadable:no`
(no wrapper means no restart-into-the-other-binary is possible, so
offering the switch would be dishonest).

**Swap choreography, step by step**:

1. `POST /v1/models/select {"path": "/data/models/inkling-small.gguf"}`
   lands on whichever server is currently running (say, `ds4-server`).
2. It validates the target against the scanned list and the sidecar,
   then rewrites `DS4_MODEL`/`DS4_ARCH`/`DS4_CTX`/`DS4_CACHE_BUDGET`/
   `DS4_EXTRA_FLAGS` into `DS4_ENV_FILE`.
3. It answers `{"status":"swapping", ...}` (`200`) immediately.
4. It drains: stops accepting new requests, finishes in-flight
   generation.
5. It exits with code `42`.
6. `Restart=on-failure` fires; systemd re-sources the (now-rewritten)
   `EnvironmentFile` and re-invokes `ExecStart`.
7. `ExecStart` is `accretion-serve`, which reads the new `DS4_ARCH` and
   execs `ds4-inkling-server` this time — a different binary than the
   one that was running a moment ago, on the same unit, same port.

Any other nonzero exit from either binary is a genuine crash, and
`Restart=on-failure` handles it exactly the same way — the wrapper does
not distinguish "planned swap" from "crash"; that distinction lives in
the exit code (`42` vs anything else) and the drain/log lines emitted
before it.

**Honesty note — `ds4-inkling-server`'s v1 API is a subset**: it serves
chat completions (including SSE streaming), `GET /v1/capabilities`,
`GET /v1/activity`, `GET /v1/models`, `GET /v1/models/available`, and
`POST /v1/models/select`. It does **not** serve `/v1/messages` (no
Anthropic dialect), `/v1/responses`, `POST /v1/prewarm`, or
`GET /v1/routing-stats` — those stay `ds4-server`-only for now. The
console degrades those sections honestly (missing, not silently empty)
when an inkling model is active.

## v1 roadmap

- prepare-pipeline progress (task #14 UX): live view of accretion-prepare
  stages with per-stage progress and ETA
- guarded settings mutation (explicit confirm, no silent writes)
- prewarm status + trigger (expert-cache prewarm from the console)
