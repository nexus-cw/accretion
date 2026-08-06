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
  see (`GET /v1/models/available`): name, size, quant summary, architecture,
  an `[active]` badge, and a switch button per loadable non-active model.
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
     "has_manifest": true, "loadable": "yes"}
  ]
}
```

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

## v1 roadmap

- prepare-pipeline progress (task #14 UX): live view of accretion-prepare
  stages with per-stage progress and ETA
- guarded settings mutation (explicit confirm, no silent writes)
- prewarm status + trigger (expert-cache prewarm from the console)
