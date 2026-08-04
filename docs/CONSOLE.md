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

v0 is strictly read-only: no settings mutation from the browser.

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

## v1 roadmap

- prepare-pipeline progress (task #14 UX): live view of accretion-prepare
  stages with per-stage progress and ETA
- model picker (multiple prepared models on disk, switch which is served)
- guarded settings mutation (explicit confirm, no silent writes)
- prewarm status + trigger (expert-cache prewarm from the console)
