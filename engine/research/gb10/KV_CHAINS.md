# KV anchor chains + prewarm (task #24)

Layered ("sectioned") KV disk anchors and a proactive prefill operation.
Builds on task #30 canonical deep-cold anchors (commit ee2722d) and task #16
pinning. ds4 commits: 47fca67 (chains + prewarm), + two follow-ups
(near-final turn boundaries; fully-idle prewarm guard).

## Premise verification

`ds4_kvstore_find_text_prefix` (ds4_kvstore.c) scans ALL stored entries and
picks the longest entry whose stored bytes sha-match a prefix of the incoming
canonical prompt text. Restore-then-continue-prefill from an arbitrary depth
works (task #30 ACC2, re-confirmed here at 5 depths). Therefore layered
anchors need NO new matching machinery: store a chain of checkpoints at
increasing depths, each keyed by the canonical text prefix up to its depth,
and an edit at depth D automatically restores the deepest link before D.

## Chain rules

During a request that qualifies for cold/deep-cold anchor storage (cached==0,
kv enabled, prompt >= min_tokens), before the terminal cold store the server
parks prefill at each chain depth and checkpoints there (incremental: the KV
state at each depth exists exactly once, while prefill is passing it).
Depths (`ds4_kvstore_chain_depths`):

- turn boundaries: canonical positions of each user/assistant marker token
  (prefix excludes the marker, same convention as the chat anchor);
- interval fills: every DS4_KV_CHAIN_INTERVAL tokens, covering long spans
  (the 50k-system-prompt / AGENTS.md-edit-in-the-middle case);
- filtered: >= min_tokens deep, >= 2048 apart, more than min_tokens (512)
  below the terminal store (turn boundaries just below final are kept — a
  new session sharing only earlier turns restores exactly there);
- evenly thinned to DS4_KV_CHAIN_MAX, always keeping the shallowest and
  deepest survivors.

Chain entries are normal `cold` store entries: existing keying, pinning,
budget/eviction all apply. Kill switch: interval 0 disables; greedy identity
verified unaffected (chain-on vs chain-off outputs byte-identical).

Env:
- `DS4_KV_CHAIN_INTERVAL` (default 8192, 0 disables)
- `DS4_KV_CHAIN_MAX` (default 8, 0 disables, max 64)

Known minor: if a chain depth coincides with the continued-store frontier the
progress callback can write the file first with reason=continued (same sha,
deduped); the entry then lacks the 2x anchor eviction prior. Harmless.

## Budget / GC

Measured on this model: ~16 MiB per 1k tokens (8192-token file = 130.4 MiB).
One 49.7k-prompt lineage (5 chain + 1 continued + terminal) = ~2.7 GiB.
A 128k lineage at defaults ≈ 8 chain links averaging ~60k depth + ~2 GiB
terminal ≈ ~10 GiB — most of the current 16 GiB budget. Eviction stays
LRU-by-effective-hits with the anchor prior; per-lineage count is capped at
store time by DS4_KV_CHAIN_MAX (deeper links are more expensive to recreate,
so no shallow-first sibling rule). **Recommendation: raise the kv budget to
32-48 GB (root NVMe has ~109 GB free) if chain use at >64k depth becomes
routine.** Service file not changed. Self-report: `/v1/activity` kv_events
gains `chain_stores` and `prewarm_requests`.

## POST /v1/prewarm

Body: a standard messages request in either dialect (Anthropic-shaped bodies
— top-level `system`/`anthropic_version` — are parsed with the Anthropic
parser first, falling back to the OpenAI chat parser; and vice versa).
The server runs canonicalization + anchor restore + prefill + chain/cold
storage EXACTLY as a normal request, generates zero tokens, and returns:

    {"restored_tokens":N,"prefilled_tokens":N,"anchors_stored":N,"wall_ms":F}

Guard: idle-priority. Refused with 503 unless the server is fully idle (no
queued and no active job) — with batched slots, a prewarm admitted beside an
active generation would run on the mixed 128-token quantum and crawl; when
admitted it always takes the existing idle prefill quantum path. This is the
primitive for the ship/install/watch prewarming tiers (#14/#27); the watcher
daemon is out of scope here.

## Layer-2 session handle: verdict SKIP (follow-up)

The anthropic surface emits `signature` on thinking blocks (ds4_server.c:7378)
but ignores it on parse. Wiring emit->parse for O(1) exact session identity
needs: parse-side capture in the anthropic content-block walker, a
signature->slot/anchor registry with invalidation on evict/restart, and
trust/validation semantics for client-supplied signatures. Honest estimate
>1 day; meanwhile find_text_prefix is O(entries) with a dir rescan per
lookup, fine at current store sizes (tens of entries). Documented as
follow-up work, not built.
