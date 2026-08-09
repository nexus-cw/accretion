# ARCH_SEAM — what adding a MoE family costs (task #26)

Ratified design: the **converter is the primary arch seam**. Per-family
knowledge for ingest lives in `tools/accretion-prepare/descriptors/`
(one module per family; contract documented in
`descriptors/__init__.py`). Engine graph modules exist only for
genuinely different architectures. Dense models are out of scope.

## The tiers

| Tier | What it buys | What it costs | Evidence |
|------|--------------|---------------|----------|
| T1 — descriptor only | Artifact **loads and prepares**: normalization (if any), expert-major 4096-aligned layout, (layer,expert) manifest. Serves **only if** the engine already has the arch. | One Python module, ~50–200 lines; hours. laguna_s21 needed ~40 lines because its GGUF is already canonical. | `descriptors/deepseek4.py` (full), `descriptors/laguna_s21.py` (pattern+alignment only) |
| T2 — engine graph module | The arch actually runs: attention flavor, router semantics, layer structure, per-backend kernels. | The dominant cost, and it is **per backend**. Upstream's `laguna-s2.1` branch: 17 commits, ~17.6k insertions over 32 core files (excluding eval fixtures). | breakdown below |
| T3 — streaming enablement | `--ssd-streaming` for the arch: routed experts fetched on demand through the CUDA expert cache instead of device-resident. | Moderate: a per-arch selected-load path + per-expert-pointer kernel variants; GLM precedent says weeks, not days. | file:line map below |

## T2 touch surface (upstream laguna-s2.1 branch, diffstat vs main)

`git diff --stat upstream/main...upstream/laguna-s2.1` (eval-response
fixtures excluded), categorized:

- **Core model/graph (`ds4.c` +5,994, `ds4.h` +14)** — family/variant
  enums (`DS4_MODEL_FAMILY_LAGUNA` ds4.c:481, `DS4_SHAPE_LAGUNA_S21`
  :669), per-layer head-count/SWA logic (`ds4_laguna_layer_is_swa`
  :1139), weight-mapping/validation (`weights_laguna_layer_has_required`
  :4915), the laguna graph path, and ~20 `ds4_session_is_laguna()`
  gates through sampling/agent/session code.
- **CUDA (`ds4_cuda.cu` +2,206, `ds4_gpu.h` +291)** — laguna MoE +
  attention kernels and the `ds4_gpu_laguna_*` extern API
  (`ds4_gpu_laguna_routed_shared_moe_one_tensor`,
  `..._qkvg_f16_tensor`, `..._attn_output_residual_f16_tensor`).
- **Metal (`ds4_metal.m` +2,792, `metal/laguna.metal` +1,182,
  `metal/moe.metal` +454, `metal/dense.metal` +335, dflash +145)**.
- **ROCm (`rocm/ds4_rocm_laguna.cuh` +870, `ds4_rocm_moe.cuh` +731,
  moe_launch +292, matmul +277, dflash +244, q8 +228, misc ~300)**.
- **Product surface** — `ds4_server.c` +406, `ds4_agent.c` +504,
  cli/help/README/download ~260.
- **Tests** — `tests/ds4_test.c` +930.

Takeaway: T2 is ~3 backends x (attention + MoE kernels) plus one graph
function plus session gates. A new family that reuses an existing
attention flavor (MLA) and router (DeepSeek-style) skips most of it.

## T3 — the Laguna streaming gap (file:line map)

The rejection: branch `ds4.c:59486` —
`"ds4: --ssd-streaming is not implemented for Laguna S 2.1 yet"` — an
open-time guard in the same validation block that rejects slicing/TP/
multi-GPU for Laguna (:59481–59500).

Why it is guarded, from the code:

1. **Whole-tensor dispatch.** Laguna's graph builds
   `ds4_gpu_laguna_moe_desc routed_moe/shared_moe` from *base pointers
   of whole device-resident `ffn_*_exps` tensors* (branch
   `ds4.c:49325–49365`) and calls
   `ds4_gpu_laguna_routed_shared_moe_one_tensor`
   (`ds4_cuda.cu:29574`; kernels `laguna_moe_down_*` :19483–19600,
   `laguna_routed_moe_tc_prefill` :26828, `laguna_routed_moe_q34_batch`
   :27185). There is no per-expert pointer indirection anywhere in the
   laguna path.
2. **The streaming machinery is per-expert.** Our CUDA cache/stream
   path (accretion `engine/ds4_cuda.cu`) works through
   `ds4_gpu_stream_expert_table` (:25206) consumed at decode
   (:25249–25403, persistent LRU `cuda_stream_expert_cache_peek`
   :628), entered via `ds4_gpu_stream_expert_cache_begin_selected_load`
   (:29773) — with an **arch-specific GLM variant**
   `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor`
   (:29119). That GLM fork is the precedent: each family gets its own
   selected-load entry matched to its kernel's expert addressing.
3. **C-side plumbing is deepseek4/glm-shaped.** Expert-bytes and
   locality helpers (`streaming_layer_routed_expert_bytes`
   `engine/ds4.c:4554`, `ds4_streaming_routed_expert_bytes` :4627),
   the hotlists (`ds4_streaming_hotlist*.inc`, included at :1358), and
   the `g->ssd_streaming` conditionals threaded through the deepseek4
   graph (:18504–:21800 band) have no laguna counterparts.

So the answer to "is it a per-arch dispatch function needing the
stream-cache calls, or deeper?": **one level deeper than a dispatch
shim, but not architectural.** The port is:

- (a) a per-expert-pointer variant of the laguna MoE kernels (or a
  gather stage feeding the existing one_tensor kernels);
- (b) a laguna `begin_selected_load` entry against the expert table
  (GLM precedent, ~300 LOC in ds4_cuda.cu);
- (c) laguna expert-bytes/locality plumbing + `g->ssd_streaming`
  threading in the laguna graph, and dropping the :59486 guard;
- (d) nothing for shared experts / layer-0 dense FFN — they are dense
  weights and stay resident (they are in the manifest dense table).

Favorable: laguna's router (`ffn_gate_inp` + `exp_probs_b`, top-10 of
256) already produces device-side selected-expert ids — exactly what
the selected-load path consumes; and accretion-prepare already emits
the 4096-aligned (layer,expert) manifest for laguna (36,096 entries,
verified). **Estimate: ~1.5–3k LOC in engine/ds4.c + engine/ds4_cuda.cu,
2–4 weeks single-dev including locality capture + QA, judged against
the GLM streaming port.** CUDA-only first; Metal/ROCm multiply it.

## The wakestone row

kethril-thel's future family needs from this seam exactly what the #14
manifest and #31 rung 3 already ratified, nothing more: (1)
**manifest-native loading** — the engine reads the per-(layer,expert)
offset table instead of GGUF tensor enumeration, which the `location`
field and dense-tensor table were designed for; (2) **dynamic expert
count** — the expert set can grow after prepare, so manifest expert
ids must be identities rather than dense 0..N-1 indices; (3) **lineage
ids** — an expert records which parent artifact it was accreted from.
The descriptor contract reserves the `lineage_hooks` slot
(`dynamic_expert_ids`, `parent_lineage`, `manifest_extra` — defined,
unused) so wakestone lands as descriptor fields plus manifest entries,
not a schema fork. No further design is invented here.

## The K3 row

Known (public, UNVERIFIED against an artifact — see
`descriptors/kimi_k3.py`): 384 routed experts/layer, MLA-family
attention (DeepSeek lineage), DeepSeek-style routing (sigmoid gating +
expert bias). Port-cost prediction: **T1 + small T2.** T1 is the
skeleton descriptor promoted to `verified=True` once a real header is
inspected (likely laguna-style: canonical names, no normalization).
T2 is small *if* the MLA and router reuse holds — the deepseek4 graph
already speaks both, so the engine work would be shape/variant tables
and weight mapping rather than new kernels; the laguna-scale 17.6k-line
cost was driven by a *new* attention flavor (GQA+SWA dual-rope) and new
per-backend MoE kernels, which K3 should not need. T3 for K3 would
then ride the existing deepseek4 streaming path largely for free —
the same reason its T2 is small. All of this is prediction, flagged
UNVERIFIED until a K3 GGUF header is read.

## The inkling row

Unlike the other rows above (predictions or upstream diffstats), inkling
is a completed data point: a **self-contained engine**, not a graph
module bolted onto `ds4.c`. On the ds4 fork's `inkling-port` branch, the
family lives in its own files rather than threading `ds4_session_is_*()`
gates through the shared graph:

- `ds4_inkling.c` (~1700 LOC) — model/graph, weight mapping, sampling
  path, own session structs.
- `ds4_inkling_cuda.cu` (~2000 LOC) — CUDA attention + MoE kernels,
  entirely separate from `ds4_gpu_laguna_*`/`ds4_gpu_stream_expert_*`.
- A thin server, `ds4-inkling-server` (~850 LOC) — the v1 API subset
  documented in `docs/CONSOLE.md` ("Architecture swap"): chat
  completions incl. SSE, capabilities, activity, models,
  models/available, select. No `/v1/messages`, no `/v1/responses`, no
  prewarm, no routing-stats.

**Estimate: ~4,550 LOC total** (1700 + 2000 + 850), CUDA-only, no Metal
or ROCm path yet — call it a **T-tier of its own**: cheaper than T2
(laguna's ~17.6k insertions across 3 backends threaded into the shared
graph) because nothing shares code with `ds4.c`/`ds4_cuda.cu` and there
is no per-backend multiplication yet, but far more than T1 (a ~50–200
line descriptor) because it is a real second engine, not ingest-only.
The self-contained-file structure is also why the arch-swap wrapper
(`build/accretion-serve`) works at all: two independent binaries, no
shared process state to reconcile on swap.

Pending operator step: `inkling-port` lives on the **ds4 fork**, not yet
subtree-synced into this repo's `engine/`. `scripts/build-release.sh`'s
`make -C engine ds4-inkling-server` step fails loudly (naming the
missing sync) until that subtree sync lands — see the fork's
`inkling-port` branch and this repo's engine-subtree sync convention.
