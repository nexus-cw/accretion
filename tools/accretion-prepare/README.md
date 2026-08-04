# accretion-prepare

Ingest a community GGUF and repackage it as a DwarfStar-native,
streaming-optimized artifact. The product's front door: community
conversions come in whatever dialect their converter spoke; what comes out
loads on ds4 with **zero dialect-compat code paths firing** and is laid
out for O_DIRECT/cuFile expert streaming.

## Usage

```
python3 prepare.py <source.gguf> [--out DIR]
    [--expected-size N] [--expected-sha256 H] [--source-sha256]
    [--skip-normalize] [--skip-optimize]
```

Output: `<source>.accretion.gguf` + `<source>.accretion.manifest.json`.

Requires Python 3 + numpy. Streaming throughout — never loads the model
into RAM (64 MiB chunks; peak RSS is a few hundred MB).

## Stages

1. **FETCH** — v0 is verify-only (local file; size and optional sha256
   checked against expectations). Downloader is v1.
2. **NORMALIZE** — deepseek4 dialect normalization moved from ds4
   load-time to convert-time (transplant of nexus-cw/ds4 commits
   9c4b760, 99e7f1a, f7ec45f, 3106c3c, 0528b32):
   - derive canonical `deepseek4.*` metadata keys community converters
     omit (vocab_size, output_lora_rank/group_count, hash_layer_count,
     hyper_connection.count from tensor shapes/presence;
     sinkhorn_iterations / compress_rope_freq_base / hc epsilon /
     compress_ratios from shape constants)
   - rename dialect tensor names to the canonical llama.cpp/ds4 GGUF
     convention (missing `.weight`/`.bias` suffixes, `attn_kv_latent`,
     `attn_compress_*`, `indexer.compress_*`, `hc_head_*`)
   - convert dense BF16/Q6_K (ndim<=2) and F32 (ndim==2) tensors to F16
     (clamped to +-65504, RTNE — bit-identical to ds4's load-time
     conversion, so greedy outputs are unchanged)
3. **OPTIMIZE** — layout rewrite:
   - `general.alignment=4096`; every tensor offset padded to a 4KB
     boundary. Routed-expert slice sizes are multiples of 4096, so every
     (layer,expert) slice lands on a 4KB boundary — O_DIRECT reads with
     zero edge waste.
   - data order: all dense tensors first (source order), then per layer
     `gate/up/down` routed-expert tensors adjacent. Experts are
     sequential inside each tensor. A true cross-tensor (layer,expert)
     interleave is impossible in valid GGUF (a tensor's data must be one
     contiguous run); per-expert tensors / the appendable expert store is
     the v1 direction.
4. **MANIFEST** — JSON alongside the GGUF: `schema_version`, source
   identity (name/size/sha256/url), full transform log, per-
   (layer,expert,proj) offset table `{tensor, file, offset, len, dtype,
   location}` and a dense-tensor table. `location` defaults to the local
   file (`file://...`) — location-aware/appendable-store-ready.

## Manifest schema (v1) sample entries

```json
{"layer": 3, "expert": 17, "proj": "up",
 "tensor": "blk.3.ffn_up_exps.weight",
 "file": "model.accretion.gguf", "offset": 123456789, "len": 4456448,
 "dtype": "MXFP4", "location": "file:///data/gguf/model.accretion.gguf"}

{"name": "token_embd.weight", "dtype": "F16", "dims": [4096, 129280],
 "file": "model.accretion.gguf", "offset": 8192, "len": 1059061760,
 "location": "file:///data/gguf/model.accretion.gguf"}
```

## v0 limits

- deepseek4-family descriptor only (FLASH/PRO shape constants).
- FETCH does not download; the file must already be local.
- No sidecar/appendable expert store — the manifest is
  store-compatible, the store itself is future work.
- Runs from source (`python3 prepare.py`); prebuilt product packaging
  is later.

## Testing

`python3 test_prepare.py` builds a synthetic GGUF exercising every
normalize/optimize path and checks bytes, alignment, ordering, dtype
conversion (incl. Q6_K dequant vs a scalar reference), and manifest
offsets. `verify_manifest.py MANIFEST SOURCE [N]` spot-verifies expert
slices of a real artifact byte-for-byte against the source.
