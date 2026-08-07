"""Inkling family descriptor (Thinking Machines; Small = 276B/A12B).

VERIFIED 2026-08-07 (task #35 S2): end-to-end prepare run on robo-dog
against unsloth/Inkling-Small-GGUF UD-IQ2_XXS (82.3GB merged from 3
splits): manifest = 30720 expert entries (40 MoE layers x 3 projections
x 256 experts, exactly as predicted from the header) + 840 dense
entries; 25-expert + dense spot-verify byte-IDENTICAL at 4096-aligned
offsets, FAILS=0. Routed expert slices: gate/up 2162688 B (IQ2_XXS),
down 3211264 B (IQ3_XXS). Note ds4 cannot SERVE this arch yet (task
#35 S3) -- prepare and serve are independent seams, by design.

Observed facts (Inkling-Small header):

  general.architecture = 'inkling', general.license = apache-2.0
  inkling.block_count = 42, inkling.dense_block_count = 2
    -> 2 dense FFN blocks (ffn_{gate,up,down}.weight, ffn 16384);
       40 MoE blocks.
  inkling.expert_count = 256, expert_used_count = 6 (top-6),
  expert_shared_count = 2 (blk.N.ffn_*_shexp.weight, dims [...,2],
    dense, NOT enumerated in the expert table),
  expert_gating_func = 2 (sigmoid) + blk.N.exp_probs_b.bias
    (DeepSeek-style bias gating), expert_weights_scale = 8.0,
  expert_feed_forward_length = 2048 -> routed expert =
    3 x (4096x2048) ~ 25M weights: a much finer streaming granule
    than deepseek4.
  Router: blk.N.ffn_gate_inp.weight dims [4096, 258] -- 256 routed
    + 2 shared columns; the router scores shared experts too. Plus
    per-block ffn_gscale scalar.
  Attention: GQA 32 heads / per-layer kv-head array (8s), head 128,
    q/k norms; hybrid local/global via sliding_window=512 + per-layer
    boolean sliding_window_pattern; positional scheme is relative
    attention (blk.N.attn_r [4096,512] + attn_rel_proj [512,16],
    d_rel=16, rel_extent 1024/512) + log scaling -- NOT RoPE. Per-block
    4-tap shortconv_{attn,mlp,k,v} tensors (LFM-style).
  Multimodal: image/audio live in a separate mmproj GGUF; the text
    model artifact contains no vision/audio tensors, so prepare sees
    pure text-model structure.
  tokenizer.ggml.model = gpt2, pre = 'inkling'; context_length 2^20.

The unsloth GGUF uses canonical llama.cpp names and carries complete
inkling.* metadata: NO dialect normalization -- no aliases, no
metadata derivation, no dense-type conversion. The descriptor's job is
expert enumeration for the manifest (standard ffn_*_exps pattern; the
2 dense blocks simply have no match) and the 4096-byte alignment for
the OPTIMIZE stage. Expected manifest: 40 routed layers x 3
projections x 256 experts = 30720 entries.
"""
from . import Descriptor


class InklingDescriptor(Descriptor):
    name = 'inkling'
    verified = True           # promoted by the 2026-08-07 prepare run (see docstring)
    alignment = 4096
    tensor_aliases = {}       # unsloth GGUF is already canonical

    def identify(self, g):
        return g.metadata.get('general.architecture', (None, None))[1] == 'inkling'

    def normalizes(self):
        # Complete inkling.* metadata + canonical names + quant dense
        # tensors kept as-is: the NORMALIZE stage has no work.
        return False

    # expert_pattern: base-class ffn_*_exps regex confirmed against the
    # shard-2 tensor table (ffn_{gate,up,down}_exps.weight, 256-expert
    # trailing dim). dense_type_policy: base class (never convert).
