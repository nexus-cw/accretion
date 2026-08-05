"""Laguna S 2.1 family descriptor (poolside, 118B, 256x4.5B).

Grounded in the task #17 prepare run: header + prepared manifest of
laguna-s-2.1-Q4_K_M.gguf (robo-dog /data/gguf, prepared to
~/models/laguna-s21/). Observed facts:

  general.architecture = 'laguna'
  laguna.block_count = 48, laguna.leading_dense_block_count = 1
    -> layer 0 is a dense FFN (blk.0.ffn_{gate,up,down}.weight);
       layers 1..47 are MoE (47 routed layers).
  laguna.expert_count = 256, laguna.expert_used_count = 10 (top-10),
  laguna.expert_gating_func = 2, expert_weights_norm, scale 2.5.
  Shared expert per MoE layer: blk.N.ffn_*_shexp.weight (dense, NOT
    enumerated in the expert table).
  Router: blk.N.ffn_gate_inp.weight + blk.N.exp_probs_b.bias.
  Attention: GQA (48 heads / 8 KV) with alternating sliding-window
    (sliding_window=512, per-layer head_count array), q/k norms,
    attn_gate. Dual rope (freq_base 500000 full / 10000 swa, yarn).

The community GGUF already uses canonical llama.cpp names and carries
complete laguna.* metadata: NO dialect normalization is needed --
no aliases, no metadata derivation, no dense-type conversion. The
descriptor's job is expert enumeration for the manifest (the standard
ffn_*_exps pattern; layer 0 simply has no match) and the 4096-byte
alignment for the OPTIMIZE stage.
"""
from . import Descriptor


class LagunaS21Descriptor(Descriptor):
    name = 'laguna_s21'
    verified = True
    alignment = 4096
    tensor_aliases = {}   # community GGUF is already canonical

    def identify(self, g):
        return g.metadata.get('general.architecture', (None, None))[1] == 'laguna'

    def normalizes(self):
        # Complete metadata + canonical names + quantized (Q4_K_M/NVFP4)
        # dense tensors we keep as-is: the NORMALIZE stage has no work.
        return False

    # expert_pattern: base-class ffn_*_exps regex is correct for laguna;
    # 47 routed layers x 3 projections x 256 experts = 36096 manifest
    # entries (matches the prepared artifact).
    # dense_type_policy: base class (never convert) -- quant stays as-is.
