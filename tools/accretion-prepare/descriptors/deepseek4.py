"""deepseek4 family descriptor.

Verbatim extraction of the v0 hardcoded family knowledge from
prepare.py (metadata derivation = transplant of ds4.c dialect compat
9c4b760 + f7ec45f; tensor renames 99e7f1a; dense F16 conversion policy
3106c3c + 0528b32). Behavior must not change: this module is the same
logic relocated behind the descriptor contract.
"""
import re
import sys

import gguf

from . import Descriptor

# Shape constants for keys no tensor encodes, keyed by block_count
# (mirrors ds4.c DS4_SHAPE_FLASH / DS4_SHAPE_PRO dialect-compat tier).
DS4_SHAPES = {
    43: dict(name='deepseek-v4-flash', n_hc_sinkhorn_iter=20,
             compress_rope_freq_base=160000.0, hc_eps=1e-06,
             compress_ratio=lambda il: 0 if il < 2 else (4 if il % 2 == 0 else 128)),
    61: dict(name='deepseek-v4-pro', n_hc_sinkhorn_iter=20,
             compress_rope_freq_base=160000.0, hc_eps=1e-06,
             compress_ratio=lambda il: 128 if il < 2 else (4 if il % 2 == 0 else 128)),
}

# alias (community dialect) -> canonical (llama.cpp/ds4 GGUF convention).
# Per-layer names use {L}. Transplant of ds4.c commit 99e7f1a.
TENSOR_ALIASES = {
    'hc_head_base': 'output_hc_base.weight',
    'hc_head_fn': 'output_hc_fn.weight',
    'hc_head_scale': 'output_hc_scale.weight',
    'blk.{L}.hc_attn_fn': 'blk.{L}.hc_attn_fn.weight',
    'blk.{L}.hc_attn_scale': 'blk.{L}.hc_attn_scale.weight',
    'blk.{L}.hc_attn_base': 'blk.{L}.hc_attn_base.weight',
    'blk.{L}.hc_ffn_fn': 'blk.{L}.hc_ffn_fn.weight',
    'blk.{L}.hc_ffn_scale': 'blk.{L}.hc_ffn_scale.weight',
    'blk.{L}.hc_ffn_base': 'blk.{L}.hc_ffn_base.weight',
    'blk.{L}.attn_sinks': 'blk.{L}.attn_sinks.weight',
    'blk.{L}.attn_kv_latent.weight': 'blk.{L}.attn_kv.weight',
    'blk.{L}.exp_probs_b': 'blk.{L}.exp_probs_b.bias',
    'blk.{L}.ffn_gate_tid2eid': 'blk.{L}.ffn_gate_tid2eid.weight',
    'blk.{L}.attn_compress_ape': 'blk.{L}.attn_compressor_ape.weight',
    'blk.{L}.attn_compress_kv.weight': 'blk.{L}.attn_compressor_kv.weight',
    'blk.{L}.attn_compress_gate.weight': 'blk.{L}.attn_compressor_gate.weight',
    'blk.{L}.attn_compress_norm.weight': 'blk.{L}.attn_compressor_norm.weight',
    'blk.{L}.indexer.compress_ape': 'blk.{L}.indexer_compressor_ape.weight',
    'blk.{L}.indexer.compress_kv.weight': 'blk.{L}.indexer_compressor_kv.weight',
    'blk.{L}.indexer.compress_gate.weight': 'blk.{L}.indexer_compressor_gate.weight',
    'blk.{L}.indexer.compress_norm.weight': 'blk.{L}.indexer_compressor_norm.weight',
}


def log(msg):
    sys.stderr.write('accretion-prepare: %s\n' % msg)
    sys.stderr.flush()


def get_u32(md, key):
    return md[key][1] if key in md else None


def find_tensor(g, name):
    for t in g.tensors:
        if t.name == name:
            return t
    return None


class Deepseek4Descriptor(Descriptor):
    name = 'deepseek4'
    verified = True
    alignment = 4096
    tensor_aliases = TENSOR_ALIASES

    def identify(self, g):
        return g.metadata.get('general.architecture', (None, None))[1] == 'deepseek4'

    def normalizes(self):
        return True

    def derive_metadata(self, g, xlog):
        """Add canonical deepseek4.* keys the source omits, derived from tensor
        shapes/presence (preferred) or shape constants (fallback). Transplant of
        ds4.c deepseek4 dialect compat (9c4b760 + f7ec45f), run once at convert
        time so the output loads with zero dialect-compat lines."""
        md = g.metadata
        arch = md.get('general.architecture', (None, None))[1]
        if arch != 'deepseek4':
            raise SystemExit('deepseek4 descriptor requires general.architecture=deepseek4 (got %r)' % arch)
        n_layer = get_u32(md, 'deepseek4.block_count')
        n_embd = get_u32(md, 'deepseek4.embedding_length')
        n_head = get_u32(md, 'deepseek4.attention.head_count')
        n_head_dim = get_u32(md, 'deepseek4.attention.key_length')
        shape = DS4_SHAPES.get(n_layer)

        def add(key, vtype, val, how):
            if key in md:
                return
            md[key] = (vtype, val)
            xlog.append({'stage': 'normalize', 'action': 'derive_metadata',
                         'key': key, 'value': val if vtype != gguf.T_ARR else 'array(n=%d)' % len(val[1]),
                         'derivation': how})
            log('normalize: derived %s = %r (%s)' % (key, val if vtype != gguf.T_ARR else '<array>', how))

        # vocab_size from tokenizer token list, cross-checked vs embedding dims
        if 'deepseek4.vocab_size' not in md:
            toks = md.get('tokenizer.ggml.tokens')
            if toks:
                n_vocab = len(toks[1][1])
                for tn in ('token_embd.weight', 'output.weight'):
                    t = find_tensor(g, self.canonical_name(tn)) or find_tensor(g, tn)
                    if t and len(t.dims) >= 2 and t.dims[1] != n_vocab:
                        raise SystemExit('vocab_size derivation disagrees: tokens=%d vs %s dim1=%d'
                                         % (n_vocab, tn, t.dims[1]))
                add('deepseek4.vocab_size', gguf.T_U32, n_vocab,
                    'tokenizer.ggml.tokens length, cross-checked vs token_embd/output vocab dim')

        # output_lora_rank / output_group_count from attn_output_a/b dims
        if ('deepseek4.attention.output_lora_rank' not in md or
                'deepseek4.attention.output_group_count' not in md):
            ta = find_tensor(g, 'blk.0.attn_output_a.weight')
            tb = find_tensor(g, 'blk.0.attn_output_b.weight')
            if ta and tb and n_head and n_head_dim:
                prod = n_head * n_head_dim
                if ta.dims[0] and prod % ta.dims[0] == 0:
                    groups = prod // ta.dims[0]
                    if groups and tb.dims[0] % groups == 0:
                        add('deepseek4.attention.output_group_count', gguf.T_U32, groups,
                            'n_head*key_length / dim0(blk.0.attn_output_a.weight)')
                        add('deepseek4.attention.output_lora_rank', gguf.T_U32, tb.dims[0] // groups,
                            'dim0(blk.0.attn_output_b.weight) / output_group_count')

        # hash_layer_count = contiguous run of ffn_gate_tid2eid from layer 0
        if 'deepseek4.hash_layer_count' not in md:
            cnt = 0
            for il in range(n_layer):
                if not (find_tensor(g, 'blk.%d.ffn_gate_tid2eid.weight' % il) or
                        find_tensor(g, 'blk.%d.ffn_gate_tid2eid' % il)):
                    break
                cnt += 1
            add('deepseek4.hash_layer_count', gguf.T_U32, cnt,
                'count of present blk.<i>.ffn_gate_tid2eid tensors')

        # hyper_connection.count from hc_attn_fn dim0 / n_embd
        if 'deepseek4.hyper_connection.count' not in md:
            t = (find_tensor(g, 'blk.0.hc_attn_fn.weight') or find_tensor(g, 'blk.0.hc_attn_fn'))
            if t and n_embd and t.dims[0] % n_embd == 0:
                add('deepseek4.hyper_connection.count', gguf.T_U32, t.dims[0] // n_embd,
                    'dim0(blk.0.hc_attn_fn) / embedding_length')

        # pure shape constants (no tensor encodes them)
        if shape:
            add('deepseek4.hyper_connection.sinkhorn_iterations', gguf.T_U32,
                shape['n_hc_sinkhorn_iter'], 'shape constant (%s)' % shape['name'])
            add('deepseek4.attention.compress_rope_freq_base', gguf.T_F32,
                shape['compress_rope_freq_base'], 'shape constant (%s)' % shape['name'])
            add('deepseek4.hyper_connection.epsilon', gguf.T_F32,
                shape['hc_eps'], 'shape constant (%s)' % shape['name'])
            if 'deepseek4.attention.compress_ratios' not in md:
                ratios = [shape['compress_ratio'](il) for il in range(n_layer)]
                add('deepseek4.attention.compress_ratios', gguf.T_ARR,
                    (gguf.T_I32, ratios), 'shape constant pattern (%s)' % shape['name'])

    def dense_type_policy(self, t):
        """ds4.c tensor_is_dense_conversion_candidate, deepseek4 family."""
        tn = gguf.type_name(t.ggml_type)
        nd = len(t.dims)
        if nd > 2:
            return False
        if tn in ('BF16', 'Q6_K'):
            return True
        if tn == 'F32' and nd == 2:
            return True
        return False
