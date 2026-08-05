"""Kimi K3 family descriptor -- SKELETON, UNVERIFIED.

Exists so the third family lands in a file, not a code fork. Every
field below is from public arch knowledge only; NOTHING here has been
checked against a real K3 GGUF header. verified=False keeps this
descriptor off the selection path: a K3 artifact today falls to the
generic graceful-skip path, by design.

Public arch knowledge (all UNVERIFIED against an artifact):
  - 384 routed experts per MoE layer (vs deepseek4's 256-class scale);
    expert_used_count TODO.
  - MLA-family attention (latent KV compression, DeepSeek lineage) --
    tensor names likely attn_kv_a/_b-style, TODO confirm.
  - DeepSeek-style routing (sigmoid gating + expert bias, grouped
    top-k) -- likely ffn_gate_inp + exp_probs_b tensors, TODO confirm.
  - Community GGUFs likely use canonical llama.cpp names under a
    'kimi-k3' (or 'kimik3'? TODO) architecture string with complete
    metadata -> probably no dialect normalization, like laguna.
"""
from . import Descriptor


class KimiK3Descriptor(Descriptor):
    name = 'kimi_k3'
    verified = False          # SKELETON -- never selected
    alignment = 4096          # standard accretion O_DIRECT alignment

    # TODO(UNVERIFIED): exact general.architecture string.
    ARCH_STRINGS = ('kimi-k3', 'kimik3', 'kimi_k3')

    def identify(self, g):
        return g.metadata.get('general.architecture', (None, None))[1] in self.ARCH_STRINGS

    def normalizes(self):
        # TODO(UNVERIFIED): assumed no dialect normalization needed
        # (canonical community converter, like laguna). Revisit with a
        # real header.
        return False

    # expert_pattern: TODO(UNVERIFIED) assumed standard ffn_*_exps
    # (base class). 384 experts/layer expected in the manifest.
    # dense_type_policy: TODO(UNVERIFIED) base class (never convert).
    # tensor_aliases: TODO(UNVERIFIED) assumed empty.
