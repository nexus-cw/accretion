"""Descriptor-selection tests (task #26): deepseek4 artifact selects the
deepseek4 descriptor, laguna selects laguna_s21, an unknown arch falls to
the generic path with the graceful-skip behavior, and the kimi_k3
skeleton (verified=False) is never selected. Also exercises --dry-run."""
import os
import subprocess
import sys
import tempfile

import numpy as np

import gguf
import descriptors


class FakeHeader:
    def __init__(self, arch):
        self.metadata = {'general.architecture': (gguf.T_STR, arch)}
        self.tensors = []


# -- pure selection --
assert descriptors.select(FakeHeader('deepseek4')).name == 'deepseek4'
assert descriptors.select(FakeHeader('laguna')).name == 'laguna_s21'
assert descriptors.select(FakeHeader('qwen3moe')).name == 'generic'
# kimi skeleton matches identify but is unverified -> generic
k3 = descriptors.REGISTRY[-1]
assert k3.name == 'kimi_k3' and not k3.verified
assert k3.identify(FakeHeader('kimi-k3'))
assert descriptors.select(FakeHeader('kimi-k3')).name == 'generic'

# laguna descriptor: no normalization, standard expert pattern, 4096 align
lag = descriptors.select(FakeHeader('laguna'))
assert not lag.normalizes()
assert lag.alignment == 4096
m = lag.expert_pattern.match('blk.7.ffn_up_exps.weight')
assert m and m.group(1) == '7' and m.group(2) == 'up'
assert not lag.expert_pattern.match('blk.7.ffn_up_shexp.weight')
# lineage hook slot reserved
assert set(lag.lineage_hooks) == {'dynamic_expert_ids', 'parent_lineage',
                                  'manifest_extra'}


# -- end-to-end graceful skip + dry-run on a tiny synthetic file --
def toy_gguf(path, arch):
    rng = np.random.default_rng(3)
    md = {
        'general.architecture': (gguf.T_STR, arch),
        '%s.block_count' % arch: (gguf.T_U32, 1),
        '%s.expert_count' % arch: (gguf.T_U32, 2),
    }
    raws = []
    infos = []
    pos = 0
    for name, dims in (('token_embd.weight', [4, 3]),
                       ('blk.0.ffn_gate_exps.weight', [32, 2, 2])):
        if 'exps' in name:
            n_el = 32 * 2 * 2
            raw = rng.integers(0, 256, (n_el // 32) * 17, dtype=np.uint8).tobytes()
            tname = 'MXFP4'
        else:
            raw = rng.standard_normal(12).astype(np.float32).tobytes()
            tname = 'F32'
        pos = (pos + 31) // 32 * 32
        ti = gguf.TensorInfo(name, dims, gguf.TYPE_BY_NAME[tname], pos).finish()
        assert ti.nbytes == len(raw)
        infos.append(ti)
        raws.append(raw)
        pos += ti.nbytes
    with open(path, 'wb') as f:
        ds = gguf.write_header(f, md, infos, 32)
        for ti, raw in zip(infos, raws):
            f.seek(ds + ti.offset)
            f.write(raw)


d = tempfile.mkdtemp()
here = os.path.dirname(os.path.abspath(__file__))


def run_prepare(src, *extra):
    return subprocess.run([sys.executable, os.path.join(here, 'prepare.py'),
                           src, '--out', os.path.join(d, 'out')] + list(extra),
                          capture_output=True, text=True)


# unknown arch: generic descriptor + graceful skip, still prepares
src = os.path.join(d, 'toy_unknown.gguf')
toy_gguf(src, 'qwen3moe')
r = run_prepare(src)
sys.stderr.write(r.stderr)
assert r.returncode == 0
assert 'descriptor: generic' in r.stderr
assert 'skipping dialect normalization' in r.stderr

# laguna arch: laguna descriptor, no-normalization path, dry-run writes nothing
src = os.path.join(d, 'toy_laguna.gguf')
toy_gguf(src, 'laguna')
r = run_prepare(src, '--dry-run')
sys.stderr.write(r.stderr)
assert r.returncode == 0
assert 'descriptor: laguna_s21' in r.stderr
assert 'requires no dialect normalization' in r.stderr
assert 'dry-run' in r.stderr
assert not os.path.exists(os.path.join(d, 'out', 'toy_laguna.accretion.gguf'))

print('ALL DESCRIPTOR TESTS PASS')
