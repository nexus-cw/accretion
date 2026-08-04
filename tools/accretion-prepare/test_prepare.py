"""Unit test: build a tiny synthetic deepseek4-dialect GGUF exercising every
normalize/optimize path, run the pipeline, verify output + manifest."""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

import gguf

rng = np.random.default_rng(7)
d = tempfile.mkdtemp()
src = os.path.join(d, 'toy.gguf')

N_LAYER = 2
N_EXPERT = 4
tensors = []  # (name, dims, type_name, raw_bytes)


def add(name, dims, tname, raw):
    tensors.append([name, dims, tname, raw])


# routed experts: MXFP4 3D (per llama.cpp layout dims=[in, out, n_expert])
for il in range(N_LAYER):
    for proj in ('gate', 'down', 'up'):
        n_el = 32 * 4 * N_EXPERT
        nblocks = n_el // 32
        raw = rng.integers(0, 256, nblocks * 17, dtype=np.uint8).tobytes()
        add('blk.%d.ffn_%s_exps.weight' % (il, proj), [32, 4, N_EXPERT], 'MXFP4', raw)
# dense BF16 (2D) -> convert
bf = rng.standard_normal(8 * 5).astype(np.float32)
bf16 = (bf.view(np.uint32) >> 16).astype(np.uint16)
add('token_embd.weight', [8, 5], 'BF16', bf16.tobytes())
# dense F32 2D -> convert
f2 = rng.standard_normal(8 * 3).astype(np.float32)
add('blk.0.hc_attn_fn', [8, 3], 'F32', f2.tobytes())  # aliased name too
# dense F32 1D -> keep
f1 = rng.standard_normal(8).astype(np.float32)
add('blk.0.attn_sinks', [8], 'F32', f1.tobytes())  # aliased
# Q6_K 2D -> convert
q6 = rng.integers(0, 256, 210, dtype=np.uint8)
q6[192:208] = rng.integers(0, 30, 16)  # sane scales
q6[208:210] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
add('blk.0.attn_q_a.weight', [256], 'Q6_K', q6.tobytes())
# I32 keep
i32 = rng.integers(0, 100, 6 * 4, dtype=np.int32)
add('blk.0.ffn_gate_tid2eid', [6, 4], 'I32', i32.tobytes())
# vocab-check tensor path: use dims[1]==n_vocab==5
emb2 = rng.standard_normal(4 * 5).astype(np.float32)
add('output.weight', [4, 5], 'F32', emb2.tobytes())

md = {}
md['general.architecture'] = (gguf.T_STR, 'deepseek4')
md['deepseek4.block_count'] = (gguf.T_U32, N_LAYER)
md['deepseek4.embedding_length'] = (gguf.T_U32, 8)
md['deepseek4.attention.head_count'] = (gguf.T_U32, 4)
md['deepseek4.attention.key_length'] = (gguf.T_U32, 2)
md['deepseek4.expert_count'] = (gguf.T_U32, N_EXPERT)
md['tokenizer.ggml.tokens'] = (gguf.T_ARR, (gguf.T_STR, ['a', 'b', 'c', 'd', 'e']))

# write source with alignment 32
infos = []
pos = 0
for name, dims, tname, raw in tensors:
    pos = (pos + 31) // 32 * 32
    ti = gguf.TensorInfo(name, dims, gguf.TYPE_BY_NAME[tname], pos).finish()
    assert ti.nbytes == len(raw), (name, ti.nbytes, len(raw))
    infos.append(ti)
    pos += ti.nbytes
with open(src, 'wb') as f:
    ds = gguf.write_header(f, md, infos, 32)
    for ti, (name, dims, tname, raw) in zip(infos, tensors):
        f.seek(ds + ti.offset)
        f.write(raw)

out = os.path.join(d, 'out')
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'prepare.py'),
                    src, '--out', out, '--source-sha256'],
                   capture_output=True, text=True)
sys.stderr.write(r.stderr)
assert r.returncode == 0

og = gguf.read_header(os.path.join(out, 'toy.accretion.gguf'))
assert og.alignment == 4096
names = {t.name for t in og.tensors}
assert 'blk.0.hc_attn_fn.weight' in names and 'blk.0.attn_sinks.weight' in names
assert 'blk.0.ffn_gate_tid2eid.weight' in names
md2 = og.metadata
assert md2['deepseek4.vocab_size'][1] == 5
assert md2['deepseek4.hash_layer_count'][1] == 1
assert md2['deepseek4.hyper_connection.count'][1] == 1  # 8/8
# offsets aligned
for t in og.tensors:
    assert t.offset % 4096 == 0, t.name
# routed order: dense first then per-layer gate/up/down
routed = [t.name for t in og.tensors if 'exps' in t.name]
# order list in header order should equal offset order
offs = [t.offset for t in og.tensors]
assert offs == sorted(offs)
seq = [t.name for t in sorted(og.tensors, key=lambda x: x.offset) if 'exps' in t.name]
assert seq == ['blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight',
               'blk.1.ffn_gate_exps.weight', 'blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight'], seq

# byte identity for routed (no conversion)
with open(os.path.join(out, 'toy.accretion.gguf'), 'rb') as f:
    for t in og.tensors:
        if 'exps' not in t.name:
            continue
        f.seek(og.data_start + t.offset)
        got = f.read(t.nbytes)
        orig = dict((n, raw) for n, _, _, raw in tensors)[t.name]
        assert got == orig, t.name

# conversion correctness: token_embd BF16 -> F16
tt = [t for t in og.tensors if t.name == 'token_embd.weight'][0]
assert gguf.type_name(tt.ggml_type) == 'F16'
with open(os.path.join(out, 'toy.accretion.gguf'), 'rb') as f:
    f.seek(og.data_start + tt.offset)
    got = np.frombuffer(f.read(tt.nbytes), dtype=np.float16)
ref = np.clip(bf16.astype(np.uint32) .astype(np.uint32), 0, None)
ref = ((bf16.astype(np.uint32) << 16).view(np.float32)).astype(np.float16)
assert np.array_equal(got, ref)
# 1D F32 stays F32
ts = [t for t in og.tensors if t.name == 'blk.0.attn_sinks.weight'][0]
assert gguf.type_name(ts.ggml_type) == 'F32'
# Q6_K converted, spot-check against scalar dequant of first block
tq = [t for t in og.tensors if t.name == 'blk.0.attn_q_a.weight'][0]
assert gguf.type_name(tq.ggml_type) == 'F16'
with open(os.path.join(out, 'toy.accretion.gguf'), 'rb') as f:
    f.seek(og.data_start + tq.offset)
    gotq = np.frombuffer(f.read(tq.nbytes), dtype=np.float16)
x = q6.tobytes()
ql, qh, sc = x[0:128], x[128:192], np.frombuffer(x[192:208], dtype=np.int8)
dscale = np.frombuffer(x[208:210], dtype=np.float16)[0].astype(np.float32)
yref = np.zeros(256, dtype=np.float32)
p_ql, p_qh, p_sc = 0, 0, 0
for n in range(0, 256, 128):
    for l in range(32):
        is_ = l // 16
        q1 = ((ql[p_ql + l] & 0x0F) | (((qh[p_qh + l] >> 0) & 3) << 4)) - 32
        q2 = ((ql[p_ql + l + 32] & 0x0F) | (((qh[p_qh + l] >> 2) & 3) << 4)) - 32
        q3 = ((ql[p_ql + l] >> 4) | (((qh[p_qh + l] >> 4) & 3) << 4)) - 32
        q4 = ((ql[p_ql + l + 32] >> 4) | (((qh[p_qh + l] >> 6) & 3) << 4)) - 32
        yref[n + l + 0] = dscale * sc[p_sc + is_ + 0] * q1
        yref[n + l + 32] = dscale * sc[p_sc + is_ + 2] * q2
        yref[n + l + 64] = dscale * sc[p_sc + is_ + 4] * q3
        yref[n + l + 96] = dscale * sc[p_sc + is_ + 6] * q4
    p_ql += 64
    p_qh += 32
    p_sc += 8
assert np.array_equal(gotq, yref.astype(np.float16))

# manifest
man = json.load(open(os.path.join(out, 'toy.accretion.manifest.json')))
assert man['schema_version'] == 1
assert len(man['experts']) == N_LAYER * 3 * N_EXPERT
e0 = man['experts'][0]
assert e0['offset'] % 4096 == 0 or e0['expert'] > 0
assert man['source']['sha256']
with open(os.path.join(out, 'toy.accretion.gguf'), 'rb') as f:
    ent = [x for x in man['experts'] if x['layer'] == 1 and x['expert'] == 2 and x['proj'] == 'up'][0]
    f.seek(ent['offset'])
    got = f.read(ent['len'])
orig = dict((n, raw) for n, _, _, raw in tensors)['blk.1.ffn_up_exps.weight']
sb = len(orig) // N_EXPERT
assert got == orig[2 * sb:3 * sb]

print('ALL TESTS PASS')
