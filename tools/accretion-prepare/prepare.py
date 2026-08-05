#!/usr/bin/env python3
"""accretion-prepare: ingest a community GGUF and repackage it as a
DwarfStar-native, streaming-optimized artifact.

Stages (each skippable):
  FETCH      verify the local source file (size, optional sha256)
  NORMALIZE  deepseek4 dialect normalization at convert time:
             - derive canonical deepseek4.* metadata keys that community
               converters omit (transplant of ds4's load-time dialect
               compat: commits 9c4b760, f7ec45f)
             - rename dialect tensor names to canonical (99e7f1a)
             - convert dense BF16/Q6_K (ndim<=2) and F32 (ndim==2)
               tensors to F16 (3106c3c, 0528b32), clamped RTNE
  OPTIMIZE   expert-major data layout (routed-expert tensors grouped
             per layer, gate/up/down adjacent, experts sequential inside
             each tensor) + general.alignment=4096 with every tensor
             offset padded to a 4KB boundary (O_DIRECT/cuFile-native)
  MANIFEST   JSON manifest: source identity, transform log, per-
             (layer,expert) offset table, dense-tensor table; every
             entry carries a `location` field (local file for v0)

Family knowledge (metadata derivation rules, tensor-name aliases,
dense-type conversion policy, expert-tensor patterns, alignment) lives
in per-family descriptors under descriptors/ -- the primary arch seam
(task #26). See descriptors/__init__.py for the contract.

v0 limits: FETCH is verify-only (no downloader); pure-python streaming
(numpy for dtype conversion).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

import gguf
import descriptors

SCHEMA_VERSION = 1
CHUNK = 64 * 1024 * 1024


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


# ---------------------------------------------------------------------------
# dtype conversion (transplant of ds4.c 3106c3c + 0528b32, convert-time)
# ---------------------------------------------------------------------------

F16_MAX = 65504.0


def f32_to_f16_clamped(a_f32, stats):
    over = np.count_nonzero(a_f32 > F16_MAX) + np.count_nonzero(a_f32 < -F16_MAX)
    if over:
        stats['clamped'] += int(over)
        a_f32 = np.clip(a_f32, -F16_MAX, F16_MAX)
    return a_f32.astype(np.float16)  # RTNE, same as ds4 f32_to_f16


def bf16_bytes_to_f16(buf, stats):
    u = np.frombuffer(buf, dtype=np.uint16)
    f = (u.astype(np.uint32) << 16).view(np.float32)
    return f32_to_f16_clamped(f, stats).tobytes()


def f32_bytes_to_f16(buf, stats):
    f = np.frombuffer(buf, dtype=np.float32)
    return f32_to_f16_clamped(f, stats).tobytes()


def q6k_bytes_to_f16(buf, stats):
    """Dequant Q6_K blocks (256 elems / 210 bytes) to F16.
    Port of llama.cpp dequantize_row_q6_K (via ds4.c 3106c3c)."""
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 210)
    nb = raw.shape[0]
    ql = raw[:, 0:128]
    qh = raw[:, 128:192]
    sc = raw[:, 192:208].view(np.int8)
    d = raw[:, 208:210].copy().view(np.float16).astype(np.float32)  # (nb,1)
    y = np.empty((nb, 256), dtype=np.float32)
    for half in range(2):  # n = 0, 128
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        sch = sc[:, half * 8:(half + 1) * 8]
        l = np.arange(32)
        is_ = l // 16  # (32,)
        q1 = ((qlh[:, 0:32] & 0x0F) | (((qhh >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((qlh[:, 32:64] & 0x0F) | (((qhh >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((qlh[:, 0:32] >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((qlh[:, 32:64] >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.int32) - 32
        base = half * 128
        y[:, base + 0:base + 32] = d * sch[:, is_ + 0].astype(np.float32) * q1
        y[:, base + 32:base + 64] = d * sch[:, is_ + 2].astype(np.float32) * q2
        y[:, base + 64:base + 96] = d * sch[:, is_ + 4].astype(np.float32) * q3
        y[:, base + 96:base + 128] = d * sch[:, is_ + 6].astype(np.float32) * q4
    return f32_to_f16_clamped(y.reshape(-1), stats).tobytes()


# converter: ggml type name -> (fn, src granular bytes)
CONVERTERS = {
    'BF16': (bf16_bytes_to_f16, 2),
    'F32': (f32_bytes_to_f16, 4),
    'Q6_K': (q6k_bytes_to_f16, 210),
}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    sz = os.path.getsize(path)
    done = 0
    t0 = time.time()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            done += len(b)
            if done % (8 * CHUNK) == 0:
                el = time.time() - t0
                log('fetch: sha256 %d/%d MiB (%.0f MiB/s, ETA %.0fs)' %
                    (done >> 20, sz >> 20, (done >> 20) / max(el, 1e-9),
                     (sz - done) / max(done / el, 1)))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(prog='accretion-prepare')
    ap.add_argument('source', help='path to source GGUF (v0: local file only)')
    ap.add_argument('--out', default='.', help='output directory')
    ap.add_argument('--expected-size', type=int, default=None)
    ap.add_argument('--expected-sha256', default=None)
    ap.add_argument('--source-sha256', action='store_true',
                    help='compute source sha256 (extra full read pass)')
    ap.add_argument('--skip-normalize', action='store_true')
    ap.add_argument('--skip-optimize', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='parse header, select descriptor, plan normalize/'
                         'optimize and print a summary; write nothing')
    args = ap.parse_args()

    xlog = []
    t_start = time.time()

    # ---- FETCH (verify-only in v0) ----
    src = args.source
    if not os.path.isfile(src):
        raise SystemExit('source not found: %s (v0 has no downloader)' % src)
    src_size = os.path.getsize(src)
    if args.expected_size is not None and src_size != args.expected_size:
        raise SystemExit('size mismatch: %d != expected %d' % (src_size, args.expected_size))
    log('fetch: %s size=%d verified' % (src, src_size))
    src_sha = None
    if args.source_sha256 or args.expected_sha256:
        src_sha = sha256_file(src)
        log('fetch: sha256 %s' % src_sha)
        if args.expected_sha256 and src_sha != args.expected_sha256:
            raise SystemExit('sha256 mismatch')
    xlog.append({'stage': 'fetch', 'size': src_size, 'sha256': src_sha,
                 'expected_size_checked': args.expected_size is not None})

    g = gguf.read_header(src)
    log('parsed header: %d tensors, %d kv, data_start=%d, alignment=%d' %
        (len(g.tensors), len(g.metadata), g.data_start, g.alignment))

    # ---- descriptor selection (the arch seam) ----
    arch = g.metadata.get('general.architecture', (None, None))[1]
    desc = descriptors.select(g, log=log)
    log('descriptor: %s (architecture %r)' % (desc.name, arch))
    xlog.append({'stage': 'select', 'action': 'descriptor',
                 'descriptor': desc.name, 'architecture': arch})
    routed_re = desc.expert_pattern

    # ---- NORMALIZE ----
    stats = {'clamped': 0}
    renames = 0
    if not desc.normalizes() and not args.skip_normalize:
        if desc.name == 'generic':
            # graceful skip for unknown archs (ee2f023 behavior)
            log("normalize: architecture %r has no descriptor -- skipping "
                "dialect normalization (generic stages only)" % arch)
            xlog.append({'stage': 'normalize', 'action': 'skip_foreign_arch',
                         'architecture': arch})
        else:
            log('normalize: descriptor %s requires no dialect normalization'
                % desc.name)
            xlog.append({'stage': 'normalize',
                         'action': 'no_normalization_required',
                         'descriptor': desc.name, 'architecture': arch})
        args.skip_normalize = True
    if not args.skip_normalize:
        desc.derive_metadata(g, xlog)
        for t in g.tensors:
            cn = desc.canonical_name(t.name)
            if cn != t.name:
                xlog.append({'stage': 'normalize', 'action': 'rename_tensor',
                             'from': t.name, 'to': cn})
                t.name = cn
                renames += 1
        log('normalize: %d tensor renames' % renames)
        conv = [t for t in g.tensors if desc.dense_type_policy(t)]
        for t in conv:
            xlog.append({'stage': 'normalize', 'action': 'convert_dense',
                         'tensor': t.name, 'from': gguf.type_name(t.ggml_type), 'to': 'F16'})
        log('normalize: %d dense tensors will be converted to F16' % len(conv))
    else:
        conv = []
    conv_set = set(id(t) for t in conv)

    # ---- OPTIMIZE: layout plan ----
    alignment = 32 if args.skip_optimize else desc.alignment
    if not args.skip_optimize:
        g.metadata['general.alignment'] = (gguf.T_U32, alignment)
        routed = [t for t in g.tensors if routed_re.match(t.name)]
        dense = [t for t in g.tensors if not routed_re.match(t.name)]
        dense.sort(key=lambda t: t.offset)
        by_layer = {}
        for t in routed:
            m = routed_re.match(t.name)
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = t
        order = list(dense)
        for il in sorted(by_layer):
            for proj in ('gate', 'up', 'down'):
                if proj in by_layer[il]:
                    order.append(by_layer[il][proj])
        xlog.append({'stage': 'optimize', 'action': 'layout',
                     'order': 'dense-first then per-layer routed gate/up/down',
                     'alignment': alignment, 'n_dense': len(dense), 'n_routed': len(routed),
                     'note': 'expert slices inside each routed tensor are expert-sequential; '
                             'slice size is a multiple of 4096 so every (layer,expert) slice '
                             'starts on a 4KB boundary. True cross-tensor (layer,expert) '
                             'interleave needs per-expert tensors (appendable store, v1).'})
    else:
        order = sorted(g.tensors, key=lambda t: t.offset)

    # source offsets must be captured before rewriting
    src_off = {id(t): t.offset for t in g.tensors}
    src_type = {id(t): t.ggml_type for t in g.tensors}
    src_nbytes = {id(t): t.nbytes for t in g.tensors}

    # assign output types/sizes/offsets
    pos = 0
    for t in order:
        if id(t) in conv_set:
            t.ggml_type = gguf.TYPE_BY_NAME['F16']
            t.nbytes = t.n_elements * 2
        pos = (pos + alignment - 1) // alignment * alignment
        t.offset = pos
        pos += t.nbytes
    total_out_data = pos

    if args.dry_run:
        n_routed_plan = sum(1 for t in order if routed_re.match(t.name))
        log('dry-run: descriptor=%s arch=%r alignment=%d tensors=%d '
            '(routed %d, dense %d) renames=%d dense-convert=%d '
            'planned_data_bytes=%d -- nothing written'
            % (desc.name, arch, alignment, len(order), n_routed_plan,
               len(order) - n_routed_plan, renames, len(conv), total_out_data))
        return

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(args.out, base + '.accretion.gguf')
    man_path = os.path.join(args.out, base + '.accretion.manifest.json')

    # ---- write ----
    out_sha = hashlib.sha256()

    def preallocate(f, size):
        """Preallocate `size` bytes contiguously on f's fd (best effort).

        A single up-front posix_fallocate lets the filesystem reserve one
        (or few) contiguous extent(s) instead of growing the file in
        fragments as we stream ~150 GiB through it (task #31 rung 1).
        """
        try:
            os.posix_fallocate(f.fileno(), 0, size)
            log('preallocated %.1f GiB output (posix_fallocate)' % (size / 2**30))
            return True
        except (AttributeError, OSError) as e:
            log('WARNING: preallocation unavailable on target fs (%s); '
                'output may be fragmented' % e)
            return False

    class HashedFile:
        def __init__(self, f):
            self.f = f

        def write(self, b):
            out_sha.update(b)
            return self.f.write(b)

        def tell(self):
            return self.f.tell()

    log('writing %s (%.1f GiB tensor data)' % (out_path, total_out_data / 2**30))
    written = 0
    t0 = time.time()
    with open(src, 'rb') as fin, open(out_path, 'wb') as fout_raw:
        fout = HashedFile(fout_raw)
        data_start = gguf.write_header(fout, g.metadata, order, alignment)
        preallocate(fout_raw, data_start + total_out_data)
        for t in order:
            # pad to alignment
            gap = data_start + t.offset - fout.tell()
            assert 0 <= gap < alignment, (t.name, gap)
            if gap:
                fout.write(b'\x00' * gap)
            fin.seek(g.data_start + src_off[id(t)])
            n = src_nbytes[id(t)]
            if id(t) in conv_set:
                fn, gran = CONVERTERS[gguf.type_name(src_type[id(t)])]
                step = (CHUNK // gran) * gran
                left = n
                while left:
                    take = min(step, left)
                    buf = fin.read(take)
                    assert len(buf) == take
                    fout.write(fn(buf, stats))
                    left -= take
            else:
                left = n
                while left:
                    take = min(CHUNK, left)
                    buf = fin.read(take)
                    assert len(buf) == take
                    fout.write(buf)
                    left -= take
            written += n
            el = time.time() - t0
            rate = written / max(el, 1e-9)
            if written % (16 * CHUNK) < n or n > CHUNK:
                log('progress: %5.1f%% %d/%d MiB (%.0f MiB/s, ETA %.0fs) [%s]' %
                    (100.0 * written / src_size, written >> 20, src_size >> 20,
                     rate / 2**20, (src_size - written) / rate, t.name))

    out_size = os.path.getsize(out_path)
    if stats['clamped']:
        log('*** WARNING *** %d values clamped to +-65504 during dense '
            'conversion -- correctness red flag, investigate' % stats['clamped'])
    xlog.append({'stage': 'normalize', 'action': 'convert_dense_summary',
                 'tensors_converted': len(conv), 'values_clamped': stats['clamped']})

    # ---- MANIFEST ----
    n_layer = get_u32(g.metadata, '%s.block_count' % arch)
    n_expert = get_u32(g.metadata, '%s.expert_count' % arch)
    out_base = os.path.basename(out_path)
    experts = []
    for il in range(n_layer or 0):
        for proj in ('gate', 'up', 'down'):
            t = find_tensor(g, 'blk.%d.ffn_%s_exps.weight' % (il, proj))
            if not t:
                continue
            ne = t.dims[-1]
            slice_bytes = t.nbytes // ne
            for e in range(ne):
                experts.append({
                    'layer': il, 'expert': e, 'proj': proj,
                    'tensor': t.name, 'file': out_base,
                    'offset': data_start + t.offset + e * slice_bytes,
                    'len': slice_bytes, 'dtype': gguf.type_name(t.ggml_type),
                    'location': 'file://' + os.path.abspath(out_path),
                })
    dense_tab = []
    for t in order:
        if routed_re.match(t.name):
            continue
        dense_tab.append({
            'name': t.name, 'dtype': gguf.type_name(t.ggml_type),
            'dims': t.dims, 'file': out_base,
            'offset': data_start + t.offset, 'len': t.nbytes,
            'location': 'file://' + os.path.abspath(out_path),
        })
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'tool': 'accretion-prepare v0',
        'created_unix': int(time.time()),
        'source': {'name': os.path.basename(src), 'path': os.path.abspath(src),
                   'size': src_size, 'sha256': src_sha, 'url': None},
        'output': {'name': out_base, 'size': out_size,
                   'sha256': out_sha.hexdigest(), 'data_start': data_start,
                   'alignment': alignment},
        'model': {'architecture': arch, 'descriptor': desc.name,
                  'n_layer': n_layer, 'n_expert': n_expert},
        'transforms': xlog,
        'experts': experts,
        'dense_tensors': dense_tab,
    }
    with open(man_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    log('manifest: %s (%d expert entries, %d dense entries)' %
        (man_path, len(experts), len(dense_tab)))
    log('done in %.0fs: %s (%d bytes, sha256 %s)' %
        (time.time() - t_start, out_path, out_size, out_sha.hexdigest()))


if __name__ == '__main__':
    main()
