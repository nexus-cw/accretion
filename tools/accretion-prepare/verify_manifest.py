"""Verification ladder step 1: spot-verify manifest expert offsets.

Reads N experts' bytes at manifest offsets from the prepared file and
byte-compares against the same (layer,expert) slice extracted from the
SOURCE file (MXFP4 routed experts undergo no conversion, so bytes must be
identical). Also spot-verifies a few dense entries (converted dense
tensors are checked for correct length/dtype only; unconverted ones
byte-compared).
"""
import json
import random
import sys

import gguf


def main(manifest_path, source_path, n=5, seed=42):
    man = json.load(open(manifest_path))
    out_path = man['experts'][0]['location'].replace('file://', '')
    sg = gguf.read_header(source_path)
    src_t = {t.name: t for t in sg.tensors}

    rng = random.Random(seed)
    picks = rng.sample(man['experts'], n)
    fails = 0
    with open(out_path, 'rb') as fo, open(source_path, 'rb') as fs:
        for ent in picks:
            fo.seek(ent['offset'])
            got = fo.read(ent['len'])
            st = src_t[ent['tensor']]
            ne = st.dims[-1]
            slice_bytes = st.nbytes // ne
            assert slice_bytes == ent['len'], (ent, slice_bytes)
            fs.seek(sg.data_start + st.offset + ent['expert'] * slice_bytes)
            ref = fs.read(slice_bytes)
            ok = got == ref
            print('expert L%d E%d %s: %s (%d bytes, out_off=%d, aligned4k=%s)' %
                  (ent['layer'], ent['expert'], ent['proj'],
                   'IDENTICAL' if ok else 'MISMATCH', ent['len'], ent['offset'],
                   ent['offset'] % 4096 == 0))
            fails += 0 if ok else 1
        for ent in rng.sample(man['dense_tensors'], 3):
            st = src_t.get(ent['name'])
            if st is None:
                # renamed: find by manifest name unavailable in source; skip
                print('dense %s: renamed (skip byte check)' % ent['name'])
                continue
            if gguf.type_name(st.ggml_type) == ent['dtype']:
                fo.seek(ent['offset'])
                got = fo.read(ent['len'])
                fs.seek(sg.data_start + st.offset)
                ref = fs.read(st.nbytes)
                ok = got == ref
                print('dense %s: %s (%d bytes)' % (ent['name'], 'IDENTICAL' if ok else 'MISMATCH', ent['len']))
                fails += 0 if ok else 1
            else:
                exp_len = st.n_elements * 2  # converted to F16
                ok = ent['len'] == exp_len and ent['dtype'] == 'F16'
                print('dense %s: converted %s->%s len %s' %
                      (ent['name'], gguf.type_name(st.ggml_type), ent['dtype'],
                       'OK' if ok else 'BAD'))
                fails += 0 if ok else 1
    print('FAILS=%d' % fails)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
